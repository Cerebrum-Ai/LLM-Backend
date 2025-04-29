from flask import Flask, request, jsonify
import ngrok
import os
import sys
import time
import threading
import signal
import requests
from dotenv import load_dotenv
from waitress import serve
import tempfile
import base64
import subprocess
from pathlib import Path
import json

# Import ML models - audio processing is just one of potentially many ML types
from models.audio.emotion.audio_processor import SimpleAudioAnalyzer
from models.typing.pattern.typing_processor import KeystrokeProcessor

load_dotenv()

NODE_HANDLER_URL = os.environ.get("NODE_HANDLER_URL")
HEARTBEAT_INTERVAL = 30
ngrok_auth_tokens_str = os.environ.get("NGROK_AUTH_TOKENS")
if not ngrok_auth_tokens_str:
    print("Error: NGROK_AUTH_TOKENS environment variable not set.")
    sys.exit(1)
ngrok_auth_tokens = ngrok_auth_tokens_str.split(',')
current_ngrok_url = None
shutdown_event = threading.Event()
i = 0

# Flask application for ML model service
app = Flask(__name__)

# Centralized registry for all ML models available in this service
class MLModels:
    # Dictionary to hold initialized model instances
    _instances = {}
    
    # Model registry - maps model type and name to initialization function
    _registry = {
        'audio': {
            'emotion': lambda: SimpleAudioAnalyzer.get_instance()
        },
        'typing': {
            'pattern': lambda: KeystrokeProcessor.get_instance()
        }
        # Add more model types and names here as they become available:
        # 'image': {
        #    'classification': lambda: ImageClassifier.get_instance()
        # },
        # 'text': {
        #    'sentiment': lambda: SentimentAnalyzer.get_instance()
        # }
    }
    
    @classmethod
    def initialize_all(cls):
        """Initialize all registered ML models"""
        print("Initializing all ML models...")
        
        # Check if FFmpeg is installed (needed for audio models)
        check_ffmpeg_installed()
        
        successful_inits = 0
        total_models = 0
        
        # Iterate through all registered model types and models
        for model_type, models in cls._registry.items():
            print(f"Initializing {model_type} models...")
            for model_name, init_func in models.items():
                total_models += 1
                try:
                    print(f"  - Initializing {model_type}/{model_name} model...")
                    model_instance = init_func()
                    cls._instances[f"{model_type}/{model_name}"] = model_instance
                    print(f"  {model_type}/{model_name} model initialized successfully")
                    successful_inits += 1
                except Exception as e:
                    print(f"  Error initializing {model_type}/{model_name} model: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        print(f"ML model initialization complete: {successful_inits}/{total_models} models initialized")
        return successful_inits > 0  # Return True if at least one model was initialized
    
    @classmethod
    def get_model(cls, model_type, model_name):
        """Get an initialized model by type and name"""
        model_key = f"{model_type}/{model_name}"
        return cls._instances.get(model_key)
    
    @classmethod
    def process(cls, data_type, model, data):
        """Process data with the specified model type"""
        model_instance = cls.get_model(data_type, model)
        if not model_instance:
            return {"error": f"Model {data_type}/{model} not found or not initialized"}
        
        # Route to appropriate processing function based on model type
        if data_type == 'audio':
            if model == 'emotion':
                return cls._process_audio_emotion(model_instance, data)
        elif data_type == 'typing':
            if model == 'pattern':
                return cls._process_typing_pattern(model_instance, data)
        
        # Add more model type processors as they become available
        
        return {"error": f"Processing for {data_type}/{model} not implemented"}
    
    @classmethod
    def _process_audio_emotion(cls, model_instance, audio_data):
        """Process audio data with emotion detection model"""
        try:
            return model_instance.analyze_audio(audio_data)
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "detected_emotion": "unknown"}

    @classmethod
    def _process_typing_pattern(cls, model_instance, typing_data):
        """Process typing data with pattern detection model"""
        try:
            # Print detailed debug information about the input data
            print(f"\n=== TYPING DATA DEBUG ===")
            print(f"Type of typing_data: {type(typing_data)}")
            
            # Handle the specific data structure from the test script
            # The test script sends data in format: {"url": {"keystrokes": [...], "text": "..."}, "data_type": "typing", "model": "pattern"}
            if isinstance(typing_data, dict):
                # First check if this is the complete request payload
                if 'url' in typing_data and 'data_type' in typing_data and typing_data['data_type'] == 'typing':
                    print("Found complete request payload, extracting typing data from 'url' field")
                    if isinstance(typing_data['url'], dict) and 'keystrokes' in typing_data['url']:
                        typing_data = typing_data['url']
                    else:
                        print(f"Warning: 'url' field does not contain expected typing data structure")
                
                print(f"Keys in typing_data: {list(typing_data.keys())}")
                
                if "keystrokes" in typing_data:
                    keystrokes = typing_data["keystrokes"]
                    print(f"Number of keystrokes: {len(keystrokes)}")
                    if keystrokes:
                        print(f"First keystroke: {keystrokes[0]}")
                        print(f"Keys in first keystroke: {list(keystrokes[0].keys())}")
                        
                        # Convert press_time/release_time to timeDown/timeUp format for compatibility
                        for keystroke in keystrokes:
                            if "press_time" in keystroke and "timeDown" not in keystroke:
                                keystroke["timeDown"] = keystroke["press_time"]
                            if "release_time" in keystroke and "timeUp" not in keystroke:
                                keystroke["timeUp"] = keystroke["release_time"]
                        print(f"After conversion, first keystroke: {keystrokes[0]}")
            else:
                print(f"typing_data is not a dictionary: {typing_data}")
            print(f"=== END DEBUG ===\n")
            
            return model_instance.analyze_typing(typing_data)
        except Exception as e:
            print(f"Error processing typing: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

def initialize_models():
    """Initialize all ML models"""
    return MLModels.initialize_all()

def check_ffmpeg_installed():
    """Check if ffmpeg is installed and accessible"""
    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("FFmpeg is installed and accessible")
            return True
        else:
            print("Warning: FFmpeg test command returned non-zero exit code")
            return False
    except Exception as e:
        print(f"Warning: FFmpeg is not installed or not in PATH: {str(e)}")
        print("Audio conversion functionality may not work properly")
        return False

# --- Node Registration, Heartbeat, Deregistration ---
def wait_for_node_handler(max_retries=10):
    retry_count = 0
    retry_delay = 2
    while retry_count < max_retries:
        try:
            print(f"Attempting to connect to node handler (attempt {retry_count + 1}/{max_retries})...")
            response = requests.get(f"{NODE_HANDLER_URL}/status", timeout=5)
            if response.status_code == 200:
                print("Successfully connected to node handler")
                return True
        except Exception as e:
            print(f"Error connecting to node handler: {str(e)}")
        print(f"Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 30)
        retry_count += 1
    print("Failed to connect to node handler after multiple attempts")
    return False

def register_with_node_handler():
    global current_ngrok_url
    try:
        if not wait_for_node_handler():
            print("Node handler not ready after multiple attempts")
            return False
        print(f"Registering with node handler at {NODE_HANDLER_URL}...")
        response = requests.post(
            f"{NODE_HANDLER_URL}/register",
            json={"url": current_ngrok_url, "type": "ml"},
            timeout=5
        )
        if response.status_code == 200:
            print("Successfully registered with node handler")
            return True
        else:
            print(f"Failed to register with node handler: {response.text}")
            return False
    except Exception as e:
        print(f"Error registering with node handler: {str(e)}")
        return False

def deregister_from_node_handler():
    global current_ngrok_url
    try:
        print(f"Deregistering from node handler...")
        requests.post(f"{NODE_HANDLER_URL}/deregister", json={"url": current_ngrok_url}, timeout=5)
    except Exception as e:
        print(f"Error deregistering: {str(e)}")

def handle_node_handler_disconnect():
    """Handle disconnection from node handler"""
    print("Node handler disconnected. Attempting to reconnect...")
    max_retries = 5
    retry_delay = 2  # Start with 2 seconds
    
    for attempt in range(max_retries):
        try:
            # Try to reconnect
            if register_with_node_handler():
                print("Successfully reconnected to node handler")
                return True
            
            # If registration failed, wait before retrying
            print(f"Reconnection attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30)  # Cap at 30 seconds
            
        except Exception as e:
            print(f"Error during reconnection attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)
    
    print("Failed to reconnect after multiple attempts")
    return False

def send_heartbeat():
    """Send periodic heartbeat to node handler"""
    global current_ngrok_url
    consecutive_failures = 0
    max_consecutive_failures = 1  # Respond quickly to disconnections
    reconnection_in_progress = False
    
    while not shutdown_event.is_set():
        try:
            if reconnection_in_progress:
                time.sleep(HEARTBEAT_INTERVAL)
                continue
                
            print("\nSending heartbeat to node handler...")
            response = requests.post(
                f"{NODE_HANDLER_URL}/heartbeat",
                json={"url": current_ngrok_url},
                timeout=5
            )
            if response.status_code != 200:
                print(f"Heartbeat not acknowledged: {response.text}")
            consecutive_failures = 0
        except Exception as e:
            print(f"Heartbeat error: {str(e)}")
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print("\n=== Node Handler Connection Lost ===")
                print("Error detected. Starting reconnection process...")
                reconnection_in_progress = True
                if handle_node_handler_disconnect():
                    consecutive_failures = 0
                    reconnection_in_progress = False
                    print("✓ Successfully reconnected to node handler")
        time.sleep(HEARTBEAT_INTERVAL)

def handle_shutdown(signum, frame):
    print("\nReceived shutdown signal. Cleaning up...")
    shutdown_event.set()
    time.sleep(2)
    try:
        ngrok.disconnect()
    except Exception:
        pass
    deregister_from_node_handler()
    sys.exit(0)

# --- Ngrok Logic ---
def run_ngrok():
    global current_ngrok_url, i
    listener = None
    try:
        ngrok.set_auth_token(ngrok_auth_tokens[i])
        time.sleep(2)
        listener = ngrok.forward(9000)
        current_ngrok_url = listener.url()
        print(f"ML Node ingress at: {current_ngrok_url}")
        if not register_with_node_handler():
            print("Registration failed. Retrying...")
            registration_thread = threading.Thread(target=registration_retry, daemon=True)
            registration_thread.start()
        heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
        heartbeat_thread.start()
        while not shutdown_event.is_set():
            time.sleep(1)
    except Exception as e:
        if "ERR_NGROK_108" in str(e):
            print("Ngrok session limit reached. Trying next authtoken...")
            if i < len(ngrok_auth_tokens) - 1:
                i += 1
                run_ngrok()
            else:
                print("All ngrok authtokens exhausted. Exiting.")
                return
        else:
            print(f"Ngrok error: {str(e)}")
    finally:
        print("Cleaning up ngrok...")
        if listener:
            try:
                ngrok.disconnect()
                print("Ngrok disconnected successfully")
                deregister_from_node_handler()
            except Exception as e:
                print(f"Error disconnecting ngrok: {str(e)}")

def registration_retry():
    while not shutdown_event.is_set():
        if register_with_node_handler():
            break
        time.sleep(10)

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# --- Flask Endpoints (existing ML endpoint remains) ---
@app.route('/')
def root():
    """Root endpoint for health checks"""
    return jsonify({
        'status': 'ok',
        'service': 'ml_models',
        'message': 'ML models service is running'
    })

@app.route('/ml/process', methods=['POST'])
def process_ml_data():
    """
    Accepts a JSON payload with:
      - url: str (URL to data)
      - data_type: str (one of: image, gait, audio, typing)
      - model: str (name of ML model to use)
    """
    data = request.get_json()
    url = data.get('url')
    data_type = data.get('data_type')
    model = data.get('model')

    # Validate input
    if not url or not data_type or not model:
        return jsonify({'error': 'Missing required fields (url, data_type, model)'}), 400
    if data_type not in ['image', 'gait', 'audio', 'typing']:
        return jsonify({'error': 'Invalid data_type'}), 400

    # Process based on data type
    result = {}
    
    try:
        # Check if this is a file URL or a data URL
        if data_type == 'audio':
            print(f"Processing audio data with model: {model}")
            
            try:
                # Fetch the audio data from the URL
                try:
                    audio_data = requests.get(url).content
                    
                    # Save to a temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_path = temp_file.name
                        temp_file.write(audio_data)
                    
                    try:
                        # Process with the appropriate model
                        print(f"Analyzing audio from {temp_path}")
                        result = MLModels.process('audio', model, temp_path)
                    finally:
                        # Clean up
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
                except requests.RequestException as e:
                    print(f"Error fetching audio from URL: {str(e)}")
                    result = {"error": f"Could not fetch audio from URL: {str(e)}", "detected_emotion": "unknown"}
            except Exception as e:
                print(f"Error processing audio: {str(e)}")
                import traceback
                traceback.print_exc()
                result = {"error": str(e), "detected_emotion": "unknown"}
        elif data_type == 'typing':
            print(f"Processing typing data with model: {model}")
            
            try:
                # Process typing data (either JSON string or direct object)
                try:
                    # Determine if data is direct object or JSON string
                    if isinstance(url, str) and url.startswith('{'):
                        # It's a JSON string
                        typing_data = json.loads(url)
                    elif isinstance(url, dict):
                        # Already a dict object
                        typing_data = url
                    else:
                        # Assume it's a URL with data
                        try:
                            response = requests.get(url)
                            typing_data = response.json()
                        except:
                            # Try as file path
                            if os.path.exists(url):
                                with open(url, 'r') as f:
                                    typing_data = json.load(f)
                            else:
                                raise ValueError(f"Invalid typing data source: {url}")
                    
                    # Debug the typing data structure
                    print(f"Received typing data in /ml/process: {json.dumps(typing_data)[:200]}...")
                    
                    # For typing data coming from the test script, we need to pass the entire request payload
                    # The test script sends data in format: {"url": {"keystrokes": [...], "text": "..."}, "data_type": "typing", "model": "pattern"}
                    if data_type == 'typing':
                        print(f"Processing typing data with special handling for test script format")
                        # We'll pass the entire request payload to the processor, which will handle the extraction
                    
                    # Process with the appropriate model
                    print(f"Analyzing typing data")
                    result = MLModels.process('typing', model, typing_data)
                except requests.RequestException as e:
                    print(f"Error fetching typing data: {str(e)}")
                    result = {"error": f"Could not fetch typing data: {str(e)}"}
                except json.JSONDecodeError as e:
                    print(f"Error parsing typing data JSON: {str(e)}")
                    result = {"error": f"Invalid JSON format: {str(e)}"}
            except Exception as e:
                print(f"Error processing typing: {str(e)}")
                import traceback
                traceback.print_exc()
                result = {"error": str(e)}
        else:
            # Other data type handling will be added here as models are implemented
            print(f"Received ML request for unimplemented data type: {data_type}/{model}")
            result = {
                'status': 'error',
                'message': f'Processing for {data_type}/{model} not implemented yet',
                'url': url,
                'data_type': data_type,
                'model': model
            }
    except Exception as e:
        print(f"Unexpected error in ML processing: {str(e)}")
        import traceback
        traceback.print_exc()
        result = {"error": str(e)}

    return jsonify(result), 200

@app.route('/deregister', methods=['POST'])
def handle_deregister():
    """Handle deregistration request from node handler (shutdown notice)"""
    global current_ngrok_url
    try:
        data = request.json
        if data and data.get('url') == current_ngrok_url:
            print("\n=== Node Handler Shutdown Notice ===")
            print("Received deregister request from node handler")
            print("Node handler is shutting down. Preparing for reconnection...")
            # Start reconnection in a background thread
            def reconnect_loop():
                print("\n=== Reconnection Process ===")
                print("Starting reconnection attempts...")
                max_retries = 5
                retry_delay = 2
                for attempt in range(max_retries):
                    try:
                        print(f"\nReconnection attempt {attempt + 1}/{max_retries}")
                        print(f"Trying to connect to node handler at {NODE_HANDLER_URL}...")
                        if register_with_node_handler():
                            print("Successfully reconnected to node handler")
                            return
                        print(f"Reconnection attempt {attempt + 1} failed")
                        print(f"Waiting {retry_delay} seconds before next attempt...")
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 30)
                    except Exception as e:
                        print(f"Error in background reconnection: {str(e)}")
                        print(f"Waiting {retry_delay} seconds before next attempt...")
                        time.sleep(retry_delay)
                print("Failed to reconnect after multiple attempts")
            reconnect_thread = threading.Thread(target=reconnect_loop, daemon=True)
            reconnect_thread.start()
            # Return 202 Accepted to indicate we're handling the request
            return jsonify({
                'status': 'accepted',
                'message': 'Node is preparing for reconnection'
            }), 202
        return jsonify({'error': 'Invalid request'}), 400
    except Exception as e:
        print(f"Error handling deregister request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize all ML models centrally
    if not initialize_models():
        print("Failed to initialize any ML models. Exiting.")
        sys.exit(1)
    
    flask_thread = threading.Thread(target=lambda: serve(app, host='0.0.0.0', port=9000, threads=4), daemon=True)
    flask_thread.start()
    # Wait for Flask to be ready
    def wait_for_flask(port, max_retries=5):
        import socket
        for j in range(max_retries):
            try:
                with socket.create_connection(("localhost", port), timeout=1.0):
                    return True
            except (ConnectionRefusedError, socket.timeout):
                print(f"Waiting for Flask to start (attempt {j+1}/{max_retries})...")
                time.sleep(2)
        return False
    if not wait_for_flask(9000):
        print("Flask app failed to start. Exiting.")
        sys.exit(1)
    print("Flask app is running.")
    ngrok_thread = threading.Thread(target=run_ngrok, daemon=True)
    print("Starting ngrok tunnel...")
    ngrok_thread.start()
    while not shutdown_event.is_set():
        time.sleep(1)