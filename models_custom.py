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
import numpy as np
import cv2
import torch
import shutil # Import shutil to find the localtunnel executable
import platform # Import platform to check the operating system
from flask import Flask, request, jsonify






# Import ML models - audio processing is just one of potentially many ML types
from models.audio.emotion.audio_processor import SimpleAudioAnalyzer
from models.typing.pattern.typing_processor import KeystrokeProcessor
from models.image.classification.medvit_classifier import MedViTClassifier
from models.image.ecg.integration import ECG_FM_Handler
from models.image.ecg.digitization import digitize_ecg_image
from models.health.diabetes import diabetes_predictor

load_dotenv()

NODE_HANDLER_URL = os.environ.get("NODE_HANDLER_URL")
HEARTBEAT_INTERVAL = 30

# Removed ngrok specific variables
# ngrok_auth_tokens_str = os.environ.get("NGROK_AUTH_TOKENS")
# if not ngrok_auth_tokens_str:
#     print("Error: NGROK_AUTH_TOKENS environment variable not set.")
#     sys.exit(1)
# ngrok_auth_tokens = ngrok_auth_tokens_str.split(',')
# current_ngrok_url = None
# i = 0

current_tunnel_url = None # Variable to store the public localtunnel URL
shutdown_event = threading.Event()
localtunnel_process = None # Global variable to store the localtunnel subprocess


# Preload all MedViT models at startup (only once)
MedViTClassifier.preload_all_models()

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
        },
        'image': {
            'classification': lambda: MedViTClassifier,
            'ecg': lambda: ECG_FM_Handler()
        },
        'health': {
            'diabetes': lambda: diabetes_predictor if diabetes_predictor is not None else None
        }
    }

    @classmethod
    def initialize_all(cls):
        """Initialize all registered ML models"""
        print("\n=== ML Model Initialization ===")
        print("Initializing all ML models...")

        # Check if FFmpeg is installed (needed for audio models)
        check_ffmpeg_installed()

        successful_inits = 0
        total_models = 0

        # Iterate through all registered model types and models
        for model_type, models in cls._registry.items():
            print(f"\nInitializing {model_type} models...")
            for model_name, init_func in models.items():
                total_models += 1
                try:
                    print(f"  - Initializing {model_type}/{model_name} model...")
                    model_instance = init_func()
                    if model_instance is None:
                        print(f"  ✗ {model_type}/{model_name} model initialization returned None")
                        continue
                    cls._instances[f"{model_type}/{model_name}"] = model_instance
                    print(f"  ✓ {model_type}/{model_name} model initialized successfully")
                    print(f"    Model type: {type(model_instance)}")
                    successful_inits += 1
                except Exception as e:
                    print(f"  ✗ Error initializing {model_type}/{model_name} model: {str(e)}")
                    import traceback
                    traceback.print_exc()

        print(f"\nML model initialization complete: {successful_inits}/{total_models} models initialized")
        return successful_inits > 0  # Return True if at least one model was initialized

    @classmethod
    def get_model(cls, model_type, model_name):
        """Get an initialized model by type and name"""
        model_key = f"{model_type}/{model_name}"
        return cls._instances.get(model_key)

    @classmethod
    def process(cls, data_type, model, data):
        """Process data with the specified model type"""
        print(f"\n=== MLModels.process ===")
        print(f"Processing {data_type}/{model}")

        model_instance = cls.get_model(data_type, model)
        if not model_instance:
            print(f"Error: Model {data_type}/{model} not found or not initialized")
            return {"error": f"Model {data_type}/{model} not found or not initialized"}

        print(f"Found model instance: {type(model_instance)}")

        # Route to appropriate processing function based on model type
        if data_type == 'audio':
            if model == 'emotion':
                return cls._process_audio_emotion(model_instance, data)
        elif data_type == 'typing':
            if model == 'pattern':
                return cls._process_typing_pattern(model_instance, data)
        elif data_type == 'image':
            if model == 'classification':
                return cls._process_image_classification(model_instance, data)
            elif model == 'ecg':
                return cls._process_ecg_classification(model_instance, data)
        elif data_type == 'health':
            if model == 'diabetes':
                return cls._process_diabetes_prediction(model_instance, data)

        print(f"Error: Processing for {data_type}/{model} not implemented")
        return {"error": f"Processing for {data_type}/{model} not implemented"}

    @classmethod
    def _process_image_classification(cls, model_instance, image_data):
        """Process image data with MedViTClassifier or run_all_models if class is passed"""
        try:
            import tempfile, requests, os
            if isinstance(image_data, dict) and 'url' in image_data:
                image_data = image_data['url']
            if isinstance(image_data, str) and image_data.startswith('http'):
                # Download the image to a temp file
                r = requests.get(image_data)
                r.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
                    f.write(r.content)
                    temp_path = f.name
                image_data = temp_path
            # If model_instance is a class, run across all models
            if isinstance(model_instance, type):
                return model_instance.run_all_models(image_data)
            else:
                return model_instance.predict(image_data)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback; traceback.print_exc()
            return {"error": str(e)}

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

    @classmethod
    def _process_ecg_classification(cls, model_instance, data):
        """Process ECG data with ECG-FM model"""
        try:
            print("\n=== ECG Processing Debug ===")
            print(f"Model instance type: {type(model_instance)}")
            print(f"Input data: {data}")

            from models.image.ecg.digitization import digitize_ecg_image
            import tempfile

            # Handle different input formats
            if isinstance(data, dict):
                if 'url' in data:
                    print(f"Processing URL: {data['url']}")
                    # Handle URL input
                    response = requests.get(data['url'])
                    response.raise_for_status()
                    print(f"Successfully downloaded image, size: {len(response.content)} bytes")

                    # Save image to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                        temp_path = temp_file.name
                        temp_file.write(response.content)
                        print(f"Saved image to: {temp_path}")

                    try:
                        # Convert image to ECG signal
                        print("Starting digitization...")
                        digitized_data = digitize_ecg_image(temp_path)
                        print(f"Digitization complete. Signal shape: {digitized_data['signal'].shape}")

                        # Process with ML model
                        print("Processing with ML model...")
                        results = model_instance.process_ecg(
                            digitized_data['signal'],
                            digitized_data['text_description']
                        )
                        print(f"ML model results: {results}")

                        # Format the results
                        class_names = model_instance.get_class_names()
                        formatted_results = {
                            'prediction': class_names[results['prediction']],
                            'probabilities': {
                                class_name: float(prob)
                                for class_name, prob in zip(class_names, results['probabilities'])
                            },
                            'segment_predictions': [class_names[p] for p in results['segment_predictions']],
                            'segment_probabilities': [
                                {class_name: float(prob) for class_name, prob in zip(class_names, probs)}
                                for probs in results['segment_probabilities']
                            ]
                        }
                        print(f"Formatted results: {formatted_results}")
                        return formatted_results

                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            print(f"Cleaned up temporary file: {temp_path}")

                elif 'signal' in data:
                    print("Processing direct signal input...")
                    # Handle direct signal input
                    ecg_signal = data['signal']
                    text_description = data.get('text_description', 'ECG recording with 12 leads.')

                    # Process with ML model
                    results = model_instance.process_ecg(ecg_signal, text_description)

                    # Format the results
                    class_names = model_instance.get_class_names()
                    formatted_results = {
                        'prediction': class_names[results['prediction']],
                        'probabilities': {
                            class_name: float(prob)
                            for class_name, prob in zip(class_names, results['probabilities'])
                        },
                        'segment_predictions': [class_names[p] for p in results['segment_predictions']],
                        'segment_probabilities': [
                            {class_name: float(prob) for class_name, prob in zip(class_names, probs)}
                            for probs in results['segment_probabilities']
                        ]
                    }
                    return formatted_results
                else:
                    print("Invalid data format: missing 'url' or 'signal' key")
                    return {"error": "Invalid data format. Expected 'url' or 'signal' key"}
            else:
                print(f"Invalid data type: {type(data)}")
                return {"error": "Invalid data format. Expected dict with 'url' or 'signal', or array"}

        except Exception as e:
            print(f"Error processing ECG data: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    @classmethod
    def _process_diabetes_prediction(cls, model_instance, data):
        """Process health data with diabetes prediction model"""
        try:
            if not isinstance(data, dict):
                return {"error": "Input data must be a dictionary"}

            # Extract data from the 'url' field
            if 'url' in data:
                data = data['url']

            # Handle case-insensitive field names
            if 'HbA1c_level' in data:
                data['hba1c_level'] = data.pop('HbA1c_level')

            # Validate required fields
            required_fields = [
                'gender', 'age', 'hypertension', 'heart_disease',
                'smoking_history', 'bmi', 'hba1c_level', 'blood_glucose_level'
            ]

            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return {"error": f"Missing required fields: {', '.join(missing_fields)}"}

            # Make prediction
            result = model_instance.predict(data)
            return result

        except Exception as e:
            print(f"Error processing diabetes prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


def find_localtunnel_executable(custom_dir=None):
    """
    Find the localtunnel executable, first in a custom directory, then in PATH.
    Returns the full path to the executable or None if not found.
    """
    # Determine potential executable names based on OS
    os_name = platform.system()
    executable_names = []
    if os_name == "Windows":
        # On Windows, look for .cmd, .bat, and potentially .exe or the shell script
        executable_names = ['lt.cmd', 'lt.bat', 'lt.exe', 'lt']
    else: # macOS, Linux, etc.
        # On Unix-like systems, the executable is typically just 'lt' or the shell script
        executable_names = ['lt']

    lt_path = None

    # 1. Check the custom directory first
    if custom_dir and os.path.isdir(custom_dir):
        print(f"Checking custom directory for localtunnel executable: {custom_dir}")
        for name in executable_names:
            potential_path = os.path.join(custom_dir, name)
            # Check if the file exists and is executable
            if os.path.exists(potential_path) and os.path.isfile(potential_path):
                 # On Windows, .cmd and .bat are executable. On Unix, check os.access
                 if os_name == "Windows" or os.access(potential_path, os.X_OK):
                    lt_path = potential_path
                    print(f"Found localtunnel executable in custom directory: {lt_path}")
                    return lt_path # Found it, return immediately

    # 2. If not found in the custom directory, check the system's PATH
    print("Localtunnel executable not found in custom directory. Checking system PATH...")
    lt_path = shutil.which('lt') # shutil.which handles OS-specific extensions

    if lt_path:
        print(f"Found localtunnel executable in system PATH: {lt_path}")
        return lt_path
    else:
        # Provide helpful message if not found anywhere
        print("Error: 'lt' command not found in the custom directory or your system's PATH.")
        print("Please ensure localtunnel is installed globally (`npm install -g localtunnel`)")
        print("and that the directory containing the 'lt' executable is in your system's PATH,")
        print("or that the 'custom_ngrok' folder exists in the script's directory")
        print("and contains the correct 'lt' executable for your OS.")
        if os_name == "Windows":
             print("On Windows, npm typically installs to %AppData%\\npm or %AppData%\\npm\\node_modules\\.bin")
        elif os_name == 'Darwin' or os_name == 'Linux':
             print("On macOS/Linux, npm typically installs to /usr/local/bin or ~/.npm-global/bin")
        return None


def run_localtunnel():
    """
    Set up a localtunnel tunnel by running the 'lt' command
    and register with node handler automatically.
    Prioritizes finding the 'lt' executable in a 'custom_ngrok' folder
    in the script's directory, then falls back to system PATH.
    """
    global current_tunnel_url, localtunnel_process

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    custom_localtunnel_dir = os.path.join(script_dir, 'custom_ngrok')

    # Find the localtunnel executable path
    lt_path = find_localtunnel_executable(custom_localtunnel_dir)

    if not lt_path:
        # Error message already printed by find_localtunnel_executable
        return None

    print(f"Using localtunnel executable at: {lt_path}")
    print("Starting localtunnel...")

    try:
        # Start localtunnel process using the found path and forwarding port 9000
        # Use '--port' argument to specify the local port
        process_command = [lt_path, '--port', '9000'] # Use port 9000 for this service

        # Use subprocess.Popen to run the command in the background
        localtunnel_process = subprocess.Popen(
            process_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, # Decode stdout/stderr as text
            # shell=True is generally discouraged unless necessary due to security risks
        )

        # Read stdout line by line to find the URL printed by localtunnel
        url = None
        start_time = time.time()
        timeout = 60 # Timeout for finding the URL (increased for potentially slow startups)

        while True:
            # Check if process exited prematurely
            if localtunnel_process.poll() is not None:
                 stderr_output = localtunnel_process.stderr.read()
                 stdout_output = localtunnel_process.stdout.read()
                 print(f"Localtunnel process exited unexpectedly with return code {localtunnel_process.returncode}.")
                 if stdout_output:
                      print(f"Stdout: {stdout_output.strip()}")
                 if stderr_output:
                      print(f"Stderr: {stderr_output.strip()}")
                 return None

            # Read a line from stdout
            output_line = localtunnel_process.stdout.readline()

            # Check if the line contains the URL
            if "your url is:" in output_line:
                url = output_line.split("your url is:")[1].strip()
                break # Found the URL, exit the loop

            # Check for errors in stderr (non-blocking read)
            try:
                error_output = localtunnel_process.stderr.readline()
                if error_output:
                    print(f"Localtunnel stderr: {error_output.strip()}")
            except Exception as e:
                 print(f"Error reading stderr: {e}")


            # Add a timeout to prevent infinite waiting if URL is not printed
            if time.time() - start_time > timeout:
                 print(f"Timeout waiting for localtunnel URL after {timeout} seconds.")
                 # Attempt to terminate the process before returning None
                 try:
                     localtunnel_process.terminate()
                     localtunnel_process.wait(timeout=5)
                 except:
                     pass
                 return None

            time.sleep(0.1) # Small delay to avoid busy waiting

        # If a URL was successfully found
        if url:
            current_tunnel_url = url
            print(f"Ingress established at: {current_tunnel_url}")

            # Register with node handler after localtunnel is set up
            # Only attempt registration if NODE_HANDLER_URL is set in environment variables
            if NODE_HANDLER_URL:
                def registration_retry():
                    """Retry registration with node handler until successful"""
                    while not shutdown_event.is_set():
                        if register_with_node_handler():
                            print("Successfully registered with node handler")
                            break # Exit retry loop on success
                        time.sleep(10) # Wait 10 seconds between attempts

                # Initial registration attempt
                if not register_with_node_handler():
                    print("Failed to register with node handler. Will keep trying in background...")
                    # Start a daemon thread for retries so it doesn't prevent script exit
                    registration_thread = threading.Thread(target=registration_retry, daemon=True)
                    registration_thread.start()

                # Start heartbeat thread to keep connection alive and report status
                heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
                heartbeat_thread.start()

            # The subprocess itself keeps the tunnel alive.
            # The main thread will be kept alive by the shutdown_event loop later.
            return current_tunnel_url
        else:
            print("Failed to get localtunnel URL from output.")
            # Attempt to terminate the process if URL wasn't found but process is still running
            try:
                if localtunnel_process and localtunnel_process.poll() is None:
                    localtunnel_process.terminate()
                    localtunnel_process.wait(timeout=5)
            except:
                pass
            return None

    except FileNotFoundError:
        # This exception should be less likely now with find_localtunnel_executable,
        # but kept as a fallback. It would indicate the found path is invalid.
        print(f"Error: Executable not found at the determined path: {lt_path}")
        return None
    except Exception as e:
        print(f"Error setting up localtunnel: {str(e)}")
        import traceback
        traceback.print_exc()
        # Attempt to terminate the process in case of other exceptions
        try:
            if localtunnel_process and localtunnel_process.poll() is None:
                localtunnel_process.terminate()
                localtunnel_process.wait(timeout=5)
        except:
                pass
        return None


# --- Node Registration, Heartbeat, Deregistration ---
def wait_for_node_handler(max_retries=10):
    """Wait for node handler to be ready"""
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
    global current_tunnel_url # Use the localtunnel URL
    if not current_tunnel_url or not NODE_HANDLER_URL: # Ensure URL and handler URL are available
        print("Cannot register with node handler - no tunnel URL or NODE_HANDLER_URL available")
        return False

    try:
        # Wait for node handler to be ready
        if not wait_for_node_handler():
            print("Node handler not ready after multiple attempts")
            return False

        print(f"Attempting to register with node handler at {NODE_HANDLER_URL}...")
        response = requests.post(
            f"{NODE_HANDLER_URL}/register",
            json={"url": current_tunnel_url, "type": "ml"}, # Send the localtunnel URL
            timeout=15 # Increased timeout
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
    global current_tunnel_url # Use the localtunnel URL
    if not current_tunnel_url or not NODE_HANDLER_URL: # Ensure URL and handler URL are available
        return # Nothing to deregister if no URL or handler URL

    try:
        print(f"Deregistering from node handler...")
        # Send the current tunnel URL for deregistration
        requests.post(f"{NODE_HANDLER_URL}/deregister", json={"url": current_tunnel_url}, timeout=15) # Increased timeout
        print("Deregistration request sent.")
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
    global current_tunnel_url, HEARTBEAT_INTERVAL  # Use the localtunnel variable
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
                json={"url": current_tunnel_url}, # Use the localtunnel variable
                timeout=15
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
    """Handle shutdown signals"""
    print("\nReceived shutdown signal. Cleaning up...")
    shutdown_event.set() # Signal other threads to stop
    time.sleep(1) # Give threads a moment to notice the shutdown event

    # Deregister from node handler first
    if NODE_HANDLER_URL:
        deregister_from_node_handler()

    # Clean up localtunnel process
    print("Cleaning up localtunnel process...")
    try:
        if localtunnel_process and localtunnel_process.poll() is None:
            # Use terminate() first (sends SIGTERM) for graceful shutdown
            localtunnel_process.terminate()
            try:
                localtunnel_process.wait(timeout=10) # Wait a bit for it to terminate gracefully
            except subprocess.TimeoutExpired:
                print("Localtunnel process did not terminate gracefully, killing...")
                localtunnel_process.kill() # If it doesn't terminate, kill it (SIGKILL)
                localtunnel_process.wait() # Wait for it to be killed
        print("Localtunnel process stopped.")
    except Exception as e:
        print(f"Error stopping localtunnel process: {str(e)}")


    sys.exit(0) # Exit the main process


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
def process_ml_request():
    """Process ML data with the specified model"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        url = data.get('url')
        data_type = data.get('data_type')
        model = data.get('model')

        if not url or not data_type or not model:
            return jsonify({'error': 'Missing required fields (url, data_type, model)'}), 400

        # Process the request using MLModels class
        result = MLModels.process(data_type, model, url)
        return jsonify(result)
    except Exception as e:
        print(f"Error processing ML request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/workflow/process', methods=['POST'])
def process_workflow_request():
    """Process workflow model requests (ER Triage, Lab Analysis, ECG Analysis, Diabetes Prediction)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        model = data.get('model')
        workflow_data = data.get('data')

        if not model or not workflow_data:
            return jsonify({'error': 'Missing required fields (model, data)'}), 400

        # Process based on model type
        if model == 'er_triage':
            from models.workflow.er_triage import ERTriageModel
            result = ERTriageModel().process(workflow_data)
        elif model == 'lab_analysis':
            from models.workflow.lab_analysis import LabAnalysisModel
            result = LabAnalysisModel().process(workflow_data)
        elif model == 'ecg_analysis':
            from models.image.ecg.integration import ECG_FM_Handler
            result = ECG_FM_Handler().process_ecg(workflow_data['ecg_signal'], workflow_data.get('text_description', ''))
        elif model == 'diabetes':
            from models.health.diabetes import diabetes_predictor
            result = diabetes_predictor.predict(workflow_data)
        else:
            return jsonify({'error': f'Invalid model: {model}'}), 400

        return jsonify(result)
    except Exception as e:
        print(f"Error processing workflow request: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/deregister', methods=['POST'])
def handle_deregister():
    """Handle deregistration request from node handler (shutdown notice)"""
    global current_tunnel_url # Use the localtunnel variable
    try:
        data = request.json
        if data and data.get('url') == current_tunnel_url:
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

def wait_for_flask(port, max_retries=5):
    """Wait for Flask to start accepting connections"""
    import socket

    for i in range(max_retries):
        try:
            with socket.create_connection(("localhost", port), timeout=1.0):
                return True
        except (ConnectionRefusedError, socket.timeout):
            print(f"Waiting for Flask to start (attempt {i+1}/{max_retries})...")
            time.sleep(2)
    return False


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


if __name__ == '__main__':
    try:
        # Initialize all ML models centrally
        if not initialize_models():
            print("Failed to initialize any ML models. Exiting.")
            sys.exit(1)

        flask_port = 9000 # Use port 9000 for this service
        flask_thread = threading.Thread(target=lambda: serve(app, host='0.0.0.0', port=flask_port, threads=4), daemon=True)
        print(f"Starting Flask app on port {flask_port}...")
        flask_thread.start()

        # Wait for Flask app to be ready before starting localtunnel
        if not wait_for_flask(flask_port):
             print("Flask app failed to start. Exiting.")
             sys.exit(1)
        print("Flask app is running.")

        # Now start localtunnel in its thread
        # run_localtunnel function handles finding the executable, starting the process,
        # getting the URL, and initiating registration/heartbeat if NODE_HANDLER_URL is set.
        localtunnel_thread = threading.Thread(target=run_localtunnel, daemon=True)
        print("Starting localtunnel tunnel...")
        localtunnel_thread.start()

        # Keep the main thread alive to handle signals
        print("Main process is running. Press Ctrl+C to stop.")
        while not shutdown_event.is_set():
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down application due to KeyboardInterrupt...")
    except Exception as e:
        print(f"An unhandled error occurred during startup: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure graceful shutdown is initiated
        shutdown_event.set()
        # Give threads a moment to clean up
        time.sleep(2)
        # The handle_shutdown signal handler will perform final cleanup and exit
        # If the script exits without signal (e.g., unhandled exception),
        # the daemon threads will be terminated automatically, but explicit
        # cleanup in handle_shutdown is more reliable.
        # Explicitly call deregister and process termination here as a fallback
        # in case the signal handler isn't invoked cleanly.
        print("Performing final cleanup...")
        if NODE_HANDLER_URL and current_tunnel_url:
             try:
                 deregister_from_node_handler()
             except Exception as e:
                 print(f"Error during final deregistration: {str(e)}")
        if localtunnel_process and localtunnel_process.poll() is None:
             try:
                 localtunnel_process.terminate()
                 localtunnel_process.wait(timeout=5)
             except:
                 pass
        print("Final cleanup complete. Exiting.")
        sys.exit(0)