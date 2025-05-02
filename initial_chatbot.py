from flask import Flask, request, jsonify
import requests
import os
import sys
import time
import threading
import signal
import ngrok
import tempfile
import base64
from dotenv import load_dotenv

load_dotenv()

# Configuration
NODE_HANDLER_URL = os.environ.get("NODE_HANDLER_URL")
ML_MODELS_URL = os.environ.get("ML_MODELS_URL", "http://localhost:9000")
LLM_APP_URL = os.environ.get("LLM_APP_URL", "http://localhost:5050")
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN")
NGROK_AUTH_TOKENS = os.environ.get("NGROK_AUTH_TOKENS")
HEARTBEAT_INTERVAL = 30  # seconds

app = Flask(__name__)
current_ngrok_url = None
shutdown_event = threading.Event()

# ==========================================
# Chatbot Logic
# ==========================================

from PIL import Image
from supabase import create_client, Client
import io
import requests as pyrequests

SUPABASE_URL = os.environ.get("SUPABASE_URL")
# Prefer SUPABASE_ADMIN_KEY (service_role) if set, fall back to SUPABASE_KEY
SUPABASE_ADMIN_KEY = os.environ.get("SUPABASE_ADMIN_KEY")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "images")

supabase: Client = None
supabase_secret = SUPABASE_ADMIN_KEY or SUPABASE_KEY
if SUPABASE_URL and supabase_secret:
    supabase = create_client(SUPABASE_URL, supabase_secret)
# If neither is set, supabase will remain None and uploading will error


def process_and_upload_image(image_input):
    """
    Accepts either a URL (str) or a file-like object (e.g., BytesIO, FileStorage), processes and uploads the image, and returns the public URL.
    """
    # Handle file-like object or URL
    if hasattr(image_input, 'read'):
        # It's a file-like object (e.g., Flask FileStorage)
        image_bytes = image_input.read()
        img = Image.open(io.BytesIO(image_bytes))
    elif isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, str):
        # Assume it's a URL
        resp = pyrequests.get(image_input)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
    else:
        raise ValueError('Unsupported image input type for upload.')
    w, h = img.size
    # Upscale if needed
    scale = max(640 / w, 480 / h, 1)
    if w < 640 or h < 480:
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    # Convert to RGB if image has alpha (transparency)
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")
    # Save to buffer
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    # Upload to Supabase
    filename = f"image_{int(time.time())}.png"
    storage_path = f"{filename}"
    if supabase is None:
        raise Exception("Supabase client not initialized. Set SUPABASE_URL and SUPABASE_KEY.")
    buf.seek(0)
    print(f"Uploading to Supabase: storage_path={storage_path}, buf_type={type(buf)}")
    # Use correct content-type for PNG
    try:
        supabase.storage.from_(SUPABASE_BUCKET).upload(storage_path, buf, file_options={"content-type": "image/png"})
    except TypeError as e:
        print(f"TypeError uploading BytesIO, retrying with bytes: {e}")
        buf.seek(0)
        supabase.storage.from_(SUPABASE_BUCKET).upload(storage_path, buf.getvalue(), file_options={"content-type": "image/png"})
    except Exception as e:
        print(f"Error uploading image to Supabase: {e}")
        raise
    # Get public URL
    pub_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(storage_path)
    print("Successfully recieved url")
    return pub_url


def analyze_input(data):
    """
    Analyze incoming data to determine what services to use
    
    Returns a dict with:
    - data_types: list of detected data types (text, image, audio, etc.)
    - processing_plan: how to process this data
    """
    data_types = []
    if "question" in data:
        data_types.append("text")
    
    if "image" in data and data["image"]:
        data_types.append("image")
    
    if "audio" in data and data["audio"]:
        data_types.append("audio")
    
    if "gait" in data and data["gait"]:
        data_types.append("gait")
    
    if "typing" in data and data["typing"]:
        data_types.append("typing")
    
    # Determine processing plan
    processing_plan = {
        "requires_ml": any(x in data_types for x in ["audio", "gait", "typing", "image"]),
        "requires_llm": "text" in data_types or "image" in data_types,
        "parallel_processing": "image" in data_types,  # Images go to both ML and LLM
        "ml_models": []
    }
    
    # Determine which ML models to use
    if "audio" in data_types:
        processing_plan["ml_models"].append({"data_type": "audio", "model": "emotion"})
    
    if "image" in data_types:
        processing_plan["ml_models"].append({"data_type": "image", "model": "classification"})
    
    if "gait" in data_types:
        processing_plan["ml_models"].append({"data_type": "gait", "model": "analysis"})
    
    if "typing" in data_types:
        processing_plan["ml_models"].append({"data_type": "typing", "model": "pattern"})
    
    return {
        "data_types": data_types,
        "processing_plan": processing_plan
    }

def process_ml_data(data, data_type, model, ml_url=None):
    """Process data with an ML model via the ML service"""
    try:
        print(f"Processing {data_type} data with ML model: {model}")
        
        # Use provided ML URL or default
        active_ml_url = ml_url or ML_MODELS_URL
        
        # For file data types, save to temp file and get URL
        temp_path = None
        url = None
        
        if hasattr(data, 'save'):  # FileStorage object
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=f'.{data_type}', delete=False) as temp_file:
                temp_path = temp_file.name
                data.save(temp_file)
                url = temp_path
        else:
            # Assume it's already a path or URL
            url = data
        
        # Call ML service
        response = requests.post(
            f"{active_ml_url}/ml/process",
            json={
                "url": url,
                "data_type": data_type,
                "model": model
            },
            timeout=40
        )
        
        # Clean up temp file if created
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error from ML service: {response.text}")
            return {"error": f"ML service error: {response.text}"}
            
    except Exception as e:
        print(f"Error processing {data_type} with ML: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Image preprocessing is now handled exclusively in main.py

def process_with_llm(data, ml_results=None, llm_url=None):
    """Process data with the LLM service"""
    try:
        print(f"Processing with LLM service")
        
        # Use provided LLM URL or default
        active_llm_url = llm_url or LLM_APP_URL
        
        print(f"Using LLM URL: {active_llm_url}")
        
        # Image preprocessing is now handled exclusively in main.py
        # Just pass the image URL directly
        
        # If image is present, process and upload it, replacing with the public URL
        if "image" in data and isinstance(data["image"], str) and data["image"].startswith("http"):
            try:
                public_url = process_and_upload_image(data["image"])
                data["image"] = public_url
            except Exception as e:
                print(f"Error processing and uploading image: {e}")
                # If upload fails, keep the original URL
                pass

        # Merge all ML results into the data dict for unified LLM context
        if ml_results:
            for dtype, result in ml_results.items():
                if dtype == "image":
                    # Always keep 'image' as the processed public image URL for LLM capabilities
                    if "image" in data:
                        data["image_ml"] = result  # Add ML results under a separate key
                    else:
                        # If for some reason image is not present, fallback to old logic
                        data["image_ml"] = result
                else:
                    # If data[dtype] is a dict, update it with ML result
                    if dtype in data and isinstance(data[dtype], dict):
                        if isinstance(result, dict):
                            data[dtype].update(result)
                        else:
                            data[dtype][f"{dtype}_ml_result"] = result
                    else:
                        entry = {}
                        if dtype in data:
                            entry["input"] = data[dtype]
                        if isinstance(result, dict):
                            entry.update(result)
                        else:
                            entry[f"{dtype}_ml_result"] = result
                        data[dtype] = entry
        
        # Forward to LLM service
        print(f"Sending request to LLM service at {active_llm_url}")
        response = requests.post(
            f"{active_llm_url}/api/chat",
            json=data,
            timeout=90  # Increased timeout to 30 seconds
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error from LLM service: {response.text}")
            return {"error": f"LLM service error: {response.text}"}
            
    except Exception as e:
        print(f"Error processing with LLM: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# ==========================================
# API Endpoints
# ==========================================

@app.route('/')
def root():
    """Root endpoint for health checks"""
    return jsonify({
        'status': 'ok',
        'service': 'initial_chatbot',
        'message': 'Initial chatbot is running'
    })

@app.route('/status', methods=['GET'])
def status():
    """Status endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Initial chatbot is running',
        'connected_services': {
            'node_handler': NODE_HANDLER_URL,
            'ml_models': ML_MODELS_URL,
            'llm_app': LLM_APP_URL
        }
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint that orchestrates the processing flow"""
    try:
        # Get ML and LLM URLs from headers if provided by node_handler
        ml_url = request.headers.get('X-ML-Node-URL')
        llm_url = request.headers.get('X-LLM-Node-URL')
        
        # Use the provided URLs if available, otherwise fall back to env variables
        active_ml_url = ml_url if ml_url and ml_url != "No active ML node available" else ML_MODELS_URL
        active_llm_url = llm_url if llm_url and llm_url != "No active LLM node available" else LLM_APP_URL
        
        print(f"Using ML URL: {active_ml_url}")
        print(f"Using LLM URL: {active_llm_url}")
        
        # Handle form data
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            question = request.form.get('question')
            
            # Process file uploads
            data = {'question': question}
            
            for file_type in ['image', 'audio', 'gait']:
                if file_type in request.files:
                    data[file_type] = request.files[file_type]
            
            # Handle typing data if present
            if 'typing' in request.form:
                data['typing'] = request.form.get('typing')
                
        # Handle JSON data
        elif request.content_type and request.content_type.startswith('application/json'):
            data = request.get_json(silent=True)
            if not data:
                return jsonify({'error': 'Invalid JSON data'}), 400
        else:
            return jsonify({'error': 'Unsupported content type'}), 415
        
        # Analyze input to determine processing path
        analysis = analyze_input(data)
        print(f"Input analysis: {analysis}")
        
        processing_plan = analysis['processing_plan']
        ml_results = {}
        
        # Process with ML models if needed
        if processing_plan['requires_ml']:
            for model_info in processing_plan['ml_models']:
                data_type = model_info['data_type']
                model = model_info['model']
                
                if data_type in data and data[data_type]:
                    ml_result = process_ml_data(data[data_type], data_type, model, active_ml_url)
                    ml_results[data_type] = ml_result
        
        # Process with LLM if needed
        llm_result = None
        if processing_plan['requires_llm']:
            llm_result = process_with_llm(data, ml_results, active_llm_url)
        
        # Combine results
        response = {
            'status': 'success',
            'analysis': {}
        }
        
        if llm_result:
            if isinstance(llm_result, dict):
                if 'analysis' in llm_result:
                    response['analysis'] = llm_result['analysis']
                elif 'error' in llm_result:
                    response['llm_error'] = llm_result['error']
                elif 'answer' in llm_result:
                    response['answer'] = llm_result['answer']
            else:
                response['answer'] = llm_result
        
        # Add ML results to response if LLM didn't use them
        for data_type, result in ml_results.items():
            key = f"{data_type}_analysis" 
            if key not in response['analysis']:
                response['analysis'][key] = result
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ==========================================
# Ngrok and Server Setup
# ==========================================

def run_ngrok():
    """Set up an ngrok tunnel with token cycling and register with node handler automatically (using ngrok package)"""
    global current_ngrok_url
    import ngrok
    auth_tokens = []
    if NGROK_AUTH_TOKEN:
        auth_tokens.append(NGROK_AUTH_TOKEN)
    if NGROK_AUTH_TOKENS:
        auth_tokens.extend(NGROK_AUTH_TOKENS.split(','))
    if not auth_tokens:
        print("Error: No ngrok auth tokens available. Set NGROK_AUTH_TOKEN or NGROK_AUTH_TOKENS in .env")
        return None
    listener = None
    for i, token in enumerate(auth_tokens):
        try:
            print(f"Trying ngrok auth token {i + 1}/{len(auth_tokens)}...")
            ngrok.set_auth_token(token)
            time.sleep(2)
            listener = ngrok.forward(5100)
            current_ngrok_url = listener.url()
            print(f"Ingress established at: {current_ngrok_url}")
            # Register with node handler after ngrok is set up
            def registration_retry():
                while not shutdown_event.is_set():
                    if register_with_node_handler():
                        break
                    time.sleep(10)
            if not register_with_node_handler():
                print("Failed to register with node handler. Will keep trying...")
                registration_thread = threading.Thread(target=registration_retry, daemon=True)
                registration_thread.start()
            # Start heartbeat thread
            heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
            heartbeat_thread.start()
            # Keep the tunnel alive
            while not shutdown_event.is_set():
                time.sleep(1)
            return current_ngrok_url
        except Exception as e:
            print(f"Error setting up ngrok: {str(e)}")
            continue
    print("Failed to establish ngrok tunnel with any available tokens.")
    return None

def register_with_node_handler():
    """Register this chatbot with the node handler"""
    global current_ngrok_url
    if not current_ngrok_url:
        print("Cannot register with node handler - no ngrok URL available")
        return False
        
    try:
        print(f"Registering chatbot with node handler at {NODE_HANDLER_URL}...")
        response = requests.post(
            f"{NODE_HANDLER_URL}/register",
            json={"url": current_ngrok_url, "type": "chatbot"},
            timeout=15
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
    """Deregister from the node handler"""
    global current_ngrok_url
    if not current_ngrok_url:
        return
        
    try:
        print(f"Deregistering from node handler...")
        requests.post(f"{NODE_HANDLER_URL}/deregister", json={"url": current_ngrok_url}, timeout=15)
    except Exception as e:
        print(f"Error deregistering: {str(e)}")

def send_heartbeat():
    """Send periodic heartbeat to node handler"""
    global current_ngrok_url, HEARTBEAT_INTERVAL  # Add global declaration for HEARTBEAT_INTERVAL
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

def registration_retry():
    """Retry registration with node handler until successful"""
    while not shutdown_event.is_set():
        if register_with_node_handler():
            print("Successfully registered with node handler")
            break
        time.sleep(10)  # Wait 10 seconds between attempts

def handle_shutdown(signum, frame):
    """Handle shutdown signals"""
    print("\nReceived shutdown signal. Cleaning up...")
    shutdown_event.set()
    time.sleep(1)
    try:
        # Clean up ngrok
        print("Cleaning up ngrok...")
        ngrok.disconnect()
        print("Ngrok disconnected successfully")
    except Exception as e:
        print(f"Error disconnecting ngrok: {str(e)}")
    
    deregister_from_node_handler()
    sys.exit(0)

if __name__ == '__main__':
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
        
        # Start Flask in a separate thread
        from waitress import serve
        flask_thread = threading.Thread(
            target=lambda: serve(app, host='0.0.0.0', port=5100, threads=4,channel_timeout=120),
            daemon=True
        )
        print("Starting chatbot service...")
        flask_thread.start()
        
        # Set up ngrok
        ngrok_url = run_ngrok()
        if not ngrok_url:
            print("Failed to set up ngrok tunnel. Exiting.")
            sys.exit(1)
        
        # Register with node handler
        if NODE_HANDLER_URL:
            register_with_node_handler()
        
        # Keep the main thread alive
        while not shutdown_event.is_set():
            time.sleep(1)
            
    except Exception as e:
        print(f"Error starting chatbot: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
