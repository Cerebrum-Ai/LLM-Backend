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

app = Flask(__name__)
current_ngrok_url = None
shutdown_event = threading.Event()

# ==========================================
# Chatbot Logic
# ==========================================

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
            timeout=10
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

def process_with_llm(data, ml_results=None, llm_url=None):
    """Process data with the LLM service"""
    try:
        print(f"Processing with LLM service")
        
        # Use provided LLM URL or default
        active_llm_url = llm_url or LLM_APP_URL
        
        print(f"Using LLM URL: {active_llm_url}")
        
        # Enhance data with ML results if available
        if ml_results:
            # For audio emotion analysis
            if "audio" in ml_results and "detected_emotion" in ml_results["audio"]:
                # Structure according to how main.py expects it
                if isinstance(data.get("audio"), dict):
                    data["audio"]["emotion"] = ml_results["audio"]
                else:
                    data["audio"] = {
                        "file": data["audio"] if "audio" in data else None,
                        "emotion": ml_results["audio"]
                    }
        
        # Forward to LLM service
        print(f"Sending request to LLM service at {active_llm_url}")
        response = requests.post(
            f"{active_llm_url}/api/chat",
            json=data,
            timeout=30  # Increased timeout to 30 seconds
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
    """Set up an ngrok tunnel"""
    global current_ngrok_url
    try:
        # Check for auth token
        auth_token = NGROK_AUTH_TOKEN
        
        # If individual token not set, try to use the tokens list that main.py and models.py use
        if not auth_token and NGROK_AUTH_TOKENS:
            auth_tokens = NGROK_AUTH_TOKENS.split(',')
            if auth_tokens:
                auth_token = auth_tokens[0]  # Use the first token
        
        if not auth_token:
            print("Error: No ngrok auth token available. Set NGROK_AUTH_TOKEN or NGROK_AUTH_TOKENS in .env")
            return None
            
        # Configure ngrok
        ngrok.set_auth_token(auth_token)
        
        # Start ngrok tunnel
        tunnel = ngrok.connect(5100)  # Use a different port than your other services
        current_ngrok_url = tunnel.url()
        print(f"Chatbot is publicly accessible at: {current_ngrok_url}")
        return current_ngrok_url
    except Exception as e:
        print(f"Error setting up ngrok: {str(e)}")
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
    """Deregister from the node handler"""
    global current_ngrok_url
    if not current_ngrok_url:
        return
        
    try:
        print(f"Deregistering from node handler...")
        requests.post(f"{NODE_HANDLER_URL}/deregister", json={"url": current_ngrok_url}, timeout=5)
    except Exception as e:
        print(f"Error deregistering: {str(e)}")

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
            target=lambda: serve(app, host='0.0.0.0', port=5100, threads=4),
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
