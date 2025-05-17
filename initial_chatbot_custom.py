from flask import Flask, request, jsonify
import requests
import os
import sys
import time
import threading
import signal
#import ngrok
import tempfile
import base64
from dotenv import load_dotenv
import subprocess
load_dotenv()
import shutil
import platform

# Configuration
NODE_HANDLER_URL = os.environ.get("NODE_HANDLER_URL")
ML_MODELS_URL = os.environ.get("ML_MODELS_URL", "http://localhost:9000")
LLM_APP_URL = os.environ.get("LLM_APP_URL", "http://localhost:5050")
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN")
NGROK_AUTH_TOKENS = os.environ.get("NGROK_AUTH_TOKENS")
HEARTBEAT_INTERVAL = 30  # seconds

app = Flask(__name__)
current_tunnel_url = None
shutdown_event = threading.Event()
localtunnel_process = None
reconnection_in_progress = threading.Event()
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

@app.route('/deregister', methods=['POST'])
def handle_deregister():
    global reconnection_in_progress
    data = request.json
    if data and data.get('url') == current_tunnel_url:
        print("Received deregister request from node handler. Starting reconnection attempts...")
        reconnection_in_progress.set()
        def delayed_reconnect():
            wait_time = 5  # seconds (adjust as needed)
            print(f"Waiting {wait_time} seconds before starting reconnection attempts...")
            time.sleep(wait_time)
            handle_node_handler_disconnect()
        threading.Thread(target=delayed_reconnect, daemon=True).start()
        return jsonify({'status': 'accepted', 'message': 'Node is preparing for reconnection'}), 202
    return jsonify({'error': 'Invalid request'}), 400

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

def run_localtunnel():
    """Set up a localtunnel tunnel and register with node handler automatically"""
    global current_tunnel_url # Use the renamed variable
    
    print("Starting localtunnel...")
    try:
        # Start localtunnel process
        # Assuming localtunnel is installed and in your PATH
        # This command forwards port 5100
        process = subprocess.Popen(
            ['lt', '--port', '5100'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Read stdout line by line to find the URL
        url = None
        while True:
            output = process.stdout.readline()
            if "your url is:" in output:
                url = output.split("your url is:")[1].strip()
                break
            # Check for errors
            error_output = process.stderr.readline()
            if error_output:
                print(f"Localtunnel stderr: {error_output.strip()}")
            if process.poll() is not None: # Check if process exited
                break
            time.sleep(0.1)

        if url:
            current_tunnel_url = url
            print(f"Ingress established at: {current_tunnel_url}")
            
            # Register with node handler after localtunnel is set up
            def registration_retry():
                while not shutdown_event.is_set():
                    if register_with_node_handler():
                        break
                    time.sleep(10)
            
            if NODE_HANDLER_URL: # Only attempt registration if NODE_HANDLER_URL is set
                 if not register_with_node_handler():
                     print("Failed to register with node handler. Will keep trying...")
                     registration_thread = threading.Thread(target=registration_retry, daemon=True)
                     registration_thread.start()

            # Start heartbeat thread
            if NODE_HANDLER_URL: # Only send heartbeats if NODE_HANDLER_URL is set
                heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
                heartbeat_thread.start()
            
            # Keep the tunnel alive - the subprocess will handle this
            # We just need to make sure the main thread doesn't exit
            return current_tunnel_url
        else:
            print("Failed to get localtunnel URL.")
            return None

    except FileNotFoundError:
        print("Error: localtunnel command not found. Make sure localtunnel is installed (`npm install -g localtunnel`) and in your PATH.")
        return None
    except Exception as e:
        print(f"Error setting up localtunnel: {str(e)}")
        return None

def register_with_node_handler():
    """Register this chatbot with the node handler"""
    global current_tunnel_url # Use the renamed variable
    if not current_tunnel_url or not NODE_HANDLER_URL: # Add check for NODE_HANDLER_URL
        print("Cannot register with node handler - no tunnel URL or NODE_HANDLER_URL available")
        return False

    try:
        print(f"Registering chatbot with node handler at {NODE_HANDLER_URL}...")
        response = requests.post(
            f"{NODE_HANDLER_URL}/register",
            json={"url": current_tunnel_url, "type": "chatbot"}, # Use the renamed variable
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
    global current_tunnel_url # Use the renamed variable
    if not current_tunnel_url or not NODE_HANDLER_URL: # Add check for NODE_HANDLER_URL
        return

    try:
        print(f"Deregistering from node handler...")
        requests.post(f"{NODE_HANDLER_URL}/deregister", json={"url": current_tunnel_url}, timeout=15) # Use the renamed variable
    except Exception as e:
        print(f"Error deregistering: {str(e)}")

def send_heartbeat():
    """Send periodic heartbeat to node handler"""
    global current_tunnel_url, HEARTBEAT_INTERVAL, reconnection_in_progress  # Use the renamed variable
    consecutive_failures = 0
    max_consecutive_failures = 1  # Respond quickly to disconnections
    

    while not shutdown_event.is_set():
        try:
            if reconnection_in_progress.is_set():
                time.sleep(HEARTBEAT_INTERVAL)
                continue

            print("\nSending heartbeat to node handler...")
            response = requests.post(
                f"{NODE_HANDLER_URL}/heartbeat",
                json={"url": current_tunnel_url}, # Use the renamed variable
                timeout=15
            )
            if response.status_code != 200:
                print(f"Heartbeat not acknowledged: {response.text}")
                consecutive_failures += 1
                reconnection_in_progress.set()
                if consecutive_failures >= max_consecutive_failures:
                    print("\n=== Node Handler Connection Lost ===")
                    print("Error detected. Starting reconnection process...")
                    reconnection_in_progress.set()
                    if handle_node_handler_disconnect():
                        consecutive_failures = 0
                        reconnection_in_progress.clear()
                        print("✓ Successfully reconnected to node handler")
                time.sleep(HEARTBEAT_INTERVAL)
                continue
            else:
                consecutive_failures = 0
        except Exception as e:
            print(f"Heartbeat error: {str(e)}")
            consecutive_failures += 1
            reconnection_in_progress.set()
            if consecutive_failures >= max_consecutive_failures:
                print("\n=== Node Handler Connection Lost ===")
                print("Error detected. Starting reconnection process...")
                reconnection_in_progress.set()
                if handle_node_handler_disconnect():
                    consecutive_failures = 0
                    reconnection_in_progress.clear()
                    print("✓ Successfully reconnected to node handler")
        time.sleep(HEARTBEAT_INTERVAL)


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
    """Set up a localtunnel tunnel and register with node handler automatically"""
    global current_tunnel_url, localtunnel_process

   # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    custom_localtunnel_dir = os.path.join(script_dir, 'custom_ngrok')
    # Find the localtunnel executable path
    lt_path = find_localtunnel_executable(custom_localtunnel_dir)


    if not lt_path:
        print("Error: 'lt' command not found in your PATH.")
        print("Make sure localtunnel is installed globally (`npm install -g localtunnel`)")
        print("and that your system's PATH includes the directory where npm installs global packages.")
        return None

    print(f"Found localtunnel executable at: {lt_path}")
    print("Starting localtunnel...")

    try:
        # Start localtunnel process using the found path
        process_command = [lt_path, '--port', '5100']
        localtunnel_process = subprocess.Popen(
            process_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # shell=True # Avoid using shell=True unless absolutely necessary
        )

        # Read stdout line by line to find the URL
        url = None
        start_time = time.time()
        timeout = 30 # Timeout for finding the URL

        while True:
            # Check if process exited prematurely
            if localtunnel_process.poll() is not None:
                 stderr_output = localtunnel_process.stderr.read()
                 stdout_output = localtunnel_process.stdout.read()
                 print(f"Localtunnel process exited unexpectedly.")
                 if stdout_output:
                      print(f"Stdout: {stdout_output}")
                 if stderr_output:
                      print(f"Stderr: {stderr_output}")
                 return None

            output = localtunnel_process.stdout.readline()
            if "your url is:" in output:
                url = output.split("your url is:")[1].strip()
                break

            # Check for errors in stderr
            error_output = localtunnel_process.stderr.readline()
            if error_output:
                print(f"Localtunnel stderr: {error_output.strip()}")
                # You might want to add logic here to decide if a stderr message is a fatal error

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


        if url:
            current_tunnel_url = url
            print(f"Ingress established at: {current_tunnel_url}")

            # Register with node handler after localtunnel is set up
            if NODE_HANDLER_URL:
                def registration_retry():
                    while not shutdown_event.is_set():
                        if register_with_node_handler():
                            break
                        time.sleep(10)

                # Initial registration attempt
                if not register_with_node_handler():
                    print("Failed to register with node handler. Will keep trying...")
                    registration_thread = threading.Thread(target=registration_retry, daemon=True)
                    registration_thread.start()

                # Start heartbeat thread
                heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
                heartbeat_thread.start()

            # Keep the tunnel alive - the subprocess handles this.
            # The main thread will be kept alive by the shutdown_event loop later.
            return current_tunnel_url
        else:
            print("Failed to get localtunnel URL from output.")
            return None

    except FileNotFoundError:
        # This error should ideally be caught by shutil.which(), but included as a fallback
        print("Error: localtunnel command not found. Make sure localtunnel is installed (`npm install -g localtunnel`) and in your PATH.")
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
        # Clean up localtunnel process
        print("Cleaning up localtunnel...")
        if localtunnel_process and localtunnel_process.poll() is None:
            localtunnel_process.terminate() # or kill()
            localtunnel_process.wait()
        print("Localtunnel disconnected successfully")
    except Exception as e:
        print(f"Error disconnecting localtunnel: {str(e)}")

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

        # Set up localtunnel
        tunnel_url = run_localtunnel() # Call run_localtunnel
        if not tunnel_url:
            print("Failed to set up localtunnel tunnel. Exiting.")
            sys.exit(1)

        # Registration and heartbeat are now handled within run_localtunnel
        # if NODE_HANDLER_URL is set.

        # Keep the main thread alive
        while not shutdown_event.is_set():
            time.sleep(1)

    except Exception as e:
        print(f"Error starting chatbot: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)