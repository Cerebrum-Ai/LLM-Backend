from flask import Flask, request, jsonify
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from llm_chat import init_llm_input, post_llm_input
from singleton import LLMManager, VectorDBManager
from langchain_core.documents import Document
from threading import Thread, Event
# import ngrok # Removed ngrok import
import time
import sys
import requests
import threading
import atexit
from dotenv import load_dotenv
import base64
import os
import signal
from waitress import serve
import subprocess # Import subprocess for running localtunnel command
import tempfile
from models.workflow.er_triage import ERTriageModel
from models.workflow.lab_analysis import LabAnalysisModel
from models.workflow.alert_system import AlertSystem
from datetime import datetime
from openrouter_llm import OpenRouterLLMManager
from llm_chat import set_llm_instance
import shutil     # Import shutil to find the localtunnel executable
import platform   # Import platform to check the operating system

load_dotenv()

# Add these constants at the top of the file
NODE_HANDLER_URL = os.environ.get("NODE_HANDLER_URL")
HEARTBEAT_INTERVAL = 30  # seconds
# Removed ngrok specific auth tokens
# auth_tokens_str = os.environ.get("NGROK_AUTH_TOKENS")

# Add this global variable at the top with other constants
# Removed current_ngrok_url
# current_ngrok_url = None
current_tunnel_url = None # Variable to store the public localtunnel URL
# Removed index for auth tokens
# i = 0  # Initialize index for auth tokens

USE_OPENROUTER = os.environ.get("USE_OPENROUTER", "false").lower() == "true"

# Check required environment variables
# Removed check for auth_tokens_str
# if not auth_tokens_str:
#     print("Error: NGROK_AUTH_TOKENS environment variable not set in .env file or environment.")
#     print("Please create a .env file in the project root and add the variable:")
#     print("NGROK_AUTH_TOKENS=token1,token2,token3,...")
#     sys.exit(1) # Exit if the variable is not set

if not NODE_HANDLER_URL:
    print("Error: NODE_HANDLER_URL environment variable not set in .env file or environment.")
    print("Please create a .env file in the project root and add the variable:")
    print("NODE_HANDLER_URL=http://your-node-handler-url")
    sys.exit(1) # Exit if the variable is not set

if USE_OPENROUTER and not os.environ.get("OPENROUTER_API_KEY"):
    print("Error: OPENROUTER_API_KEY environment variable not set but USE_OPENROUTER=true")
    print("Please add OPENROUTER_API_KEY to your .env file")
    sys.exit(1)

# Removed splitting auth tokens
# auth = auth_tokens_str.split(',')

app = Flask(__name__)
llm_instance = None
is_initialized = False
vector_db_instance = None
shutdown_event = Event()
localtunnel_process = None # Global variable to store the localtunnel subprocess

# Initialize workflow models
er_triage_model = ERTriageModel()
lab_analysis_model = LabAnalysisModel()
alert_system = AlertSystem()

def initialize_llm():
    global llm_instance, vector_db_instance, is_initialized
    try:

        # Initialize Vector DB
        print("Initializing Vector Database...")
        vector_db_instance = VectorDBManager.get_instance()
        if not vector_db_instance:
            raise RuntimeError("Failed to initialize VectorDB")

        # Initialize LLM
        print(f"Initializing LLM (Using OpenRouter: {USE_OPENROUTER})...")
        if USE_OPENROUTER:
            llm_instance = OpenRouterLLMManager.get_instance()
            if not llm_instance.llm:
                raise RuntimeError("Failed to initialize OpenRouter LLM")
            set_llm_instance(llm_instance)
            print("OpenRouter LLM initialized successfully")
            is_initialized = True
            return True

        # Only initialize local LLM if not using OpenRouter
        llm_instance = LLMManager.get_instance()
        if not llm_instance.llm:
            raise RuntimeError("Failed to initialize Local LLM")
        set_llm_instance(llm_instance)
        print("Local LLM initialized successfully")

        print("LLM initialized successfully")

        is_initialized = True
        chat_llm_instance = llm_instance
        print("All components initialized and ready")
    except Exception as e:
        print(f"Error initializing components: {str(e)}")
        is_initialized = False

def check_initialization(f):
    def wrapper(*args, **kwargs):
        if not is_initialized:
            return jsonify({
                'status': 'error',
                'message': 'Model is still initializing. Please try again later.'
            }), 503  # Service Unavailable
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__  # Give the wrapper the same name as the decorated function
    return wrapper

class State(TypedDict):
    question: str
    context: List[Document]
    data: dict
    answer: str
    image: str
    initial_diagnosis: str
    vectordb_results: str
    final_analysis: str

def init(state: State):
    """Initial diagnosis stage"""
    try:
        session_id = llm_instance.create_session()
        state["session_id"] = session_id
        data = state.get("data", {})

        # Extract ML results for LLM context
        ml_results = {}
        # Audio emotion (legacy)
        if isinstance(data.get("audio"), dict) and "emotion" in data["audio"]:
            ml_results["audio"] = data["audio"]["emotion"]
        # New: merge ML results from *_ml fields, INCLUDE ONLY audio_ml and typing_ml for init_llm_input
        for ml_field in ["audio_ml", "typing_ml"]:  # skip image_ml and gait_ml
            if ml_field in data and data[ml_field] is not None:
                ml_results[ml_field.replace('_ml', '')] = data[ml_field]

        # Get initial diagnosis WITH audio and typing ML results
        initial_response = init_llm_input(
            question=state["question"],
            image=state.get("image"),
            ml_results=ml_results if ml_results else None
        )

        state["initial_diagnosis"] = initial_response
        state["answer"] = initial_response
        state["ml_results"] = ml_results  # Save for downstream use
        state["not_none_keys_data"] = data  # Save full data for downstream use

        analysis = {
            "initial_diagnosis": initial_response,
            "vectordb_results": "",
            "final_analysis": ""
        }

        return {
            "answer": state["answer"],
            **analysis
        }
    except Exception as e:
        print(f"Error in init stage: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def retrieve(state: State):
    """Vector DB lookup stage"""
    try:
        answer_text = state['answer'] if isinstance(state['answer'], str) else str(state['answer'])
        print(f"Looking up similar data in vector DB for: {answer_text[:50]}...")
        docs = vector_db_instance.vector_store.similarity_search(answer_text, k=1)
        state["context"] = docs
        vectordb_results = "\n".join([doc.page_content for doc in docs])
        state["vectordb_results"] = vectordb_results

        print(f"Retrieved {len(docs)} documents from vector DB")

        return {
            "answer": vectordb_results,
            "stage": "retrieve",
            "initial_diagnosis": state["initial_diagnosis"],
            "vectordb_results": vectordb_results,
            "final_analysis": "",
        }
    except Exception as e:
        print(f"Error in retrieve stage: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def generate(state: State):
    """Final generation stage"""
    try:
        # Use the initial diagnosis and context to generate a final answer
        initial_diagnosis = state.get("initial_diagnosis", "")
        question = state.get("question", "")
        context = state.get("context", [])
        # Use audio, typing, and image ML results
        ml_results = {}
        data = state.get("data", {})
        # Audio emotion (legacy)
        if isinstance(data.get("audio"), dict) and "emotion" in data["audio"]:
            ml_results["audio"] = data["audio"]["emotion"]
        # Include audio_ml, typing_ml, and image_ml
        for ml_field in ["audio_ml", "typing_ml", "image_ml"]:
            if ml_field in data and data[ml_field] is not None:
                ml_results[ml_field.replace('_ml', '')] = data[ml_field]

        # Call post_llm_input for final answer
        final_answer = post_llm_input(
            initial_diagnosis=initial_diagnosis,
            question=question,
            context=context,
            ml_results=ml_results
        )

        final_answer_text = final_answer if isinstance(final_answer, str) else str(final_answer)
        print(f"Final response generated: {final_answer_text[:100]}...")
        state["final_analysis"] = final_answer

        return {
            "answer": final_answer,
            "stage": "final",
            "initial_diagnosis": state["initial_diagnosis"],
            "vectordb_results": state["vectordb_results"],
            "final_analysis": final_answer
        }
    except Exception as e:
        print(f"Error in generate stage: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"answer": str(e)}

def start_router(state: State):
    """Route to the correct start node based on presence of typing, gait, or audio."""
    data = state.get("data", {})
    if data.get("typing") not in (None, "", []):
        return "typing_mid"
    elif data.get("gait") not in (None, "", []):
        return "gait_mid"
    elif data.get("audio") not in (None, "", []):
        return "audio_mid"
    else:
        return "norm_mid"

# Initialize the graph
graph_builder = StateGraph(State)
graph_builder.add_node(init)
graph_builder.add_node(retrieve)
graph_builder.add_node(generate)
graph_builder.add_edge(START, "init")
graph_builder.add_edge("init", "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph = graph_builder.compile()

@app.route('/api/chat', methods=['POST'])
@app.route('/chat', methods=['POST'])
@check_initialization
def chat():
    try:
        llm_type = "OpenRouter" if USE_OPENROUTER else "Local"
        print(f"Using {llm_type} LLM for processing")

        question = None
        image = None
        gait = None
        audio = None
        typing = None

        # Handle form data
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            question = request.form.get('question')
            if 'image' in request.files:
                image_file = request.files['image']
                image = image_file
            if 'gait' in request.files:
                gait = request.files['gait']
            if 'audio' in request.files:
                audio = request.files['audio']
                print(f"Received audio file in chat endpoint: {audio.filename}, content_type: {audio.content_type}")
            if 'typing' in request.form:
                typing = request.form.get('typing')
        # Handle JSON data
        elif request.content_type and request.content_type.startswith('application/json'):
            data = request.get_json(silent=True)
            if data:
                question = data.get('question')
                if 'image' in data:
                    image = data['image']
                if 'gait' in data:
                    gait = data['gait']
                if 'audio' in data:
                    audio = data['audio']
                    print(f"Received audio data in JSON format in chat endpoint")
                if 'typing' in data:
                    typing = data['typing']
                # Extract all ML result fields
                image_ml = data.get('image_ml')
                audio_ml = data.get('audio_ml')
                typing_ml = data.get('typing_ml')
                gait_ml = data.get('gait_ml')
        # Fallback for other content types
        else:
            data = request.get_json(silent=True)
            question = data.get('question') if data else None
            image_ml = audio_ml = typing_ml = gait_ml = None

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        # Log the data we're processing
        print(f"Processing chat request with question: {question}")
        print(f"Audio data present: {audio is not None}")

        initial_state = {
            "question": question,
            "image": image,
            "data": {
                "gait": gait,
                "audio": audio,
                "typing": typing,
                "image_ml": image_ml,
                "audio_ml": audio_ml,
                "typing_ml": typing_ml,
                "gait_ml": gait_ml
            },
            "context": [],
            "answer": "",
            "initial_diagnosis": "",
            "vectordb_results": "",
            "final_analysis": ""
        }

        # Process the request through the graph
        print("Invoking processing graph...")
        response = graph.invoke(initial_state)
        print("Graph processing completed")

        # Clean up session after getting response
        if "session_id" in initial_state:
            llm_instance.delete_session(initial_state["session_id"])

         # Ensure all values are JSON serializable
        initial_diagnosis = response.get("initial_diagnosis", "")
        if hasattr(initial_diagnosis, 'content'):
            initial_diagnosis = initial_diagnosis.content
        elif not isinstance(initial_diagnosis, str):
            initial_diagnosis = str(initial_diagnosis)

        vectordb_results = response.get("vectordb_results", "")
        if hasattr(vectordb_results, 'content'):
            vectordb_results = vectordb_results.content
        elif not isinstance(vectordb_results, str):
            vectordb_results = str(vectordb_results)

        final_analysis = response.get("final_analysis", "")
        if hasattr(final_analysis, 'content'):
            final_analysis = final_analysis.content
        elif not isinstance(final_analysis, str):
            final_analysis = str(final_analysis)


        # Format API response
        return jsonify({
            'status': 'success',
            'analysis': {
                'initial_diagnosis': initial_diagnosis,
                'vectordb_results': vectordb_results,
                'final_analysis': final_analysis,
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/status', methods=['GET'])
def get_status():
    status = {
        'llm': bool(llm_instance and llm_instance.llm),
        'vector_db': bool(vector_db_instance and vector_db_instance.vector_store),
        # Remove audio_analyzer check as it's now in models.py
        'overall': is_initialized,
        'tunnel_url': current_tunnel_url # Include the tunnel URL in status
    }

    return jsonify({
        'status': 'ready' if is_initialized else 'initializing',
        'components': status,
        'message': 'All systems ready' if is_initialized else 'Systems are initializing'
    })

def wait_for_node_handler(max_retries=10):
    """Wait for node handler to be ready"""
    retry_count = 0
    retry_delay = 2  # Start with 2 seconds

    while retry_count < max_retries:
        try:
            print(f"Attempting to connect to node handler (attempt {retry_count + 1}/{max_retries})...")
            # Try the status endpoint
            response = requests.get(f"{NODE_HANDLER_URL}/status", timeout=5)
            if response.status_code == 200:
                print("Successfully connected to node handler")
                return True
            else:
                print(f"Unexpected status code: {response.status_code}")
        except requests.exceptions.Timeout:
            print("Connection timed out")
        except requests.exceptions.ConnectionError:
            print("Could not connect to node handler")
        except Exception as e:
            print(f"Error connecting to node handler: {str(e)}")

        # Exponential backoff
        print(f"Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 30)  # Cap at 30 seconds
        retry_count += 1

    print("Failed to connect to node handler after multiple attempts")
    return False

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
        # Start localtunnel process using the found path and forwarding port 5050
        # Use '--port' argument to specify the local port
        process_command = [lt_path, '--port', '5050'] # Use port 5050 for this service

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


def register_with_node_handler():
    """Register this node with the node handler"""
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
            json={"url": current_tunnel_url, "type": "llm"}, # Send the localtunnel URL
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
    """Deregister this node from the node handler"""
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

@app.route('/deregister', methods=['POST'])
def handle_deregister():
    """Handle deregistration request from node handler"""
    try:
        data = request.json
        # Use current_tunnel_url for comparison
        if data and data.get('url') == current_tunnel_url:
            print("\n=== Node Handler Shutdown Notice ===")
            print("Received deregister request from node handler")
            print("Node handler is shutting down. Preparing for reconnection...")

            # Start reconnection in a background thread
            def reconnect_loop():
                print("\n=== Reconnection Process ===")
                print("Starting reconnection attempts...")
                # handle_node_handler_disconnect now includes tunnel restart logic
                if handle_node_handler_disconnect():
                     print("✓ Successfully reconnected to node handler")
                else:
                    print("\n=== Background Reconnection ===")
                    print("Initial reconnection attempts failed. Starting background reconnection process...")
                    attempt_count = 0
                    while not shutdown_event.is_set():
                        try:
                            attempt_count += 1
                            print(f"\nBackground reconnection attempt {attempt_count}")
                            # handle_node_handler_disconnect attempts both tunnel restart and registration
                            if handle_node_handler_disconnect():
                                print("✓ Successfully reconnected to node handler")
                                break
                            print("✗ Reconnection failed")
                            print("Waiting 10 seconds before next attempt...")
                            time.sleep(10)
                        except Exception as e:
                            print(f"✗ Error in background reconnection: {str(e)}")
                            print("Waiting 10 seconds before next attempt...")
                            time.sleep(10)

            # Start the reconnection process in a background thread
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


def send_heartbeat():
    """Send periodic heartbeat to node handler"""
    global current_tunnel_url, HEARTBEAT_INTERVAL  # Use the renamed variable
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
                json={"url": current_tunnel_url}, # Use the renamed variable
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

# Removed ngrok shutdown logic from handle_shutdown
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


@app.route('/api/er/triage', methods=['POST'])
@check_initialization
def er_triage():
    """Handle ER triage requests"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Extract required fields
        vitals = data.get('vitals', {})
        symptoms = data.get('symptoms', [])
        medical_history = data.get('medical_history', {})

        # Validate required fields
        if not vitals or not symptoms:
            return jsonify({'error': 'Missing required fields (vitals, symptoms)'}), 400

        # Process triage
        result = er_triage_model.assess_patient(vitals, symptoms, medical_history)

        # Check for critical findings and send alerts
        if result.get('critical_findings'):
            for finding in result['critical_findings']:
                alert_system.send_alert(
                    message=finding,
                    level='CRITICAL',
                    source='ER_TRIAGE',
                    data={'vitals': vitals, 'symptoms': symptoms}
                )

        return jsonify(result), 200

    except Exception as e:
        print(f"Error in ER triage: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/lab/analyze', methods=['POST'])
@check_initialization
def lab_analysis():
    """Handle lab result analysis requests"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Extract required fields
        lab_results = data.get('results', {})
        previous_results = data.get('previous_results', {})

        # Validate required fields
        if not lab_results:
            return jsonify({'error': 'Missing required field (results)'}), 400

        # Process lab results
        result = lab_analysis_model.analyze_results(lab_results, previous_results)

        # Check for critical findings and send alerts
        if result.get('critical_findings'):
            for finding in result['critical_findings']:
                alert_system.send_alert(
                    message=finding['message'],
                    level='CRITICAL',
                    source='LAB_ANALYSIS',
                    data={'test': finding['test'], 'value': finding['value']}
                )

        return jsonify(result), 200

    except Exception as e:
        print(f"Error in lab analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
@check_initialization
def get_alerts():
    """Get alert history with optional filtering"""
    try:
        # Get filter parameters
        level = request.args.get('level')
        source = request.args.get('source')
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')

        # Convert time strings to datetime if provided
        if start_time:
            start_time = datetime.fromisoformat(start_time)
        if end_time:
            end_time = datetime.fromisoformat(end_time)

        # Get filtered alerts
        alerts = alert_system.get_alert_history(
            level=level,
            source=source,
            start_time=start_time,
            end_time=end_time
        )

        return jsonify({
            'alerts': alerts,
            'count': len(alerts)
        }), 200

    except Exception as e:
        print(f"Error getting alerts: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/clear', methods=['POST'])
@check_initialization
def clear_alerts():
    """Clear alert history"""
    try:
        alert_system.clear_alert_history()
        return jsonify({'message': 'Alert history cleared'}), 200
    except Exception as e:
        print(f"Error clearing alerts: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    try:
        # Set up signal handlers for graceful shutdown
        # The handle_shutdown function is defined above and includes localtunnel cleanup
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        # Initialize LLM with timeout
        init_thread = Thread(target=initialize_llm, daemon=True)
        print("Starting LLM/VectorDB initialization...")
        init_thread.start()

        # Add timeout for initialization
        max_init_wait = 300  # 5 minutes timeout
        init_thread.join(timeout=max_init_wait)
        if init_thread.is_alive():
            print(f"Warning: Initialization is taking longer than {max_init_wait} seconds")
            print("Continuing startup process anyway...")
        else:
            print("Initialization complete.")

        # Start Flask app in a separate thread
        flask_port = 5050
        flask_thread = Thread(target=lambda: serve(app, host='0.0.0.0', port=flask_port, threads=4), daemon=True)
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
        localtunnel_thread = Thread(target=run_localtunnel, daemon=True)
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

