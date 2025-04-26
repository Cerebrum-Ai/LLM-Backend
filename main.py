from flask import Flask, request, jsonify
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from llm_chat import init_llm_input, post_llm_input, LLMManager
from singleton import VectorDBManager
from langchain_core.documents import Document
from threading import Thread, Event
import ngrok
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
load_dotenv()
# Add these constants at the top of the file
NODE_HANDLER_URL = os.environ.get("NODE_HANDLER_URL")
HEARTBEAT_INTERVAL = 30  # seconds
auth_tokens_str = os.environ.get("NGROK_AUTH_TOKENS")
if not auth_tokens_str:
    print("Error: NGROK_AUTH_TOKENS environment variable not set in .env file or environment.")
    print("Please create a .env file in the project root and add the variable:")
    print("NGROK_AUTH_TOKENS=token1,token2,token3,...")
    sys.exit(1) # Exit if the variable is not set
auth = auth_tokens_str.split(',')

# Add this global variable at the top with other constants
current_ngrok_url = None

def image_to_base64_data_uri(input_source):
    """
    Convert an image to base64 data URI format.
    input_source can be either a file path or a URL.
    """
    if input_source.startswith(('http://', 'https://')):
        # If input is a URL, download the image
        response = requests.get(input_source)
        image_data = response.content
    else:
        # If input is a file path, read the file
        with open(input_source, "rb") as img_file:
            image_data = img_file.read()
    
    # Convert to base64
    base64_data = base64.b64encode(image_data).decode('utf-8')
    return f"data:image/png;base64,{base64_data}"


app = Flask(__name__)
llm_instance = None
is_initialized = False
vector_db_instance = None
shutdown_event = Event()

def initialize_llm():
    global llm_instance,vector_db_instance, is_initialized
    try:
        # Initialize Vector DB
        print("Initializing Vector Database...")
        vector_db_instance = VectorDBManager.get_instance()
        if not vector_db_instance.vector_store:
            raise RuntimeError("Failed to initialize Vector Database")
        print("Vector Database initialized successfully")
        
        # Initialize LLM
        print("Initializing LLM...")
        llm_instance = LLMManager.get_instance()
        if llm_instance.llm:
            print("LLM initialized and ready")
            is_initialized = True
        else:
            raise RuntimeError("Failed to initialize LLM")
            is_initialized = False
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        is_initialized = False    


def check_initialization(f):
    def wrapper(*args, **kwargs):
        if not is_initialized:
            return jsonify({
                'status': 'error',
                'message': 'Model is still initializing. Please try again later.'
            }), 503  # Service Unavailable
        return f(*args, **kwargs)
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
        not_none_keys = [k for k, v in data.items() if v not in (None, "", [])]
        initial_response = init_llm_input(state["question"], state.get("image"),not_none_keys)
        state["initial_diagnosis"] = initial_response
        state["answer"] = initial_response
        analysis = {
            "initial_diagnosis": initial_response,
            "vectordb_results": "",
            "final_analysis": ""
        }
        
        return {
            "answer": initial_response, 
            "stage": "initial",
            **analysis
        }
    except Exception as e:
        return {"error": str(e)}

def retrieve(state: State):
    """Vector DB lookup stage"""
    try:
        docs = vector_db_instance.vector_store.similarity_search(state["answer"], k=1)
        state["context"] = docs
        vectordb_results = "\n".join([doc.page_content for doc in docs])
        state["vectordb_results"] = vectordb_results
        
        
        return {
            "answer": vectordb_results, 
            "stage": "retrieve",
            "initial_diagnosis": state["initial_diagnosis"],
            "vectordb_results": vectordb_results,
            "final_analysis": ""
        }
    except Exception as e:
        return {"error": str(e)}

def generate(state: State):
    """Final detailed analysis stage"""
    try:
        docs_content = state["vectordb_results"]
        initial_diagnosis = state["initial_diagnosis"]
        question = state["question"]
        final_response = post_llm_input(state["initial_diagnosis"],state["question"], docs_content)
        state["final_analysis"] = final_response
        
        return {
            "answer": final_response,
            "stage": "final",
            "initial_diagnosis": state["initial_diagnosis"],
            "vectordb_results": state["vectordb_results"],
            "final_analysis": final_response
        }
    except Exception as e:
        return {"answer": str(e)}


def start_router(state: State):
    """Route to the correct start node based on presence of typing, gait, or audio."""
    data = state.get("data", {})
    match data:
        case {"typing": t, **rest} if t not in (None, "", []):
            return "typing_mid"
        case {"gait": g, **rest} if g not in (None, "", []):
            return "gait_mid"
        case {"audio": a, **rest} if a not in (None, "", []):
            return "audio_mid"
        case _:
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
        question = None
        image = None
        gait = None
        audio = None
        typing = None
        # Handle both form data and JSON
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            question = request.form.get('question')
            image = None
            gait = None
            audio = None
            typing = None
            match request.content_type:
                case ct if ct and ct.startswith('multipart/form-data'):
                    question = request.form.get('question')
                    if 'image' in request.files:
                        image_file = request.files['image']
                        image = image_to_base64_data_uri(image_file)
                    if 'gait' in request.files:
                        gait = request.files['gait']
                    if 'audio' in request.files:
                        audio = request.files['audio']
                    if 'typing' in request.form:
                        typing = request.form.get('typing')
                case ct if ct and ct.startswith('application/json'):
                    data = request.get_json(silent=True)
                    question = data.get('question') if data else None
                    if 'image' in data:
                        image = image_to_base64_data_uri(data['image'])
                    if 'gait' in data:
                        gait = data['gait']
                    if 'audio' in data:
                        audio = data['audio']
                    if 'typing' in data:
                        typing = data['typing']
                case _:
                    data = request.get_json(silent=True)
                    question = data.get('question') if data else None

        
        if not question:
            return jsonify({'error': 'Question is required'}), 400

        initial_state = {
            "question": question,
            "image": image,
            "data" : {
                "gait": gait,audio: audio, typing: typing},
            "context": [],
            "answer": "",
            "initial_diagnosis": "",
            "vectordb_results": "",
            "final_analysis": ""
        }

        response = graph.invoke(initial_state)

        # Clean up session after getting response
        if "session_id" in initial_state:
            llm_instance.delete_session(initial_state["session_id"])

        vectordb_data = response.get("vectordb_results", "")
        

        return jsonify({
            'status': 'success',
            'analysis': {
                'initial_diagnosis': response.get("initial_diagnosis"),
                'vectordb_results': vectordb_data,
                'final_analysis': response.get("final_analysis")
            }
        })
          

    except Exception as e:
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
        'overall': is_initialized
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

def register_with_node_handler():
    """Register this node with the node handler"""
    global current_ngrok_url
    try:
        # Wait for node handler to be ready
        if not wait_for_node_handler():
            print("Node handler not ready after multiple attempts")
            return False
            
        print(f"Attempting to register with node handler at {NODE_HANDLER_URL}...")
        response = requests.post(
            f"{NODE_HANDLER_URL}/register", 
            json={"url": current_ngrok_url},
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
    llm_manager = LLMManager.get_instance()
    global current_ngrok_url
    consecutive_failures = 0
    max_consecutive_failures = 1  # Reduced from 3 to 1 to be more responsive
    reconnection_in_progress = False
    
    while not shutdown_event.is_set():
        try:
            if llm_manager.is_inferencing:  # Using the property correctly
                time.sleep(5)  # Wait briefly before checking again
                continue
            if reconnection_in_progress:
                time.sleep(HEARTBEAT_INTERVAL)
                continue
                
            print("\nSending heartbeat to node handler...")
            response = requests.post(
                f"{NODE_HANDLER_URL}/heartbeat",
                json={"url": current_ngrok_url},
                timeout=5
            )
            if response.status_code == 200:
                consecutive_failures = 0  # Reset on success
                reconnection_in_progress = False
                print("✓ Heartbeat successful")
            else:
                print(f"✗ Heartbeat failed with status code: {response.status_code}")
                if response.text:
                    print(f"Response: {response.text[:200]}...")  # Show first 200 chars of response
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print("\n=== Node Handler Connection Lost ===")
                    print("Multiple heartbeat failures detected. Starting reconnection process...")
                    reconnection_in_progress = True
                    if handle_node_handler_disconnect():
                        consecutive_failures = 0
                        reconnection_in_progress = False
                        print("✓ Successfully reconnected to node handler")
        except requests.exceptions.Timeout:
            print("✗ Heartbeat timed out")
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print("\n=== Node Handler Connection Lost ===")
                print("Multiple heartbeat timeouts detected. Starting reconnection process...")
                reconnection_in_progress = True
                if handle_node_handler_disconnect():
                    consecutive_failures = 0
                    reconnection_in_progress = False
                    print("✓ Successfully reconnected to node handler")
        except requests.exceptions.ConnectionError:
            print("✗ Could not connect to node handler")
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print("\n=== Node Handler Connection Lost ===")
                print("Connection error detected. Starting reconnection process...")
                reconnection_in_progress = True
                if handle_node_handler_disconnect():
                    consecutive_failures = 0
                    reconnection_in_progress = False
                    print("✓ Successfully reconnected to node handler")
        except Exception as e:
            print(f"✗ Error sending heartbeat: {str(e)}")
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

def deregister_from_node_handler():
    """Deregister this node from the node handler"""
    global current_ngrok_url
    try:
        response = requests.post(
            f"{NODE_HANDLER_URL}/deregister",
            json={"url": current_ngrok_url},
            timeout=5
        )
        if response.status_code == 200:
            print("Successfully deregistered from node handler")
            return True
        else:
            print(f"Failed to deregister from node handler: {response.text}")
            return False
    except Exception as e:
        print(f"Error deregistering from node handler: {str(e)}")
        return False

@app.route('/deregister', methods=['POST'])
def handle_deregister():
    """Handle deregistration request from node handler"""
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
                        if handle_node_handler_disconnect():
                            print("✓ Successfully reconnected to node handler")
                            return
                        print(f"✗ Reconnection attempt {attempt + 1} failed")
                        print(f"Waiting {retry_delay} seconds before next attempt...")
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 30)
                    except Exception as e:
                        print(f"✗ Error during reconnection attempt {attempt + 1}: {str(e)}")
                        if attempt < max_retries - 1:
                            print(f"Waiting {retry_delay} seconds before next attempt...")
                            time.sleep(retry_delay)
                            retry_delay = min(retry_delay * 2, 30)
                
                print("\n=== Background Reconnection ===")
                print("Initial reconnection attempts failed. Starting background reconnection process...")
                attempt_count = 0
                while not shutdown_event.is_set():
                    try:
                        attempt_count += 1
                        print(f"\nBackground reconnection attempt {attempt_count}")
                        print(f"Trying to connect to node handler at {NODE_HANDLER_URL}...")
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

def run_ngrok():
    """Synchronous wrapper for async ngrok handler"""
    global current_ngrok_url
    global i
    listener = None
    try:
        # Initialize ngrok with your authtoken
        ngrok.set_auth_token(auth[i])
        # Wait a moment for cleanup
        time.sleep(2)
        # Start new tunnel
        listener = ngrok.forward(
            5050
        )
        current_ngrok_url = listener.url()
        print(f"Ingress established at: {current_ngrok_url}")
        
        # Register with node handler after ngrok is set up
        if not register_with_node_handler():
            print("Failed to register with node handler. Will keep trying...")
            # Start a background thread to keep trying registration
            def registration_retry():
                while not shutdown_event.is_set():
                    if register_with_node_handler():
                        break
                    time.sleep(10)  # Wait 10 seconds between attempts
            
            registration_thread = threading.Thread(target=registration_retry, daemon=True)
            registration_thread.start()
        
        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
        heartbeat_thread.start()

        while not shutdown_event.is_set():
            time.sleep(1)  # Check every second
                  
    except Exception as e:
        if "ERR_NGROK_108" in str(e):
            print("Warning: Ngrok session limit reached. This node will not be accessible externally.")
            print("Please ensure only one ngrok session is running at a time.")
            if i < len(auth) - 1:
                i += 1
                print(f"Retrying with next authtoken (attempt {i + 1}/{len(auth)})...")
                run_ngrok()
            else:
                print("All authtokens exhausted. Exiting...")
                return
        if shutdown_event.is_set():
            print("Ngrok interrupted by shutdown signal.")
        else:
            print(f"Ngrok error: {str(e)}")
    finally:
        print("Cleaning up...")
        if listener:
            try:
                
                ngrok.disconnect()
                print("Ngrok disconnected successfully")
                deregister_from_node_handler()
            except Exception as e:
                print(f"Error disconnecting ngrok: {str(e)}")

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

if __name__ == '__main__':
    global i 
    i = 0
    try:
        
        # Set up signal handlers for graceful shutdown
        
        def handle_shutdown(signum, frame):
            print("\nReceived shutdown signal. Cleaning up...")
            shutdown_event.set()
            # Give threads time to clean up
            time.sleep(2)
            sys.exit(0)
        
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
        
        init_thread = Thread(target=initialize_llm, daemon=True)
        ngrok_thread = Thread(target=run_ngrok, daemon=True)
        print("Starting LLM/VectorDB initialization...")
        init_thread.start()
        init_thread.join()
        print("Initialization complete.")
        # Start Flask app in a separate thread
        flask_port = 5050
        flask_thread = Thread(target=lambda: serve(app, host='0.0.0.0', port=flask_port, threads=4), daemon=True) 
        print(f"Starting Flask app on port {flask_port}...")
        flask_thread.start()

        # Wait for Flask app to be ready before starting ngrok
        if not wait_for_flask(flask_port):
             print("Flask app failed to start. Exiting.")
             sys.exit(1)
        print("Flask app is running.")

        # Now start ngrok in its thread
        ngrok_thread = Thread(target=run_ngrok, daemon=True)
        print("Starting ngrok tunnel...")
        ngrok_thread.start()

        # Keep the main thread alive to handle signals
        while not shutdown_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down application...")
    finally:
        shutdown_event.set()
        # Ensure we deregister before exiting
        deregister_from_node_handler()
        sys.exit(0)
        
