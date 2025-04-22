from flask import Flask, request, jsonify
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from llm_chat import init_llm_input, post_llm_input, LLMManager
from singleton import VectorDBManager
from langchain_core.documents import Document
from threading import Thread
import ngrok
import time
import threading
import sys




import base64

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"


app = Flask(__name__)
llm_instance = None
is_initialized = False
vector_db_instance = None

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


initialize_llm()

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
        
        initial_response = init_llm_input(state["question"], state.get("image"))
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


# Initialize the graph
graph_builder = StateGraph(State)
graph_builder.add_node(init)
graph_builder.add_node(retrieve)
graph_builder.add_node(generate)
graph_builder.add_edge(START, "init")
graph_builder.add_edge("init", "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph = graph_builder.compile()

@app.route('/chat', methods=['POST'])
@check_initialization
def chat():
    try:
        # Handle both form data and JSON
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            question = request.form.get('question')
            image = None
            if 'image' in request.files:
                image_file = request.files['image']
                image = image_to_base64_data_uri(image_file)
        else:
            data = request.get_json(silent=True)
            question = data.get('question') if data else None
            image = None
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400

        initial_state = {
             "question": question,
            "image": image,
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
        if vectordb_data:
            # Split by commas and take every third item
            items = vectordb_data.split(',')
            filtered_items = items[::3]
            processed_vectordb = ','.join(filtered_items) if filtered_items else ""
        else:
            processed_vectordb = ""

        return jsonify({
            'status': 'success',
            'analysis': {
                'initial_diagnosis': response.get("initial_diagnosis"),
                'vectordb_results': processed_vectordb,
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

def run_ngrok():
    """Synchronous wrapper for async ngrok handler"""
    try:
        # Initialize ngrok with your authtoken
        ngrok.set_auth_token("2uHfQak8CxEWThSHdtiVOIC14kq_5px1cRz1ammGDZYnboB2g")
        # Wait a moment for cleanup
        time.sleep(2)
        # Start new tunnel
        listener = ngrok.forward(
            5000,
            domain="monthly-vital-reptile.ngrok-free.app"
        )
        print(f"Ingress established at: {listener.url()}")
        
        while True:
            time.sleep(1)
            
    except Exception as e:
        print(f"Ngrok error: {str(e)}")
        raise
    finally:
        try:
            ngrok.disconnect()
        except:
            pass

def wait_for_flask(port, max_retries=5):
    """Wait for Flask to start accepting connections"""
    import socket
    import time
    
    for i in range(max_retries):
        try:
            with socket.create_connection(("localhost", port), timeout=1.0):
                return True
        except (ConnectionRefusedError, socket.timeout):
            print(f"Waiting for Flask to start (attempt {i+1}/{max_retries})...")
            time.sleep(2)
    return False

if __name__ == '__main__':
    try:
        init_thread = Thread(target=initialize_llm)
        init_thread.start()
        flask_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False))
        flask_thread.start()
        if wait_for_flask(5000):
            print("Flask server is ready")
            # Now start ngrok
            run_ngrok()
        else:
            print("Failed to start Flask server")
            sys.exit(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        ngrok.disconnect()
        exit()
        