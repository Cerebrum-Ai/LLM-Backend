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

app = Flask(__name__)

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

def send_heartbeat():
    global current_ngrok_url
    while not shutdown_event.is_set():
        try:
            requests.post(f"{NODE_HANDLER_URL}/heartbeat", json={"url": current_ngrok_url}, timeout=5)
        except Exception as e:
            print(f"Heartbeat error: {str(e)}")
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

    # (ML processing would go here)
    print(f"Received ML request: url={url}, type={data_type}, model={model}")

    # Respond with acknowledgement
    return jsonify({
        'status': 'received',
        'url': url,
        'data_type': data_type,
        'model': model
    }), 200

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
                            print("✓ Successfully reconnected to node handler")
                            return
                        print(f"✗ Reconnection attempt {attempt + 1} failed")
                        print(f"Waiting {retry_delay} seconds before next attempt...")
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 30)
                    except Exception as e:
                        print(f"✗ Error in background reconnection: {str(e)}")
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