from flask import Flask, request, jsonify, Response
import ngrok
import sqlite3
import threading
import time
from datetime import datetime
import requests
import sys
import atexit
from waitress import serve # Import waitress
import os # Keep os import
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)

# Initialize SQLite database
def init_db():
    """Initialize database with reconnection logic and node type column"""
    max_retries = 3
    retry_delay = 1  # seconds
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect('nodes.db', timeout=20)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS nodes
                         (url TEXT PRIMARY KEY,
                          status TEXT,
                          last_heartbeat TIMESTAMP,
                          type TEXT)''')
            conn.commit()
            conn.close()
            print("Database initialized successfully")
            return
        except sqlite3.Error as e:
            print(f"Database initialization attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise

# Initialize database
init_db()

def get_db():
    """Get database connection with reconnection logic"""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect('nodes.db', timeout=20)  # Add timeout
            conn.row_factory = sqlite3.Row
            # Test the connection
            conn.execute('SELECT 1')
            return conn
        except sqlite3.Error as e:
            print(f"Database connection attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise

# Define all routes before starting the server
@app.route('/')
def root():
    """Root endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Node handler is running',
        'endpoints': {
            'register': '/register',
            'deregister': '/deregister',
            'heartbeat': '/heartbeat',
            'chat': '/chat',
            'status': '/status'
        }
    })

@app.route('/status', methods=['GET'])
def status():
    """Health check endpoint"""
    try:
        conn = get_db()
        c = conn.cursor()
        try:
            c.execute('SELECT COUNT(*) FROM nodes WHERE status = "active"')
            active_nodes = c.fetchone()[0]
            c.execute('SELECT COUNT(*) FROM nodes WHERE status = "active" AND type = "llm"')
            llm_nodes = c.fetchone()[0]
            c.execute('SELECT COUNT(*) FROM nodes WHERE status = "active" AND type = "ml"')
            ml_nodes = c.fetchone()[0]
            return jsonify({
                'status': 'ok',
                'message': 'Node handler is running',
                'active_nodes': active_nodes,
                'llm_nodes': llm_nodes,
                'ml_nodes': ml_nodes
            })
        finally:
            conn.close()
    except Exception as e:
        print(f"Error in status endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/register', methods=['POST'])
def register_node():
    """Register a node with the node handler"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'Invalid registration data'}), 400

        url = data['url']
        node_type = data.get('type', 'ml')  # Default to 'ml' for backward compatibility
        
        # Skip the reachability check as it's causing issues with ngrok URLs
        # Instead, just log the registration attempt
        print(f"\n=== New Node Registration ===")
        print(f"Registering {node_type} node: {url}")

        # Add to database
        conn = get_db()
        c = conn.cursor()
        
        c.execute('SELECT * FROM nodes WHERE url = ?', (url,))
        existing = c.fetchone()
        
        if existing:
            c.execute('UPDATE nodes SET status = "active", last_heartbeat = CURRENT_TIMESTAMP, type = ? WHERE url = ?', 
                     (node_type, url))
            print(f"Updated existing node: {url}")
        else:
            c.execute('INSERT INTO nodes (url, status, last_heartbeat, type) VALUES (?, ?, CURRENT_TIMESTAMP, ?)', 
                     (url, "active", node_type))
            print(f"Added new node: {url}")
        
        conn.commit()
        conn.close()
        
        # If it's a chatbot node, also track it in the chatbot list
        with chatbot_nodes_lock:
            if node_type == 'chatbot':
                if url not in chatbot_nodes:
                    chatbot_nodes.append(url)
                print(f"Registered chatbot node: {url}")
            else:
                print(f"Registered ML node: {url}")
            
        return jsonify({'status': 'ok', 'message': f'{node_type} node registered successfully'})
    except Exception as e:
        print(f"Error in register_node: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/deregister', methods=['POST'])
def deregister_node():
    """Deregister a node from the node handler"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'Invalid deregistration data'}), 400

        url = data['url']
        
        # Remove from database
        conn = get_db()
        c = conn.cursor()
        c.execute('DELETE FROM nodes WHERE url = ?', (url,))
        conn.commit()
        conn.close()
        
        # Also remove from chatbot list if it's there
        with chatbot_nodes_lock:
            if url in chatbot_nodes:
                chatbot_nodes.remove(url)
                print(f"Deregistered chatbot node: {url}")
            else:
                print(f"Deregistered ML node: {url}")
        
        return jsonify({'status': 'ok', 'message': 'Node deregistered successfully'})
    except Exception as e:
        print(f"Error in deregister_node: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/heartbeat', methods=['POST'])
def update_heartbeat():
    """Update the heartbeat timestamp for a node"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required'}), 400
        node_url = data['url']
        print(f"\n=== Heartbeat Update ===")
        print(f"Updating heartbeat for node: {node_url}")
        conn = get_db()
        try:
            c = conn.cursor()
            c.execute('UPDATE nodes SET last_heartbeat = CURRENT_TIMESTAMP WHERE url = ?', (node_url,))
            conn.commit()
            print(f"✓ Heartbeat updated for node: {node_url}")
            return jsonify({'status': 'success'})
        except Exception as e:
            print(f"✗ Error updating heartbeat: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            conn.close()
    except Exception as e:
        print(f"✗ Error in update_heartbeat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Node handler is ready'})

# Renamed from run_flask and removed server start logic
def initialize_app():
    """Initialize the Flask application and database"""
    try:
        # Register cleanup handlers
        atexit.register(cleanup)

        # Initialize database
        init_db()
        print("\n=== Database Initialization ===")
        print("✓ Database initialized successfully")

        # Clear any existing data from previous runs
        print("\n=== Clearing Previous Data ===")
        clear_database()
        print("✓ Database cleared successfully")

        # Print all registered routes for debugging
        print("\n=== Registered Routes ===")
        for rule in app.url_map.iter_rules():
            print(f"✓ {rule.endpoint}: {rule.methods} {rule}")

        print("✓ Application initialized successfully")
        return True

    except Exception as e:
        print(f"✗ Error during app initialization: {str(e)}")
        return False

def wait_for_server_ready(url='http://localhost:8000/health', timeout=30):
    """Wait for the WSGI server to be ready"""
    start_time = time.time()
    max_retries = 10
    retry_delay = 1

    print(f"Waiting for server to be ready at {url}...")
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=1.0)
            if response.status_code == 200:
                print(f"✓ Server is ready and responding (took {time.time() - start_time:.2f}s)")
                return True
            else:
                print(f"Health check returned {response.status_code}, retrying...")
        except (ConnectionRefusedError, requests.exceptions.RequestException) as e:
            print(f"Waiting for server (attempt {attempt + 1}/{max_retries}): {e}")

        if time.time() - start_time > timeout:
            print(f"✗ Server did not become ready within {timeout} seconds.")
            return False

        time.sleep(retry_delay)
        # Optional: Increase delay slightly for subsequent retries
        # retry_delay = min(retry_delay + 0.5, 5)

    print(f"✗ Server failed to start after {max_retries} attempts.")
    return False


def run_ngrok():
    try:
        # Wait for Flask to be ready
        max_retries = 10
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.get('http://localhost:8000/status')
                if response.status_code < 500:
                    print("Flask app is ready")
                    break
            except Exception:
                pass
            print(f"Waiting for Flask to be ready (attempt {retry_count + 1}/{max_retries})...")
            time.sleep(2)
            retry_count += 1
        
        if retry_count == max_retries:
            raise Exception("Flask app failed to start")
            
        # Now that Flask is running, set up ngrok
        print("Setting up ngrok tunnel...")
        ngrok.set_auth_token(os.environ.get("NODE_AUTH_TOKEN"))
        time.sleep(2)
        
        try:
            # Start ngrok with a fixed domain
            print("Starting ngrok tunnel...")
            listener = ngrok.forward(
                8000,
                domain=os.environ.get("NGROK_HANDLER_URL")
            )
            
            # Verify tunnel is working
            retry_count = 0
            while retry_count < max_retries:
                try:
                    print(f"Verifying tunnel (attempt {retry_count + 1}/{max_retries})...")
                    response = requests.get(listener.url())
                    if response.status_code < 500:
                        print(f"Tunnel verified at: {listener.url()}")
                        return listener.url()
                except Exception as e:
                    print(f"Tunnel verification failed: {str(e)}")
                    time.sleep(2)
                    retry_count += 1
            
            raise Exception("Failed to establish ngrok tunnel after multiple attempts")
            
        except Exception as e:
            if "ERR_NGROK_108" in str(e):
                print("Error: Failed to establish ngrok tunnel. Please ensure no other ngrok sessions are running.")
                print("The node handler requires a single ngrok tunnel to function properly.")
                sys.exit(1)
            raise
            
    except Exception as e:
        print(f"Ngrok error: {str(e)}")
        raise

def cleanup():
    """Cleanup function to be called on shutdown"""
    print("Starting cleanup...")
    try:
        # Notify all nodes of shutdown and wait for responses
        notify_nodes_of_shutdown()
        
        # Clear the database
        print("Clearing database...")
        clear_database()
        
        # Disconnect ngrok
        try:
            print("Disconnecting ngrok...")
            ngrok.disconnect()
            print("Ngrok disconnected successfully")
        except Exception as e:
            print(f"Error disconnecting ngrok: {str(e)}")
            
        print("Cleanup completed successfully")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

def get_active_nodes():
    """Get list of active nodes"""
    conn = get_db()
    c = conn.cursor()
    try:
        # Only return nodes that have sent a heartbeat in the last 5 minutes
        c.execute('SELECT url FROM nodes WHERE status = "active" AND last_heartbeat > datetime("now", "-5 minutes")')
        return [row['url'] for row in c.fetchall()]
    finally:
        conn.close()

def notify_nodes_of_shutdown():
    """Notify all registered nodes that the handler is shutting down"""
    print("Notifying nodes of shutdown...")
    conn = get_db()
    c = conn.cursor()
    try:
        # Get all active nodes
        c.execute('SELECT url FROM nodes WHERE status = "active"')
        nodes = c.fetchall()
        
        # Track which nodes have responded
        responded_nodes = set()
        max_wait_time = 30  # Maximum time to wait for responses
        start_time = time.time()
        
        for node in nodes:
            try:
                print(f"Notifying node {node[0]} of shutdown...")
                # Send shutdown notification to each node with longer timeout
                response = requests.post(
                    f"{node[0]}/deregister",
                    json={'url': node[0]},
                    timeout=15  # Increased timeout to 10 seconds
                )
                if response.status_code in [200, 202]:  # Accept both success and pending responses
                    print(f"Node {node[0]} acknowledged shutdown")
                    responded_nodes.add(node[0])
                else:
                    print(f"Node {node[0]} returned unexpected status: {response.status_code}")
            except Exception as e:
                print(f"Failed to notify node {node[0]} of shutdown: {str(e)}")
        
        # Wait for remaining nodes to respond
        while len(responded_nodes) < len(nodes) and (time.time() - start_time) < max_wait_time:
            print(f"Waiting for {len(nodes) - len(responded_nodes)} nodes to respond...")
            time.sleep(2)  # Check every 2 seconds
            
            # Try to notify any nodes that haven't responded yet
            for node in nodes:
                if node[0] not in responded_nodes:
                    try:
                        response = requests.post(
                            f"{node[0]}/deregister",
                            json={'url': node[0]},
                            timeout=5
                        )
                        if response.status_code in [200, 202]:
                            print(f"Node {node[0]} acknowledged shutdown")
                            responded_nodes.add(node[0])
                    except Exception:
                        pass  # Ignore errors during retry
        
        if len(responded_nodes) < len(nodes):
            print(f"Warning: {len(nodes) - len(responded_nodes)} nodes did not respond to shutdown notice")
        else:
            print("All nodes acknowledged shutdown")
            
    finally:
        conn.close()

def clear_database():
    """Clear all entries from the database"""
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute('DELETE FROM nodes')
        conn.commit()
        print("Database cleared successfully")
    except Exception as e:
        print(f"Error clearing database: {str(e)}")
    finally:
        conn.close()

# Function to get the least loaded node
def get_least_loaded_node():
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute('SELECT url FROM nodes WHERE status = "active" ORDER BY last_heartbeat ASC LIMIT 1')
        node = c.fetchone()
        return node['url'] if node else None
    finally:
        conn.close()

def mark_node_inactive(node_url):
    """Mark a node as inactive in the database"""
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute('UPDATE nodes SET status = "inactive" WHERE url = ?', (node_url,))
        conn.commit()
        print(f"Marked node {node_url} as inactive due to timeout")
    except Exception as e:
        print(f"Error marking node inactive: {str(e)}")
    finally:
        conn.close()

# Add Response to the import
from flask import Flask, request, jsonify, Response

# Initialize ML node tracking
ml_nodes_in_use = {}
ml_node_lock = threading.Lock()

# Add chatbot tracking at module level
chatbot_nodes = []
chatbot_nodes_lock = threading.Lock()

@app.route('/api/chat', methods=['POST'])
@app.route('/chat', methods=['POST'])
def handle_chat_request():
    """Handle chat requests by forwarding them to available nodes"""
    try:
        # Get the least loaded active node
        target_node = get_least_loaded_node()

        if not target_node:
            return jsonify({'error': 'No active nodes available'}), 503

        print(f"\n=== Forwarding Chat Request ===")
        print(f"Target node: {target_node}")
        print(f"Request method: {request.method}")
        # Limit header logging for brevity if needed
        # print(f"Request headers: {dict(request.headers)}")
        print(f"Request content type: {request.content_type}")

        # Check if the request is multipart/form-data or JSON
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            # Forward multipart/form-data directly
            # Prepare headers, excluding Host and Content-Length which requests will set
            forward_headers = {k: v for k, v in request.headers if k.lower() not in ['host', 'content-length']}

            # Stream the request body to the target node
            resp = requests.post(
                f"{target_node}/chat",
                headers=forward_headers,
                data=request.get_data(), # Stream the raw data
                timeout=90,
                stream=True # Use stream=True for potentially large uploads
            )
        elif request.is_json:
             # Forward JSON data
            forward_headers = {k: v for k, v in request.headers if k.lower() not in ['host', 'content-length']}
            forward_headers['Content-Type'] = 'application/json' # Ensure correct content type

            resp = requests.post(
                f"{target_node}/chat",
                headers=forward_headers,
                json=request.get_json(),
                timeout=90,
                stream=True
            )
        else:
             # Handle other content types or lack thereof if necessary
             return jsonify({'error': 'Unsupported content type'}), 415


        print(f"Received response from node: {resp.status_code}")
        # Limit header logging for brevity if needed
        # print(f"Response headers: {dict(resp.headers)}")
        # print(f"Response content length: {resp.headers.get('Content-Length', 'N/A')} bytes")

        # Filter hop-by-hop headers before creating the response
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        headers = [(k, v) for k, v in resp.raw.headers.items() if k.lower() not in excluded_headers]

        # Create a Flask Response object, streaming the content
        response = Response(resp.iter_content(chunk_size=8192), status=resp.status_code, headers=headers)
        return response

    except requests.exceptions.Timeout:
        print(f"Request to node {target_node} timed out")
        if target_node: # Ensure target_node is not None before marking inactive
            mark_node_inactive(target_node)
        # Try to get another node
        new_node = get_least_loaded_node()
        if new_node and new_node != target_node:
            print(f"Retrying with new node: {new_node}")
            # Retry the request with the new node (recursive call)
            # Be cautious with recursion depth, consider an iterative approach for many retries
            return handle_chat_request()
        return jsonify({
            'error': 'Request to node timed out and no other nodes available'
        }), 504 # Gateway Timeout
    except requests.exceptions.ConnectionError as e:
        print(f"Failed to connect to node {target_node}: {str(e)}")
        if target_node: # Ensure target_node is not None
            mark_node_inactive(target_node)
        # Try to get another node
        new_node = get_least_loaded_node()
        if new_node and new_node != target_node:
            print(f"Retrying with new node: {new_node}")
            # Retry the request with the new node
            return handle_chat_request()
        return jsonify({
            'error': 'Failed to connect to node and no other nodes available'
        }), 502 # Bad Gateway
    except Exception as e:
        # Log the full traceback for better debugging
        app.logger.exception(f"Error forwarding request to {target_node}: {str(e)}")
        # print(f"Error forwarding request to {target_node}: {str(e)}") # Original print
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/ml/forward', methods=['POST'])
def forward_ml_request():
    """
    Forwards ML data processing requests to the models node.
    Expects JSON with: url, data_type, model.
    """
    data = request.get_json()
    url = data.get('url')
    data_type = data.get('data_type')
    model = data.get('model')

    if not url or not data_type or not model:
        return jsonify({'error': 'Missing required fields (url, data_type, model)'}), 400
    if data_type not in ['image', 'gait', 'audio', 'typing', 'health']:
        return jsonify({'error': 'Invalid data_type'}), 400

    # Get an active ML node from the database
    try:
        conn = get_db()
        c = conn.cursor()
        try:
            # Get the most recently active ML node
            c.execute('SELECT url FROM nodes WHERE type = "ml" AND status = "active" ORDER BY last_heartbeat DESC LIMIT 1')
            result = c.fetchone()
            if not result:
                return jsonify({'error': 'No active ML nodes available'}), 503
            
            ml_node_url = result[0]
            print(f"Using ML node at: {ml_node_url}")
            
            # Forward request to the ML node's process endpoint
            response = requests.post(f"{ml_node_url}/ml/process", json={
                'url': url,
                'data_type': data_type,
                'model': model
            }, timeout=20)
            return jsonify(response.json()), response.status_code
        finally:
            conn.close()
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to reach models node: {str(e)}'}), 502
    except Exception as e:
        app.logger.exception(f"Error in ML forward request: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/workflow/forward', methods=['POST'])
def forward_workflow_request():
    """
    Forwards workflow model requests (ER Triage, Lab Analysis, ECG Analysis, Diabetes Prediction).
    Expects JSON with: model, data.
    """
    data = request.get_json()
    model = data.get('model')
    workflow_data = data.get('data')

    if not model or not workflow_data:
        return jsonify({'error': 'Missing required fields (model, data)'}), 400
    
    valid_models = ['er_triage', 'lab_analysis', 'ecg_analysis', 'diabetes']
    if model not in valid_models:
        return jsonify({'error': f'Invalid model. Must be one of: {", ".join(valid_models)}'}), 400

    # Get an active ML node from the database
    try:
        conn = get_db()
        c = conn.cursor()
        try:
            # Get the most recently active ML node
            c.execute('SELECT url FROM nodes WHERE type = "ml" AND status = "active" ORDER BY last_heartbeat DESC LIMIT 1')
            result = c.fetchone()
            if not result:
                return jsonify({'error': 'No active ML nodes available'}), 503
            
            ml_node_url = result[0]
            print(f"Using ML node at: {ml_node_url}")
            
            # Forward request to the ML node's workflow endpoint
            response = requests.post(f"{ml_node_url}/workflow/process", json={
                'model': model,
                'data': workflow_data
            }, timeout=30)  # Longer timeout for workflow processing
            
            return jsonify({
                'status': 'success',
                'model': model,
                'result': response.json()
            }), response.status_code
        finally:
            conn.close()
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to reach models node: {str(e)}'}), 502
    except Exception as e:
        app.logger.exception(f"Error in workflow forward request: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

def get_active_chatbot_node():
    """Get an active chatbot node"""
    with chatbot_nodes_lock:
        if not chatbot_nodes:
            return None
        # For now just return the first one
        # In the future, you could implement load balancing for multiple chatbots
        return chatbot_nodes[0]

def get_active_llm_node():
    """Get an active LLM node (main.py instance)"""
    conn = get_db()
    c = conn.cursor()
    try:
        # First try to find nodes with type = 'llm'
        c.execute('''
            SELECT url FROM nodes 
            WHERE status = "active" 
            AND type = "llm"
            ORDER BY last_heartbeat DESC
        ''')
        node = c.fetchone()
        
        # If no nodes with explicit type, try to find by port (5050 is the default for main.py)
        if not node:
            c.execute('''
                SELECT url FROM nodes 
                WHERE status = "active" 
                AND url LIKE "%5050%" 
                ORDER BY last_heartbeat DESC
            ''')
            node = c.fetchone()
            
        # If still no nodes, return None
        if not node:
            print("No active LLM nodes found")
            return None
            
        print(f"Found active LLM node: {node[0]}")
        return node[0]
    except Exception as e:
        print(f"Error getting active LLM node: {str(e)}")
        return None
    finally:
        conn.close()

def get_available_ml_node():
    """Get an available ML node that's not currently in use"""
    conn = get_db()
    c = conn.cursor()
    try:
        # Get all active ML nodes
        c.execute('''
            SELECT url FROM nodes 
            WHERE status = "active" 
            AND (type = "ml" OR url LIKE "%9000%")
            ORDER BY last_heartbeat DESC
        ''')
        nodes = c.fetchall()
        
        if not nodes:
            return None
            
        # Find one that's not in use
        with ml_node_lock:
            for row in nodes:
                node_url = row[0]
                if node_url not in ml_nodes_in_use:
                    # Mark this node as in use
                    ml_nodes_in_use[node_url] = time.time()
                    # Schedule cleanup after 30 seconds
                    cleanup_thread = threading.Thread(
                        target=lambda url=node_url: (time.sleep(30), ml_nodes_in_use.pop(url, None)),
                        daemon=True
                    )
                    cleanup_thread.start()
                    return node_url
                    
            # If all nodes are in use, return the least recently used one
            if nodes:
                return nodes[0][0]
            return None
    except Exception as e:
        print(f"Error getting available ML node: {str(e)}")
        return None
    finally:
        conn.close()

@app.route('/api/external/chat', methods=['POST'])
def handle_external_chat():
    """Handle external chat requests by forwarding them to chatbot"""
    try:
        # Get an active chatbot node
        chatbot_node = get_active_chatbot_node()
        if not chatbot_node:
            return jsonify({'error': 'No active chatbot available'}), 503
            
        # Get an active ML node
        ml_node = get_available_ml_node()
        if not ml_node:
            ml_node = "No active ML node available"
            
        # Get an active LLM node    
        llm_node = get_active_llm_node()
        if not llm_node:
            llm_node = "No active LLM node available"
            
        print(f"\n=== Forwarding External Chat Request ===")
        print(f"Chatbot Node: {chatbot_node}")
        print(f"ML Node: {ml_node}")
        print(f"LLM Node: {llm_node}")
        
        # Add ML and LLM URLs to headers for the chatbot
        headers = {
            'X-ML-Node-URL': ml_node if ml_node else '',
            'X-LLM-Node-URL': llm_node if llm_node else ''
        }
        
        # Forward the request content and headers
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            # For multipart form data, we need to forward files
            files = {}
            data = {}
            
            # Process form fields
            for key, value in request.form.items():
                data[key] = value
                
            # Process file uploads
            for key, file in request.files.items():
                # Save to a temporary file
                temp = tempfile.NamedTemporaryFile(delete=False)
                file.save(temp.name)
                temp.close()
                
                # Re-open the file for reading
                files[key] = (file.filename, open(temp.name, 'rb'), file.content_type)
            
            # Make the request
            response = requests.post(
                f"{chatbot_node}/chat",
                data=data,
                files=files,
                headers=headers,
                timeout=90
            )
            
            # Clean up temporary files
            for key, file_tuple in files.items():
                file_tuple[1].close()  # Close file handle
                os.unlink(file_tuple[1].name)  # Delete temp file
                
        else:
            # For JSON and other content types
            response = requests.post(
                f"{chatbot_node}/chat",
                headers={**headers, 'Content-Type': request.content_type},
                data=request.get_data(),
                timeout=90
            )
        
        return Response(
            response.content,
            status=response.status_code,
            content_type=response.headers.get('Content-Type', 'application/json')
        )
        
    except requests.Timeout:
        return jsonify({'error': 'Request to chatbot timed out'}), 504
    except requests.RequestException as e:
        return jsonify({'error': f'Error forwarding request to chatbot: {str(e)}'}), 502
    except Exception as e:
        print(f"Error in handle_external_chat: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Set up signal handlers in the main thread
        import signal
        def handle_shutdown(signum, frame):
            print(f"\nReceived signal {signum}. Starting shutdown...")
            # Cleanup is registered with atexit, so it should run automatically
            # cleanup() # Optional: call explicitly if atexit doesn't cover all cases
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        # Initialize the application components
        print("\n=== Initializing Node Handler ===")
        if not initialize_app():
            print("✗ Failed to initialize the application")
            sys.exit(1)

        # Start Waitress server in a separate thread
        print("\n=== Starting Waitress Server ===")
        wsgi_thread = threading.Thread(target=lambda: serve(app, host='0.0.0.0', port=8000))
        wsgi_thread.daemon = True # Daemonize thread to exit when main thread exits
        wsgi_thread.start()

        # Wait for Waitress server to be ready
        if not wait_for_server_ready():
             raise Exception("Waitress server failed to start or become ready")

        # Start ngrok and get the URL
        print("\n=== Setting up ngrok tunnel ===")
        ngrok_url = run_ngrok() # run_ngrok already contains checks for server readiness
        if ngrok_url:
            print(f"✓ Node handler is ready at: {ngrok_url}")
            print("You can now start your nodes.")
            print("Waiting for node connections...")

            # Keep the main thread alive (Waitress runs in a daemon thread)
            while True:
                time.sleep(60) # Keep main thread alive, check less frequently
        else:
            print("✗ Node handler failed to start properly (ngrok setup failed)")
            # No need to call cleanup() here, atexit should handle it
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Shutting down...")
        # No need to call cleanup() here, atexit should handle it
        sys.exit(0)
    except Exception as e:
        print(f"✗ Failed to start node handler: {str(e)}")
        import traceback
        print(traceback.format_exc()) # Print full traceback for debugging
        # No need to call cleanup() here, atexit should handle it
        sys.exit(1)