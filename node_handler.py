from flask import Flask, request, jsonify
import ngrok
import sqlite3
import threading
import time
from datetime import datetime
import requests
import sys
import atexit
from waitress import serve # Import waitress

app = Flask(__name__)

# Initialize SQLite database
def init_db():
    """Initialize database with reconnection logic"""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect('nodes.db', timeout=20)  # Add timeout
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS nodes
                         (url TEXT PRIMARY KEY, 
                          status TEXT, 
                          last_heartbeat TIMESTAMP)''')
            conn.commit()
            conn.close()
            print("Database initialized successfully")
            return
        except sqlite3.Error as e:
            print(f"Database initialization attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
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
            return jsonify({
                'status': 'ok',
                'message': 'Node handler is running',
                'active_nodes': active_nodes
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
    """Register a new node with the node handler"""
    try:
        data = request.json
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required'}), 400
            
        node_url = data['url']
        print(f"\n=== New Node Registration ===")
        print(f"Registering node: {node_url}")
        
        conn = get_db()
        try:
            c = conn.cursor()
            # Check if node already exists
            c.execute('SELECT url FROM nodes WHERE url = ?', (node_url,))
            existing_node = c.fetchone()
            
            if existing_node:
                print(f"Node {node_url} already registered. Updating status...")
                c.execute('UPDATE nodes SET status = "active", last_heartbeat = CURRENT_TIMESTAMP WHERE url = ?', (node_url,))
            else:
                print(f"Registering new node: {node_url}")
                c.execute('INSERT INTO nodes (url, status, last_heartbeat) VALUES (?, "active", CURRENT_TIMESTAMP)', (node_url,))
            
            conn.commit()
            print(f"✓ Node {node_url} registered successfully")
            return jsonify({'status': 'success', 'message': 'Node registered successfully'})
        except Exception as e:
            print(f"✗ Error registering node: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            conn.close()
    except Exception as e:
        print(f"✗ Error in register_node: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/deregister', methods=['POST'])
def deregister_node():
    """Deregister a node from the node handler"""
    try:
        data = request.json
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required'}), 400
            
        node_url = data['url']
        print(f"\n=== Node Deregistration ===")
        print(f"Deregistering node: {node_url}")
        
        conn = get_db()
        try:
            c = conn.cursor()
            c.execute('DELETE FROM nodes WHERE url = ?', (node_url,))
            conn.commit()
            print(f"✓ Node {node_url} deregistered successfully")
            return jsonify({'status': 'success', 'message': 'Node deregistered successfully'})
        except Exception as e:
            print(f"✗ Error deregistering node: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            conn.close()
    except Exception as e:
        print(f"✗ Error in deregister_node: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/heartbeat', methods=['POST'])
def update_heartbeat():
    """Update the heartbeat timestamp for a node"""
    try:
        data = request.json
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
        ngrok.set_auth_token("2uHfQak8CxEWThSHdtiVOIC14kq_5px1cRz1ammGDZYnboB2g")
        time.sleep(2)
        
        try:
            # Start ngrok with a fixed domain
            print("Starting ngrok tunnel...")
            listener = ngrok.forward(
                8000,
                domain="pup-improved-labrador.ngrok-free.app"
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
                    timeout=10  # Increased timeout to 10 seconds
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
        print(f"Request headers: {dict(request.headers)}")
        print(f"Request content type: {request.content_type}")
        
        # Parse the incoming request
        question = request.form.get('question')
        image_url = request.form.get('image')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        print(f"Question: {question}")
        print(f"Image URL: {image_url}")
        
        # Prepare the files dictionary for multipart/form-data
        files = {
            'question': (None, question),
            'image': (None, image_url) if image_url else None
        }
        
        # Forward the request in multipart/form-data format
        response = requests.post(
            f"{target_node}/chat",
            files=files,
            timeout=90
        )
        
        print(f"Received response from node: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response content length: {len(response.content)} bytes")
        
        # Return the exact response
        return response.content, response.status_code, response.headers
        
    except requests.exceptions.Timeout:
        print(f"Request to node {target_node} timed out")
        # Mark the node as inactive
        mark_node_inactive(target_node)
        # Try to get another node
        new_node = get_least_loaded_node()
        if new_node and new_node != target_node:
            print(f"Retrying with new node: {new_node}")
            # Retry the request with the new node
            return handle_chat_request()
        return jsonify({
            'error': 'Request to node timed out and no other nodes available'
        }), 504
    except requests.exceptions.ConnectionError as e:
        print(f"Failed to connect to node {target_node}: {str(e)}")
        # Mark the node as inactive
        mark_node_inactive(target_node)
        # Try to get another node
        new_node = get_least_loaded_node()
        if new_node and new_node != target_node:
            print(f"Retrying with new node: {new_node}")
            # Retry the request with the new node
            return handle_chat_request()
        return jsonify({
            'error': 'Failed to connect to node and no other nodes available'
        }), 502
    except Exception as e:
        print(f"Error forwarding request to {target_node}: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': f'Failed to forward request to node: {str(e)}'
        }), 500

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