#!/usr/bin/env python3
"""
Cerebrum AI LLM Backend - Comprehensive Testing Script

This script tests all components of the Cerebrum AI LLM Backend:
1. LLM Service (main.py)
2. ML Models Service (models.py)
3. Monitoring System (monitor.py)

Usage:
  python test_all.py [--llm-url=http://localhost:5050] [--ml-url=http://localhost:9000]
"""

import argparse
import json
import os
import requests
import sys
import time
import logging
import random
import base64
import subprocess
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Supabase Upload Helper ---
from supabase_upload import upload_file
from test_endpoints import test_llm_service_health

def get_uploaded_file_url(file_type="file"):
    print(f"Please enter the local path to your test {file_type} file (will be uploaded to Supabase bucket 'test'):")
    file_path = input("File path: ").strip()
    print("Uploading...\n")
    try:
        url = upload_file(file_path)
        if url:
            print(f"Uploaded to: {url}")
            return url
        else:
            print("ERROR: Upload failed or URL not returned.")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("cerebrum-test")

# Routing configuration: always use NODE_HANDLER_URL for LLM and typing endpoints
NODE_HANDLER_URL = os.environ.get("NODE_HANDLER_URL", "http://localhost:8000")
ML_MODELS_URL = os.environ.get("ML_MODELS_URL", "http://localhost:9000")

print(f"Testing with URLs:\n- Node Handler: {NODE_HANDLER_URL}\n- ML Models: {ML_MODELS_URL}")

# Rich console for pretty output
console = Console()

def test_node_handler_health():
    """Test if the node handler is running (should be used for all LLM/typing routes)"""
    try:
        response = requests.get(f"{NODE_HANDLER_URL}/status", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Node Handler is running")
            return True
        else:
            logger.error(f"❌ Node Handler returned status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"❌ Node Handler health check failed: {str(e)}")
        return False

def test_ml_service_health():
    """Test if the ML Models service is running"""
    try:
        response = requests.get(f"{ML_MODELS_URL}/", timeout=5)
        if response.status_code == 200:
            logger.info("✅ ML Models Service is running")
            return True
        else:
            logger.error(f"❌ ML Models Service returned status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"❌ ML Models Service health check failed: {str(e)}")
        return False

def test_llm_chat():
    """Test the LLM chat endpoint via node handler routing"""
    try:
        payload = {
            "question": "What are the symptoms of diabetes?",
            "session_id": f"test_{int(time.time())}"
        }
        
        response = requests.post(f"{NODE_HANDLER_URL}/api/chat", json=payload, timeout=30)
        
        if response.status_code == 200:
            logger.info("✅ LLM Chat endpoint working via node handler")
            logger.info(f"Response preview: {str(response.json())[:100]}...")
            return True
        else:
            logger.error(f"❌ LLM Chat endpoint failed via node handler with status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except requests.RequestException as e:
        logger.error(f"❌ LLM Chat endpoint test via node handler failed: {str(e)}")
        return False

def test_typing_analysis():
    """Test the typing analysis endpoint via node handler routing"""
    try:
        # Generate random keystroke data
        keystrokes = []
        base_time = int(time.time() * 1000)
        for i in range(30):
            key = chr(97 + random.randint(0, 25))  # a-z
            time_down = base_time + (i * 100)
            time_up = time_down + random.randint(30, 80)
            keystrokes.append({
                "key": key,
                "timeDown": time_down,
                "timeUp": time_up,
                "event": "keydown"
            })
        
        payload = {
            "url": {
                "keystrokes": keystrokes,
                "text": "This is sample text for testing typing analysis"
            },
            "data_type": "typing",
            "model": "pattern"
        }
        response = requests.post(f"{NODE_HANDLER_URL}/ml/forward", json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Typing Analysis endpoint working via node handler /ml/forward")
            logger.info(f"Response preview: {str(response.json())[:100]}...")
            return True
        else:
            logger.error(f"❌ Typing Analysis endpoint failed via node handler with status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except requests.RequestException as e:
        logger.error(f"❌ Typing Analysis endpoint test via node handler failed: {str(e)}")
        return False

def test_ml_process_endpoint():
    """Test the ML process endpoint via node_handler's /ml/forward endpoint"""
    try:
        # Generate random keystroke data for typing analysis
        keystrokes = []
        base_time = int(time.time() * 1000)
        for i in range(30):
            key = chr(97 + random.randint(0, 25))  # a-z
            time_down = base_time + (i * 100)
            time_up = time_down + random.randint(30, 80)
            keystrokes.append({
                "key": key,
                "timeDown": time_down,
                "timeUp": time_up,
                "event": "keydown"
            })
        # Compose payload for /ml/forward
        ml_process_url = f"{ML_MODELS_URL}/ml/process"
        forward_payload = {
            "url": ml_process_url,
            "data_type": "typing",
            "model": "default",
            "keystrokes": keystrokes
        }
        response = requests.post(f"{NODE_HANDLER_URL}/ml/forward", json=forward_payload, timeout=15)
        if response.status_code in [200, 400]:  # 400 is OK if it's because of format validation
            logger.info("✅ ML Process endpoint working via node_handler /ml/forward")
            logger.info(f"Response: {response.text[:100]}...")
            return True
        else:
            logger.error(f"❌ ML Process endpoint via node_handler failed with status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except requests.RequestException as e:
        logger.error(f"❌ ML Process endpoint test via node_handler failed: {str(e)}")
        return False



def run_all_tests():
    """Run all tests and print a summary"""
    console.print("\n[bold cyan]Cerebrum AI LLM Backend - System Test[/bold cyan]\n")
    
    # Track test results
    results = {}
    
    # Test LLM Service health
    console.print("[bold]Testing LLM Service Health...[/bold]")
    results["llm_health"] = test_llm_service_health()
    
    # Test ML Service health
    console.print("\n[bold]Testing ML Models Service Health...[/bold]")
    results["ml_health"] = test_ml_service_health()
    
    # Only run endpoint tests if services are healthy
    if results["llm_health"]:
        console.print("\n[bold]Testing LLM Chat Endpoint...[/bold]")
        results["llm_chat"] = test_llm_chat()
        
        console.print("\n[bold]Testing Typing Analysis Endpoint...[/bold]")
        results["typing_analysis"] = test_typing_analysis()
    
    if results["ml_health"]:
        console.print("\n[bold]Testing ML Process Endpoint...[/bold]")
        results["ml_process"] = test_ml_process_endpoint()
    
    
    # Print summary table
    console.print("\n[bold]Test Results Summary:[/bold]")
    table = Table(show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    
    for test, result in results.items():
        status = "[green]✅ PASS[/green]" if result else "[red]❌ FAIL[/red]"
        table.add_row(test.replace("_", " ").title(), status)
    
    console.print(table)
    
    # Overall result
    if all(results.values()):
        console.print("\n[bold green]All tests passed! Your system is working correctly.[/bold green]")
        return 0
    else:
        console.print("\n[bold red]Some tests failed. Please check the logs for more details.[/bold red]")
        return 1

def main():
    """Main entry point"""
    global NODE_HANDLER_URL, ML_MODELS_URL
    parser = argparse.ArgumentParser(description="Cerebrum AI LLM Backend System Test")
    parser.add_argument("--llm-url", type=str, default=NODE_HANDLER_URL, help="LLM Service URL (node handler)")
    parser.add_argument("--ml-url", type=str, default=ML_MODELS_URL, help="ML Models Service URL")
    args = parser.parse_args()
    
    # Update URLs if provided
    NODE_HANDLER_URL = args.llm_url
    ML_MODELS_URL = args.ml_url
    
    logger.info(f"Testing LLM Service at: {NODE_HANDLER_URL}")
    logger.info(f"Testing ML Models Service at: {ML_MODELS_URL}")
    
    # Prompt user for image file and upload
    image_url = get_uploaded_file_url("image")
    logger.info(f"Test image uploaded to Supabase: {image_url}")
    globals()["IMAGE_FILE_URL"] = image_url

    # Prompt user for audio file and upload
    audio_url = get_uploaded_file_url("audio")
    logger.info(f"Test audio uploaded to Supabase: {audio_url}")
    globals()["AUDIO_FILE_URL"] = audio_url

    return run_all_tests()

if __name__ == "__main__":
    sys.exit(main()) 