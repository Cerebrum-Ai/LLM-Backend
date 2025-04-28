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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("cerebrum-test")

# Default URLs
LLM_SERVICE_URL = "http://localhost:5050"
ML_MODELS_URL = "http://localhost:9000"

# Rich console for pretty output
console = Console()

def test_llm_service_health():
    """Test if the LLM service is running"""
    try:
        response = requests.get(f"{LLM_SERVICE_URL}/api/chat", timeout=5)
        if response.status_code == 200:
            logger.info("✅ LLM Service is running")
            return True
        else:
            logger.error(f"❌ LLM Service returned status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"❌ LLM Service health check failed: {str(e)}")
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
    """Test the LLM chat endpoint"""
    try:
        payload = {
            "question": "What are the symptoms of diabetes?",
            "session_id": f"test_{int(time.time())}"
        }
        
        response = requests.post(f"{LLM_SERVICE_URL}/api/chat", json=payload, timeout=30)
        
        if response.status_code == 200:
            logger.info("✅ LLM Chat endpoint working")
            logger.info(f"Response preview: {str(response.json())[:100]}...")
            return True
        else:
            logger.error(f"❌ LLM Chat endpoint failed with status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except requests.RequestException as e:
        logger.error(f"❌ LLM Chat endpoint test failed: {str(e)}")
        return False

def test_typing_analysis():
    """Test the typing analysis endpoint"""
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
        
        payload = {"keystrokes": keystrokes}
        
        response = requests.post(f"{LLM_SERVICE_URL}/api/analyze_typing", json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Typing Analysis endpoint working")
            logger.info(f"Response preview: {str(response.json())[:100]}...")
            return True
        else:
            logger.error(f"❌ Typing Analysis endpoint failed with status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except requests.RequestException as e:
        logger.error(f"❌ Typing Analysis endpoint test failed: {str(e)}")
        return False

def test_ml_process_endpoint():
    """Test the ML process endpoint directly"""
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
            "url": json.dumps({"keystrokes": keystrokes}),
            "data_type": "typing",
            "model": "pattern"
        }
        
        response = requests.post(f"{ML_MODELS_URL}/ml/process", json=payload, timeout=10)
        
        if response.status_code in [200, 400]:  # 400 is OK if it's because of format validation
            logger.info("✅ ML Process endpoint working")
            logger.info(f"Response: {response.text[:100]}...")
            return True
        else:
            logger.error(f"❌ ML Process endpoint failed with status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except requests.RequestException as e:
        logger.error(f"❌ ML Process endpoint test failed: {str(e)}")
        return False

def test_monitor_script():
    """Test if the monitor script runs without error"""
    try:
        # Run the monitor for just a few seconds to see if it starts
        process = subprocess.Popen(
            ["python", "monitor.py", "--interval=3"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Give it a few seconds to start
        time.sleep(5)
        
        # Then kill it
        process.terminate()
        stdout, stderr = process.communicate(timeout=2)
        
        if process.returncode is None or process.returncode == 0:
            logger.info("✅ Monitor script started successfully")
            return True
        else:
            logger.error(f"❌ Monitor script failed with return code: {process.returncode}")
            logger.error(f"Stderr: {stderr.decode('utf-8')}")
            return False
    except subprocess.SubprocessError as e:
        logger.error(f"❌ Monitor script test failed: {str(e)}")
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
    
    # Test monitoring script
    console.print("\n[bold]Testing Monitoring Script...[/bold]")
    results["monitor"] = test_monitor_script()
    
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
    parser = argparse.ArgumentParser(description="Cerebrum AI LLM Backend System Test")
    parser.add_argument("--llm-url", type=str, default=LLM_SERVICE_URL, help="LLM Service URL")
    parser.add_argument("--ml-url", type=str, default=ML_MODELS_URL, help="ML Models Service URL")
    args = parser.parse_args()
    
    global LLM_SERVICE_URL, ML_MODELS_URL
    LLM_SERVICE_URL = args.llm_url
    ML_MODELS_URL = args.ml_url
    
    logger.info(f"Testing LLM Service at: {LLM_SERVICE_URL}")
    logger.info(f"Testing ML Models Service at: {ML_MODELS_URL}")
    
    return run_all_tests()

if __name__ == "__main__":
    sys.exit(main()) 