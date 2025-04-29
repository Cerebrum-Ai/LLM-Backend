#!/usr/bin/env python3
"""
Test script for the LLM-Backend endpoints.

This script tests all the main endpoints of both the LLM service and ML Models service.
It sends requests to each endpoint and prints the responses.
"""

import requests
import base64
import json
import time
import sys
import os
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default ngrok URLs - update these with your actual ngrok URLs
# All LLM and typing analysis requests must go through the node_handler (not directly to chatbot or LLM)
NODE_HANDLER_URL = os.environ.get("NODE_HANDLER_URL", "http://localhost:8000")

print(f"Testing with URL:\n- Node Handler: {NODE_HANDLER_URL}")

def test_llm_service_health():
    """Test the health of the LLM (node handler) service via /status endpoint"""
    url = f"{NODE_HANDLER_URL}/status"
    print(f"\n\033[1m🔍 Testing LLM Service Health: {url}\033[0m")
    try:
        response = requests.get(url, timeout=10)
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_ml_models_health():
    """Test the health of the ML Models service via node handler"""
    url = f"{NODE_HANDLER_URL}/ml/forward"
    print(f"\n\033[1m🔍 Testing ML Models Service Health via Node Handler: {url}\033[0m")
    # We'll send a minimal valid payload (simulate typing analysis health check)
    payload = {
        "url": {
            "keystrokes": [],
            "text": "health check"
        },
        "data_type": "typing",
        "model": "pattern"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status code: {response.status_code}")
        try:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        except Exception:
            print("Could not parse response as JSON")
        return response.status_code in [200, 400]
    except Exception as e:
        print(f"Error connecting to ML service via node handler: {str(e)}")
        return False

def test_chat_endpoint():
    """Test the chat endpoint via node handler routing (text only)"""
    url = f"{NODE_HANDLER_URL}/chat"
    print(f"\n\033[1m🔍 Testing Chat Endpoint via Node Handler: {url}\033[0m")
    payload = {
        "question": "What are the symptoms of diabetes?",
        "session_id": f"test_{int(time.time())}"
    }
    try:
        print("Sending request to chat endpoint (this may take up to 30 seconds)...")
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status code: {response.status_code}")
        try:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        except Exception:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_chat_image():
    """Test the chat endpoint with an image URL"""
    url = f"{NODE_HANDLER_URL}/api/external/chat"
    print(f"\n\033[1m🖼️ Testing Chat Endpoint with Image: {url}\033[0m")
    # Use uploaded file URL if available
    image_url = globals().get("IMAGE_FILE_URL") or "https://dermnetnz.org/assets/Uploads/site-age-specific/lowerleg9.jpg"
    payload = {
        "question": "What are the symptoms of diabetes?",
        "image_url": image_url,
        "session_id": f"test_image_{int(time.time())}"
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status code: {response.status_code}")
        try:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        except Exception:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_chat_audio():
    """Test the chat endpoint with an audio URL"""
    url = f"{NODE_HANDLER_URL}/api/external/chat"
    print(f"\n\033[1m🔊 Testing Chat Endpoint with Audio: {url}\033[0m")
    # Use uploaded file URL if available
    audio_url = globals().get("AUDIO_FILE_URL") or "https://odgfmdbnjroqktddkgkz.supabase.co/storage/v1/object/sign/test/output10.wav?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InN0b3JhZ2UtdXJsLXNpZ25pbmcta2V5XzRmYjNjMTAyLTkzMGItNDdmMi1hMTllLTA2MDI5MmQxMjExNiJ9.eyJ1cmwiOiJ0ZXN0L291dHB1dDEwLndhdiIsImlhdCI6MTc0NTg4NDA4OSwiZXhwIjoxNzc3NDIwMDg5fQ.xFzuZ9pWI8y69SXywKxgM_JFkxV8CWWXxgc5VMqeT30"
    payload = {
        "question": "What are the symptoms of diabetes?",
        "audio_url": audio_url,
        "session_id": f"test_audio_{int(time.time())}"
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status code: {response.status_code}")
        try:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        except Exception:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.Timeout:
        print("Request timed out after 30 seconds. This is normal for the first request as the LLM model loads.")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False



def test_ml_process_typing():
    """Test the ML process endpoint for typing data via node handler's /ml/forward"""
    url = f"{NODE_HANDLER_URL}/ml/forward"
    print(f"\n\033[1m🔍 Testing ML Process Endpoint (Typing) via Node Handler: {url}\033[0m")
    # Create typing data with enough keystrokes (at least 10)
    keystrokes = []
    for i in range(20):
        keystrokes.append({
            "key": chr(97 + (i % 26)),  # a-z characters
            "timeDown": 1680000000 + (i * 100),
            "timeUp": 1680000000 + (i * 100) + 50,
            "pressure": 1.0
        })
    payload = {
        "url": {
            "keystrokes": keystrokes,
            "text": "test typing"
        },
        "data_type": "typing",
        "model": "pattern"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status code: {response.status_code}")
        if response.status_code in [200, 400]:
            try:
                print(f"Response: {json.dumps(response.json(), indent=2)}")
                return True
            except Exception:
                print("Could not parse response as JSON")
                return False
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def run_all_tests():
    """Run all endpoint tests"""
    print("\033[1m🧪 Running LLM-Backend Endpoint Tests\033[0m")
    
    # Track test results
    results = {
        "llm_health": test_llm_service_health(),
        "ml_health": test_ml_models_health(),
        "chat": test_chat_endpoint(),
        "chat_image": test_chat_image(),
        "chat_audio": test_chat_audio(),
        "ml_process_typing": test_ml_process_typing()
    }
    
    # Print summary
    print("\n\033[1m📊 Test Results Summary:\033[0m")
    for test, passed in results.items():
        status = "\033[32m✅ PASSED\033[0m" if passed else "\033[31m❌ FAILED\033[0m"
        print(f"{test.ljust(20)}: {status}")
    
    # Calculate overall result
    all_passed = all(results.values())
    overall = "\033[32m✅ ALL TESTS PASSED\033[0m" if all_passed else "\033[31m❌ SOME TESTS FAILED\033[0m"
    print(f"\n\033[1m🏁 Overall Result: {overall}\033[0m")
    
    return all_passed

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test LLM-Backend endpoints')
    parser.add_argument('llm_url', nargs='?', help='URL of the Node Handler service')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()
    if args.llm_url:
        NODE_HANDLER_URL = args.llm_url
    print(f"Testing with URL:\n- Node Handler: {NODE_HANDLER_URL}")
    success = run_all_tests()
    sys.exit(0 if success else 1)