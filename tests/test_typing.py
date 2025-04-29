#!/usr/bin/env python3
"""
Test script for Cerebrum AI typing/keystroke analysis functionality.
This script generates synthetic typing data and sends it to the typing analysis endpoint.
"""

import requests
import json
import random
import time
import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def manual_typing_entry():
    """
    Use TypingDataCollector from typing_analyzer.py to capture real keystroke data, send it to the node_handler /api/analyze_typing endpoint, and print the result.
    """
    from tests.typing_analyzer import TypingDataCollector
    import time
    url = os.environ.get("NODE_HANDLER_URL", "http://localhost:8000")
    if not url.endswith("/api/analyze_typing"):
        url = f"{url.rstrip('/')}/api/analyze_typing"

    print("\nA typing window will open. Please type your sentence and press ESC when done.")
    collector = TypingDataCollector()
    collector.start_recording()
    print("Start typing...")
    # Wait for user to finish typing (ESC will stop recording)
    while collector.is_recording:
        time.sleep(0.1)
    result = collector.stop_recording()
    keystrokes = result['keystrokes']
    user_text = ''.join([k['key'] for k in keystrokes if k['key'] not in ['Key.esc', 'Key.enter', 'Key.shift', 'Key.ctrl', 'Key.alt', 'Key.cmd']])
    ml_url = os.environ.get("NODE_HANDLER_URL", "http://localhost:8000")
    if not ml_url.endswith("/ml/forward"):
        ml_url = f"{ml_url.rstrip('/')}/ml/forward"
    payload = {
        "url": {
            "keystrokes": keystrokes,
            "text": user_text
        },
        "data_type": "typing",
        "model": "pattern"
    }
    try:
        print(f"\nSending your typing data to: {ml_url}")
        response = requests.post(ml_url, json=payload, timeout=15)
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("\n\u2705 Analysis results:")
            print(json.dumps(result, indent=2))
        else:
            print(f"\n\u274C Error: {response.text}")
    except requests.RequestException as e:
        print(f"\n\u274C Request failed: {str(e)}")

def generate_typing_data(num_keystrokes=50, condition=None):
    """
    Generate synthetic typing data for testing.
    
    Args:
        num_keystrokes (int): Number of keystrokes to generate
        condition (str): Optional simulation of a specific condition
                         ('normal', 'parkinsons', 'essential_tremor', etc.)
    
    Returns:
        dict: Keystroke data in the format expected by the API
    """
    keystrokes = []
    base_time = int(time.time() * 1000)
    
    # Set parameters based on condition
    if condition == "parkinsons":
        press_duration_range = (80, 200)  # longer key presses
        interval_range = (200, 500)  # longer intervals
        pressure_range = (0.5, 0.9)
        backspace_probability = 0.15
    elif condition == "essential_tremor":
        press_duration_range = (60, 150)
        interval_range = (150, 400)
        pressure_range = (0.6, 1.0)
        backspace_probability = 0.1
    elif condition == "carpal_tunnel":
        press_duration_range = (100, 180)
        interval_range = (250, 450)
        pressure_range = (0.3, 0.7)
        backspace_probability = 0.08
    else:  # normal typing
        press_duration_range = (50, 120)
        interval_range = (100, 300)
        pressure_range = (0.6, 0.9)
        backspace_probability = 0.05
    
    # Generate keystrokes
    current_time = base_time
    for i in range(num_keystrokes):
        # Decide if this is a backspace
        is_backspace = random.random() < backspace_probability
        
        # Generate key
        if is_backspace:
            key = "Backspace"
        else:
            key = chr(97 + random.randint(0, 25))  # a-z
        
        # Generate timing and pressure
        press_duration = random.randint(*press_duration_range)
        time_down = current_time
        time_up = time_down + press_duration
        pressure = random.uniform(*pressure_range)
        
        # Add keystroke
        keystrokes.append({
            "key": key,
            "timeDown": time_down,
            "timeUp": time_up,
            "pressure": pressure
        })
        
        # Move to next keystroke
        interval = random.randint(*interval_range)
        current_time += interval
    
    # Create final payload
    return {
        "keystrokes": keystrokes,
        "text": "This is sample text for testing typing analysis"
    }

def test_typing_analysis(url=None, condition=None, num_keystrokes=50):
    # Always use NODE_HANDLER_URL + /ml/forward for typing analysis
    ml_url = os.environ.get("NODE_HANDLER_URL", "http://localhost:8000")
    if not ml_url.endswith("/ml/forward"):
        ml_url = f"{ml_url.rstrip('/')}/ml/forward"

    """
    Test the typing analysis endpoint with generated data.
    
    Args:
        url (str): Base URL of the LLM service (ignored, for compatibility)
        condition (str): Optional condition to simulate
        num_keystrokes (int): Number of keystrokes to generate
    """
    print(f"\n\U0001F50D Testing Typing Analysis Endpoint: {ml_url}")
    print(f"Condition: {condition or 'normal'}")
    print(f"Generating {num_keystrokes} keystrokes...")
    typing_data = generate_typing_data(num_keystrokes, condition)
    payload = {
        "url": typing_data,
        "data_type": "typing",
        "model": "pattern"
    }
    print("Sending request...")
    try:
        response = requests.post(ml_url, json=payload, timeout=15)
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("\n\u2705 Analysis results:")
            print(json.dumps(result, indent=2))
            if "probabilities" in result:
                print("\nProbabilities:")
                for cond, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {cond}: {prob:.2f}")

            if "features" in result:
                print("\nKey Features:")
                for feature, value in result["features"].items():
                    print(f"  {feature}: {value:.2f}")
        else:
            print(f"\n❌ Error: {response.text}")
    
    except requests.RequestException as e:
        print(f"\n❌ Request failed: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Test Cerebrum AI typing analysis")
    parser.add_argument("--url", default="http://localhost:5050", help="LLM service URL")
    parser.add_argument("--condition", choices=["normal", "parkinsons", "essential_tremor", "carpal_tunnel", "multiple_sclerosis"], 
                        help="Simulate specific typing condition")
    parser.add_argument("--keystrokes", type=int, default=50, help="Number of keystrokes to generate")
    args = parser.parse_args()

    # Step 1: Always run synthetic typing test first
    test_typing_analysis(args.url, args.condition, args.keystrokes)

    # Step 2: Prompt for manual typing
    while True:
        user_input = input("\nWould you like to try manual typing entry for live analysis? (y/n): ").strip().lower()
        if user_input == 'y':
            manual_typing_entry()
            break
        elif user_input == 'n':
            print("Exiting.")
            break
        else:
            print("Please enter 'y' or 'n'.")

if __name__ == "__main__":
    main()