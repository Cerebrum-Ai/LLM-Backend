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

def test_typing_analysis(url="http://localhost:5050", condition=None, num_keystrokes=50):
    """
    Test the typing analysis endpoint with generated data.
    
    Args:
        url (str): Base URL of the LLM service
        condition (str): Optional condition to simulate
        num_keystrokes (int): Number of keystrokes to generate
    """
    endpoint = f"{url}/api/analyze_typing"
    print(f"\n🔍 Testing Typing Analysis Endpoint: {endpoint}")
    print(f"Condition: {condition or 'normal'}")
    print(f"Generating {num_keystrokes} keystrokes...")
    
    # Generate data
    typing_data = generate_typing_data(num_keystrokes, condition)
    
    # Send request
    try:
        print("Sending request...")
        response = requests.post(endpoint, json=typing_data, timeout=15)
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Success! Analysis results:")
            print("-" * 50)
            
            if "analysis" in result:
                analysis = result["analysis"]
                
                if "detected_condition" in analysis:
                    print(f"Detected condition: {analysis['detected_condition']}")
                
                if "probabilities" in analysis:
                    print("\nProbabilities:")
                    for cond, prob in sorted(analysis["probabilities"].items(), key=lambda x: x[1], reverse=True):
                        print(f"  {cond}: {prob:.2f}")
                
                if "features" in analysis:
                    print("\nKey Features:")
                    for feature, value in analysis["features"].items():
                        print(f"  {feature}: {value:.2f}")
            else:
                print(json.dumps(result, indent=2))
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
    test_typing_analysis(args.url, args.condition, args.keystrokes)

if __name__ == "__main__":
    main() 