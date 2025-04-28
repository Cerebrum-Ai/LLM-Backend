import json
import os
import sys
from typing_processor import KeystrokeProcessor

def generate_test_data(condition="normal"):
    """Generate simulated keystroke data for testing"""
    import numpy as np
    import time
    
    now = int(time.time() * 1000)  # Current time in milliseconds
    keystrokes = []
    
    # Generate different patterns based on condition
    if condition == "parkinsons":
        avg_press = 180
        std_press = 60
        avg_between = 300 
        std_between = 120
        keys = "The quick brown fox jumps over the lazy dog."
    elif condition == "essential_tremor":
        avg_press = 150
        std_press = 50
        avg_between = 250
        std_between = 90
        keys = "The quick brown fox jumps over the lazy dog."
    elif condition == "carpal_tunnel":
        avg_press = 130
        std_press = 25
        avg_between = 200
        std_between = 50
        keys = "The quick brown fox jumps over the lazy dog."
    elif condition == "multiple_sclerosis":
        avg_press = 160
        std_press = 45
        avg_between = 280
        std_between = 100
        keys = "The quick brown fox jumps over the lazy dog."
    else:  # normal
        avg_press = 120
        std_press = 20
        avg_between = 180
        std_between = 40
        keys = "The quick brown fox jumps over the lazy dog."
    
    # Generate keystroke events
    time_down = now
    for char in keys:
        # Generate press duration
        press_duration = int(np.random.normal(avg_press, std_press))
        press_duration = max(50, press_duration)  # Minimum 50ms
        
        # Generate pressure (0-1 scale)
        pressure = np.random.normal(0.7, 0.1)
        pressure = max(0.1, min(1.0, pressure))
        
        # Add keystroke
        keystroke = {
            "key": char,
            "timeDown": time_down,
            "timeUp": time_down + press_duration,
            "pressure": pressure
        }
        keystrokes.append(keystroke)
        
        # Generate time to next keystroke
        if char != keys[-1]:
            between = int(np.random.normal(avg_between, std_between))
            between = max(100, between)  # Minimum 100ms
            time_down += between
    
    # Add some backspaces if this isn't "normal" typing
    if condition != "normal":
        raw_keystrokes = keystrokes.copy()
        for i in range(3):  # Add 3 backspaces
            idx = np.random.randint(5, len(keystrokes) - 5)
            backspace = {
                "key": "Backspace",
                "timeDown": keystrokes[idx]["timeDown"] + 100,
                "timeUp": keystrokes[idx]["timeDown"] + 180,
                "pressure": 0.8
            }
            raw_keystrokes.insert(idx, backspace)
    else:
        raw_keystrokes = keystrokes.copy()
    
    # Structured test data
    return {
        "keystrokes": keystrokes,
        "rawKeystrokes": raw_keystrokes,
        "text": keys,
        "expected_condition": condition
    }

def main():
    # Initialize the processor
    print("Initializing KeystrokeProcessor...")
    analyzer = KeystrokeProcessor.get_instance()
    
    # Test with each condition
    conditions = ["normal", "parkinsons", "essential_tremor", "carpal_tunnel", "multiple_sclerosis"]
    
    print("\nTesting with generated data for each condition:")
    print("------------------------------------------------")
    
    for condition in conditions:
        print(f"\nGenerating test data for: {condition}")
        test_data = generate_test_data(condition)
        
        print(f"Analyzing typing pattern...")
        result = analyzer.analyze_typing(test_data)
        
        print(f"Expected: {test_data['expected_condition']}")
        print(f"Detected: {result['detected_condition']}")
        
        print("Probabilities:")
        for cond, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {cond}: {prob:.2f}")
        
        print("Key Features:")
        for feature, value in result['features'].items():
            print(f"  {feature}: {value:.2f}")
    
    print("\nGenerated data test complete.")
    
    # Test with real JSON data if available
    sample_path = os.path.join(os.path.dirname(__file__), "sample_typing_data.json")
    if os.path.exists(sample_path):
        print("\nTesting with sample data from file:")
        print("----------------------------------")
        
        with open(sample_path, 'r') as f:
            real_data = json.load(f)
        
        result = analyzer.analyze_typing(real_data)
        print(f"Detected condition: {result['detected_condition']}")
        
        print("Probabilities:")
        for cond, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {cond}: {prob:.2f}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main() 