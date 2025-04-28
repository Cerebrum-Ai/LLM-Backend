#!/usr/bin/env python3
"""
Typing Visualizer for Cerebrum AI

This script visualizes keystroke data in real-time to help analyze typing patterns.
It can either:
1. Capture live typing data and visualize it
2. Load recorded typing data from a file
3. Connect to the ML service and visualize the features being extracted
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse
import json
import time
import threading
import keyboard
import requests
import os
import sys
from datetime import datetime
from collections import deque

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TypingVisualizer:
    def __init__(self, mode="live", data_file=None, ml_url="http://localhost:9000"):
        """
        Initialize the typing visualizer
        
        Args:
            mode (str): "live", "file", or "ml-connect"
            data_file (str): Path to a JSON file with typing data (for "file" mode)
            ml_url (str): URL of the ML service (for "ml-connect" mode)
        """
        self.mode = mode
        self.data_file = data_file
        self.ml_url = ml_url
        
        # Data structures
        self.keystrokes = []
        self.live_data = {
            "keystrokes": [],
            "text": ""
        }
        
        # For visualization
        self.key_times = deque(maxlen=100)
        self.key_durations = deque(maxlen=100)
        self.key_pressures = deque(maxlen=100)
        self.time_between_keys = deque(maxlen=100)
        
        # Tracking
        self.last_key_time = None
        self.recording = False
        self.analysis_results = None
        
        # Setup visualization
        self.setup_plot()
        
    def setup_plot(self):
        """Set up the matplotlib visualization"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('Cerebrum AI - Typing Pattern Visualizer')
        
        # Subplot layout
        self.ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)  # Keypress duration over time
        self.ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)  # Time between keypresses
        self.ax3 = plt.subplot2grid((3, 3), (0, 2))             # Key duration histogram
        self.ax4 = plt.subplot2grid((3, 3), (1, 2))             # Time between keys histogram
        self.ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3)  # Text area for analysis results
        
        # Style and labels
        self.ax1.set_title('Key Press Duration Over Time')
        self.ax1.set_ylabel('Duration (ms)')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('Time Between Keys')
        self.ax2.set_ylabel('Time (ms)')
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_title('Duration Histogram')
        self.ax3.set_xlabel('Duration (ms)')
        self.ax3.grid(True, alpha=0.3)
        
        self.ax4.set_title('Interval Histogram')
        self.ax4.set_xlabel('Time (ms)')
        self.ax4.grid(True, alpha=0.3)
        
        self.ax5.set_title('Analysis Results')
        self.ax5.axis('off')
        
        plt.tight_layout()
        
    def start_recording(self):
        """Start recording live typing data"""
        print("🎬 Starting to record typing data...")
        self.recording = True
        self.live_data = {
            "keystrokes": [],
            "text": ""
        }
        self.start_time = time.time() * 1000
        
        # Set up keyboard hooks
        keyboard.on_press(self.on_key_press)
        keyboard.on_release(self.on_key_release)
        
    def stop_recording(self):
        """Stop recording live typing data"""
        if not self.recording:
            return
            
        print("⏹️ Stopping recording")
        self.recording = False
        
        # Remove keyboard hooks
        keyboard.unhook_all()
        
        # Save data
        if len(self.live_data["keystrokes"]) > 0:
            self.save_recording()
            
    def save_recording(self):
        """Save the recorded typing data to a file"""
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(data_dir, exist_ok=True)
        
        filename = os.path.join(data_dir, f"typing_data_{int(time.time())}.json")
        with open(filename, 'w') as f:
            json.dump(self.live_data, f, indent=2)
        print(f"💾 Saved recording to {filename}")
            
    def on_key_press(self, event):
        """Handle key press events"""
        if not self.recording:
            return
            
        current_time = time.time() * 1000
        
        # Calculate time between keys
        if self.last_key_time is not None:
            between = current_time - self.last_key_time
            if 0 < between < 2000:  # Filter out pauses > 2 seconds
                self.time_between_keys.append(between)
        
        self.last_key_time = current_time
        
        # Store key data
        key_data = {
            "key": event.name,
            "timeDown": current_time,
            "timeUp": None,
            "pressure": 0.7,  # Simulated pressure
            "event": "keydown"
        }
        
        # Add to live data
        self.live_data["keystrokes"].append(key_data)
        
        # Update text if a printable character
        if len(event.name) == 1:
            self.live_data["text"] += event.name
        elif event.name == "space":
            self.live_data["text"] += " "
        elif event.name == "backspace" and len(self.live_data["text"]) > 0:
            self.live_data["text"] = self.live_data["text"][:-1]
            
    def on_key_release(self, event):
        """Handle key release events"""
        if not self.recording:
            return
            
        current_time = time.time() * 1000
        
        # Find the matching key press
        for key_data in reversed(self.live_data["keystrokes"]):
            if key_data["key"] == event.name and key_data["timeUp"] is None:
                key_data["timeUp"] = current_time
                key_data["event"] = "keyup"
                
                # Calculate duration
                duration = key_data["timeUp"] - key_data["timeDown"]
                self.key_durations.append(duration)
                self.key_times.append(key_data["timeDown"] - self.start_time)
                self.key_pressures.append(key_data["pressure"])
                break
                
    def load_data_from_file(self):
        """Load typing data from a JSON file"""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                
            if "keystrokes" not in data:
                print("❌ Invalid data format: no 'keystrokes' field found")
                return False
                
            self.live_data = data
            
            # Process the data for visualization
            if len(data["keystrokes"]) > 0:
                base_time = data["keystrokes"][0]["timeDown"]
                
                for i, key in enumerate(data["keystrokes"]):
                    if "timeDown" in key and "timeUp" in key:
                        # Key duration
                        duration = key["timeUp"] - key["timeDown"]
                        self.key_durations.append(duration)
                        self.key_times.append(key["timeDown"] - base_time)
                        
                        # Pressure if available
                        pressure = key.get("pressure", 0.7)
                        self.key_pressures.append(pressure)
                    
                    # Time between keys
                    if i > 0 and "timeDown" in key and "timeDown" in data["keystrokes"][i-1]:
                        between = key["timeDown"] - data["keystrokes"][i-1]["timeDown"]
                        if 0 < between < 2000:  # Filter out pauses > 2 seconds
                            self.time_between_keys.append(between)
            
            print(f"📊 Loaded {len(data['keystrokes'])} keystrokes from file")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data from file: {str(e)}")
            return False
            
    def analyze_with_ml_service(self):
        """Send the current typing data to the ML service for analysis"""
        if len(self.live_data["keystrokes"]) < 10:
            print("❌ Not enough keystroke data for analysis (min 10)")
            return
            
        try:
            response = requests.post(
                f"{self.ml_url}/ml/process",
                json={
                    "url": self.live_data,
                    "data_type": "typing",
                    "model": "pattern"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                self.analysis_results = response.json()
                print("✅ Received analysis from ML service")
            else:
                print(f"❌ Error from ML service: {response.text}")
                
        except Exception as e:
            print(f"❌ Error connecting to ML service: {str(e)}")
            
    def update_plot(self, frame):
        """Update the visualization plots with current data"""
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        
        # Reset titles and style
        self.ax1.set_title('Key Press Duration Over Time')
        self.ax1.set_ylabel('Duration (ms)')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('Time Between Keys')
        self.ax2.set_ylabel('Time (ms)')
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_title('Duration Histogram')
        self.ax3.set_xlabel('Duration (ms)')
        self.ax3.grid(True, alpha=0.3)
        
        self.ax4.set_title('Interval Histogram')
        self.ax4.set_xlabel('Time (ms)')
        self.ax4.grid(True, alpha=0.3)
        
        self.ax5.set_title('Analysis Results')
        self.ax5.axis('off')
        
        # Plot data if available
        if len(self.key_times) > 0 and len(self.key_durations) > 0:
            times = list(self.key_times)
            durations = list(self.key_durations)
            
            # Key press duration over time
            self.ax1.scatter(times, durations, color='cyan', alpha=0.7)
            if len(times) > 5:
                # Add trend line
                z = np.polyfit(times, durations, 1)
                p = np.poly1d(z)
                self.ax1.plot(times, p(times), 'r--', alpha=0.8)
            
        if len(self.time_between_keys) > 0:
            # Time between keys over time
            times = list(range(len(self.time_between_keys)))
            intervals = list(self.time_between_keys)
            self.ax2.scatter(times, intervals, color='lime', alpha=0.7)
            if len(times) > 5:
                # Add trend line
                z = np.polyfit(times, intervals, 1)
                p = np.poly1d(z)
                self.ax2.plot(times, p(times), 'r--', alpha=0.8)
        
        # Histograms
        if len(self.key_durations) > 4:
            self.ax3.hist(self.key_durations, bins=10, color='cyan', alpha=0.7)
            
        if len(self.time_between_keys) > 4:
            self.ax4.hist(self.time_between_keys, bins=10, color='lime', alpha=0.7)
            
        # Display analysis results
        if self.analysis_results:
            result_text = "ML Analysis Results:\n\n"
            
            if "detected_condition" in self.analysis_results:
                result_text += f"Detected: {self.analysis_results['detected_condition']}\n\n"
                
            if "probabilities" in self.analysis_results:
                result_text += "Probabilities:\n"
                for cond, prob in sorted(self.analysis_results["probabilities"].items(), 
                                         key=lambda x: x[1], reverse=True):
                    result_text += f"  {cond}: {prob:.2f}\n"
                    
            if "features" in self.analysis_results:
                result_text += "\nKey Features:\n"
                for feature, value in self.analysis_results["features"].items():
                    result_text += f"  {feature}: {value:.2f}\n"
                    
            self.ax5.text(0.05, 0.95, result_text, transform=self.ax5.transAxes, 
                         verticalalignment='top', fontsize=9, family='monospace')
        else:
            # Display live data status
            status = "🎬 Recording live data..." if self.recording else "⏹️ Recording stopped"
            stats_text = (f"{status}\n\n"
                         f"Keystrokes: {len(self.live_data['keystrokes'])}\n"
                         f"Avg duration: {np.mean(self.key_durations):.2f} ms\n"
                         f"Avg interval: {np.mean(self.time_between_keys):.2f} ms\n\n"
                         f"Press 'a' to analyze with ML service\n"
                         f"Press 's' to save recording\n"
                         f"Press 'q' to quit")
            
            self.ax5.text(0.05, 0.95, stats_text, transform=self.ax5.transAxes, 
                         verticalalignment='top', fontsize=10)
            
    def on_key(self, event):
        """Handle keyboard events in the matplotlib window"""
        if event.key == 'q':
            plt.close(self.fig)
        elif event.key == 's' and self.mode == "live":
            self.save_recording()
        elif event.key == 'a':
            threading.Thread(target=self.analyze_with_ml_service, daemon=True).start()
            
    def run(self):
        """Run the visualizer"""
        # Set up based on mode
        if self.mode == "file":
            if not self.load_data_from_file():
                return
        elif self.mode == "live":
            self.start_recording()
        
        # Set up animation
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100)
        
        # Connect key press event
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Show plot
        plt.tight_layout()
        plt.show()
        
        # Clean up
        if self.mode == "live":
            self.stop_recording()

def main():
    parser = argparse.ArgumentParser(description="Cerebrum AI - Typing Pattern Visualizer")
    parser.add_argument("--mode", choices=["live", "file"], default="live",
                       help="Visualization mode: live capture or from file")
    parser.add_argument("--file", help="JSON file with typing data (for file mode)")
    parser.add_argument("--ml-url", default="http://localhost:9000", 
                       help="URL of the ML service")
    
    args = parser.parse_args()
    
    if args.mode == "file" and not args.file:
        parser.error("--file is required when mode is 'file'")
        
    visualizer = TypingVisualizer(mode=args.mode, data_file=args.file, ml_url=args.ml_url)
    visualizer.run()

if __name__ == "__main__":
    main() 