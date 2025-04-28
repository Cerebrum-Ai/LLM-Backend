import os
import numpy as np
import json
import joblib
import pickle
import sys
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

class KeystrokeProcessor:
    """
    Analyses typing patterns to detect potential neurological conditions.
    
    This model looks at:
    - Key press duration
    - Time between keystrokes
    - Error rates and corrections
    - Rhythm patterns
    - Pressure (if available)
    """
    _instance = None
    _model = None
    _conditions = ["parkinsons", "essential_tremor", "carpal_tunnel", "multiple_sclerosis", "normal"]
    _scaler = None
    
    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the model - only called once"""
        if KeystrokeProcessor._instance is not None:
            raise Exception("Use get_instance() instead")
        
        # Get the directory where this file is located
        module_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load or train the model
        model_path = os.path.join(module_dir, "typing_pattern_model.pkl")
        scaler_path = os.path.join(module_dir, "typing_scaler.pkl")
        
        print(f"Looking for typing model at: {model_path}")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            print("Loading existing typing pattern model...")
            self._model = joblib.load(model_path)
            self._scaler = joblib.load(scaler_path)
        else:
            print("Training a new typing pattern model...")
            self._model, self._scaler = self._train_model()
            # Save the model and scaler
            joblib.dump(self._model, model_path)
            joblib.dump(self._scaler, scaler_path)
        
        print("Keystroke Processor initialized successfully")
    
    def _train_model(self):
        """Train a model for keystroke pattern analysis"""
        print("Training a Random Forest model for typing pattern analysis")
        
        # Create synthetic features for different conditions
        # In a real implementation, this would use real training data
        
        # Generate synthetic data
        np.random.seed(42)  # For reproducibility
        num_samples_per_condition = 100
        num_features = 12
        
        features = []
        labels = []
        
        # Feature meanings:
        # 0: avg_key_press_duration (ms)
        # 1: std_key_press_duration (ms)
        # 2: avg_time_between_keys (ms)
        # 3: std_time_between_keys (ms)
        # 4: error_rate (0-1)
        # 5: correction_rate (0-1)
        # 6: avg_pressure (0-1) - simulated
        # 7: std_pressure (0-1) - simulated
        # 8: typing_speed (chars per minute)
        # 9: rhythm_consistency (0-1)
        # 10: pause_frequency (pauses per minute)
        # 11: avg_pause_duration (ms)
        
        # Generate samples for each condition with specific patterns
        # Parkinson's: Longer key durations, higher variability, more errors
        for _ in range(num_samples_per_condition):
            parkinsons_sample = np.array([
                np.random.normal(180, 20),   # longer key press
                np.random.normal(60, 15),    # higher variability
                np.random.normal(300, 80),   # longer time between keys
                np.random.normal(120, 30),   # higher variability
                np.random.uniform(0.1, 0.25),# higher error rate
                np.random.uniform(0.2, 0.4), # higher correction rate
                np.random.normal(0.7, 0.2),  # inconsistent pressure
                np.random.normal(0.25, 0.1), # pressure variability
                np.random.normal(150, 40),   # slower typing
                np.random.uniform(0.4, 0.7), # less consistent rhythm
                np.random.normal(8, 3),      # more pauses
                np.random.normal(800, 200)   # longer pauses
            ])
            features.append(parkinsons_sample)
            labels.append("parkinsons")
        
        # Essential Tremor: Variable key durations, some errors, pauses
        for _ in range(num_samples_per_condition):
            tremor_sample = np.array([
                np.random.normal(150, 30),   # somewhat longer key press
                np.random.normal(50, 20),    # moderate variability
                np.random.normal(250, 70),   # moderate time between keys
                np.random.normal(90, 30),    # moderate variability
                np.random.uniform(0.05, 0.15),# moderate error rate
                np.random.uniform(0.1, 0.25),# moderate correction rate
                np.random.normal(0.6, 0.15), # moderate pressure
                np.random.normal(0.2, 0.1),  # pressure variability
                np.random.normal(200, 50),   # moderate typing speed
                np.random.uniform(0.5, 0.75),# moderate rhythm consistency
                np.random.normal(5, 2),      # moderate pauses
                np.random.normal(600, 150)   # moderate pauses
            ])
            features.append(tremor_sample)
            labels.append("essential_tremor")
        
        # Carpal Tunnel: More pressure, less variable, typing pauses
        for _ in range(num_samples_per_condition):
            carpal_sample = np.array([
                np.random.normal(130, 15),   # slightly longer key press
                np.random.normal(25, 10),    # lower variability
                np.random.normal(200, 40),   # normal time between keys
                np.random.normal(50, 20),    # normal variability
                np.random.uniform(0.02, 0.08),# lower error rate
                np.random.uniform(0.05, 0.15),# lower correction rate
                np.random.normal(0.85, 0.1), # higher pressure
                np.random.normal(0.15, 0.05),# lower pressure variability
                np.random.normal(240, 40),   # normal typing speed
                np.random.uniform(0.7, 0.9), # consistent rhythm
                np.random.normal(4, 1.5),    # fewer pauses
                np.random.normal(1000, 300)  # longer pauses when they occur
            ])
            features.append(carpal_sample)
            labels.append("carpal_tunnel")
        
        # Multiple Sclerosis: Fatigue pattern, starts normal, slows down
        for _ in range(num_samples_per_condition):
            ms_sample = np.array([
                np.random.normal(160, 40),   # variable key press
                np.random.normal(45, 25),    # highly variable
                np.random.normal(280, 90),   # longer and variable time between keys
                np.random.normal(100, 40),   # high variability
                np.random.uniform(0.08, 0.18),# moderate error rate
                np.random.uniform(0.15, 0.3),# moderate correction rate
                np.random.normal(0.65, 0.25),# variable pressure
                np.random.normal(0.3, 0.15), # high pressure variability
                np.random.normal(180, 60),   # variable typing speed
                np.random.uniform(0.3, 0.6), # inconsistent rhythm
                np.random.normal(7, 2.5),    # frequent pauses
                np.random.normal(750, 250)   # variable pause length
            ])
            features.append(ms_sample)
            labels.append("multiple_sclerosis")
        
        # Normal typing: Consistent, fewer errors, smooth rhythm
        for _ in range(num_samples_per_condition):
            normal_sample = np.array([
                np.random.normal(120, 10),   # normal key press duration
                np.random.normal(20, 8),     # low variability
                np.random.normal(180, 30),   # normal time between keys
                np.random.normal(40, 15),    # low variability
                np.random.uniform(0.01, 0.05),# low error rate
                np.random.uniform(0.02, 0.1),# low correction rate
                np.random.normal(0.7, 0.1),  # normal pressure
                np.random.normal(0.1, 0.03), # consistent pressure
                np.random.normal(280, 30),   # faster typing
                np.random.uniform(0.8, 0.95),# consistent rhythm
                np.random.normal(2, 1),      # few pauses
                np.random.normal(400, 100)   # shorter pauses
            ])
            features.append(normal_sample)
            labels.append("normal")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train a model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        print(f"Model trained on {len(X)} samples")
        return model, scaler
    
    def _extract_typing_features(self, typing_data):
        """
        Extract features from typing data.
        
        Expected input structure:
        {
            "keystrokes": [
                {"key": "a", "timeDown": 1620136589000, "timeUp": 1620136589080, "pressure": 0.8},
                {"key": "b", "timeDown": 1620136589200, "timeUp": 1620136589270, "pressure": 0.7},
                ...
            ],
            "text": "The actual text that was typed, including errors and corrections"
        }
        """
        try:
            keystrokes = typing_data.get("keystrokes", [])
            if not keystrokes or len(keystrokes) < 10:
                raise ValueError("Not enough keystroke data (minimum 10 required)")
            
            # Extract timing and pressure information
            press_durations = []
            between_keys = []
            pressures = []
            
            for i, event in enumerate(keystrokes):
                # Key press duration
                if "timeDown" in event and "timeUp" in event:
                    duration = event["timeUp"] - event["timeDown"]
                    press_durations.append(duration)
                
                # Time between keystrokes
                if i > 0 and "timeDown" in event and "timeDown" in keystrokes[i-1]:
                    between = event["timeDown"] - keystrokes[i-1]["timeDown"]
                    if 0 < between < 2000:  # Filter out pauses > 2 seconds
                        between_keys.append(between)
                    
                # Pressure if available
                if "pressure" in event:
                    pressures.append(event["pressure"])
            
            # Calculate statistics
            avg_press_duration = np.mean(press_durations) if press_durations else 130
            std_press_duration = np.std(press_durations) if press_durations else 20
            
            avg_between_keys = np.mean(between_keys) if between_keys else 200
            std_between_keys = np.std(between_keys) if between_keys else 40
            
            # If pressure data is available
            avg_pressure = np.mean(pressures) if pressures else 0.7
            std_pressure = np.std(pressures) if pressures else 0.1
            
            # Calculate typing speed (chars per minute)
            if len(keystrokes) >= 2:
                first_time = keystrokes[0]["timeDown"]
                last_time = keystrokes[-1]["timeDown"]
                total_time_min = (last_time - first_time) / 60000  # convert ms to minutes
                typing_speed = len(keystrokes) / total_time_min if total_time_min > 0 else 250
            else:
                typing_speed = 250  # default
            
            # Calculate error and correction rates based on text
            text = typing_data.get("text", "")
            error_rate = 0.03  # default
            correction_rate = 0.05  # default
            
            if "rawKeystrokes" in typing_data:
                raw = typing_data["rawKeystrokes"]
                backspaces = sum(1 for k in raw if k.get("key") == "Backspace")
                error_rate = backspaces / len(raw) if len(raw) > 0 else 0.03
                correction_rate = backspaces / len(text) if len(text) > 0 else 0.05
            
            # Calculate rhythm consistency
            # Higher value means more consistent timing between keystrokes
            if len(between_keys) > 5:
                rhythm_consistency = 1.0 - (std_between_keys / avg_between_keys) 
                rhythm_consistency = max(0.1, min(0.95, rhythm_consistency))
            else:
                rhythm_consistency = 0.8  # default
            
            # Calculate pause patterns
            long_pauses = [b for b in between_keys if b > 500]  # pauses > 500ms
            pause_frequency = len(long_pauses) / (len(keystrokes) / 60) if len(keystrokes) > 0 else 2
            avg_pause_duration = np.mean(long_pauses) if long_pauses else 400
            
            # Build feature vector in expected order
            features = np.array([
                avg_press_duration,
                std_press_duration,
                avg_between_keys,
                std_between_keys,
                error_rate,
                correction_rate,
                avg_pressure,
                std_pressure,
                typing_speed,
                rhythm_consistency,
                pause_frequency,
                avg_pause_duration
            ])
            
            return features
            
        except Exception as e:
            print(f"Error extracting typing features: {str(e)}")
            raise
    
    def analyze_typing(self, typing_data):
        """
        Analyze typing data to detect potential neurological conditions.
        
        Args:
            typing_data: Dictionary containing keystroke data 
            
        Returns:
            dict: Detected condition and probabilities
        """
        try:
            if not typing_data or "keystrokes" not in typing_data:
                return {
                    "error": "Invalid typing data format. Expected 'keystrokes' field.",
                    "detected_condition": "unknown"
                }
            
            # Extract features from raw typing data
            features = self._extract_typing_features(typing_data)
            
            # Scale features
            scaled_features = self._scaler.transform([features])
            
            # Get prediction and probabilities
            condition = self._model.predict(scaled_features)[0]
            probabilities = dict(zip(self._conditions, self._model.predict_proba(scaled_features)[0]))
            
            return {
                "detected_condition": condition,
                "probabilities": probabilities,
                "features": {
                    "key_press_duration": float(features[0]),
                    "duration_variability": float(features[1]),
                    "time_between_keys": float(features[2]),
                    "rhythm_variability": float(features[3]),
                    "error_rate": float(features[4]),
                    "typing_speed": float(features[8]),
                    "rhythm_consistency": float(features[9]),
                    "pause_frequency": float(features[10])
                }
            }
            
        except Exception as e:
            print(f"Error analyzing typing patterns: {str(e)}")
            return {
                "error": str(e),
                "detected_condition": "unknown"
            }

# For backwards compatibility if needed
TypingAnalyzer = KeystrokeProcessor 