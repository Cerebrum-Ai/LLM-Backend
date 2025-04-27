import os
import numpy as np
import tempfile
import base64
from utils import extract_feature 
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle
from werkzeug.datastructures import FileStorage

class SimpleAudioAnalyzer:
    _instance = None
    _model = None
    _label_encoder = None
    _emotions = ["angry", "sad", "neutral", "happy", "fear"]
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if SimpleAudioAnalyzer._instance is not None:
            raise Exception("Use get_instance() instead")
        
        # Load or train a simple model
        model_path = "audio_emotion_model.pkl"
        if os.path.exists(model_path):
            print("Loading existing audio emotion model...")
            self._model = joblib.load(model_path)
        else:
            print("Training a new audio emotion model...")
            self._model = self._train_simple_model()
            # Save the model for future use
            joblib.dump(self._model, model_path)
        
        print("Audio Analyzer initialized successfully")
    
    def _train_simple_model(self):
        # Create a simple model based on a few examples
        print("Training a simple Random Forest model for emotion detection")
        
        features = []
        labels = []
        
        # Walk through the data directory to find examples of each emotion
        for emotion in self._emotions:
            count = 0
            limit = 20  # Get 20 examples of each emotion
            
            for root, dirs, files in os.walk("data"):
                if count >= limit:
                    break
                
                for file in files:
                    if count >= limit:
                        break
                    
                    if emotion in file.lower() and file.endswith(".wav"):
                        filepath = os.path.join(root, file)
                        feature = extract_feature(filepath, mfcc=True, chroma=True, mel=True)
                        
                        if feature is not None:
                            features.append(feature)
                            labels.append(emotion)
                            count += 1
                            print(f"Added {emotion} sample: {count}/{limit}")
        
        if not features:
            print("Warning: No training data found. Using a dummy model.")
            model = RandomForestClassifier(n_estimators=10)
            # Create some dummy data to initialize the model
            model.fit(np.random.random((10, 180)), np.random.choice(self._emotions, 10))
            return model
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        
        print(f"Model trained on {len(X)} samples")
        return model
    
    def analyze_audio(self, audio_data):
        """
        Analyze audio data to detect emotion.
        
        Args:
            audio_data: Audio file path, file-like object, or FileStorage object from Flask
            
        Returns:
            dict: Detected emotions with probabilities
        """
        try:
            # Handle FileStorage object from Flask
            if isinstance(audio_data, FileStorage):
                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    audio_data.save(temp_file)
                
                # Process the audio
                try:
                    print(f"Processing uploaded file saved at {temp_path}")
                    feature = extract_feature(temp_path, mfcc=True, chroma=True, mel=True)
                    if feature is not None:
                        emotion = self._model.predict([feature])[0]
                        probabilities = dict(zip(self._emotions, self._model.predict_proba([feature])[0]))
                        return {
                            "detected_emotion": emotion,
                            "probabilities": probabilities
                        }
                    else:
                        return {"error": f"Failed to extract features from {temp_path}", "detected_emotion": "unknown"}
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            # Handle base64 encoded audio data
            elif isinstance(audio_data, str) and audio_data.startswith('data:audio'):
                # Extract base64 content
                audio_content = audio_data.split(',')[1]
                decoded_audio = base64.b64decode(audio_content)
                
                # Create temporary file to process
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(decoded_audio)
                
                # Process the audio
                try:
                    feature = extract_feature(temp_path, mfcc=True, chroma=True, mel=True)
                    if feature is not None:
                        emotion = self._model.predict([feature])[0]
                        probabilities = dict(zip(self._emotions, self._model.predict_proba([feature])[0]))
                        return {
                            "detected_emotion": emotion,
                            "probabilities": probabilities
                        }
                    else:
                        return {"error": "Failed to extract features", "detected_emotion": "unknown"}
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                # Process directly if it's a file path
                feature = extract_feature(audio_data, mfcc=True, chroma=True, mel=True)
                if feature is not None:
                    emotion = self._model.predict([feature])[0]
                    probabilities = dict(zip(self._emotions, self._model.predict_proba([feature])[0]))
                    return {
                        "detected_emotion": emotion,
                        "probabilities": probabilities
                    }
                else:
                    return {"error": f"Failed to extract features from {audio_data}", "detected_emotion": "unknown"}
        except Exception as e:
            print(f"Error analyzing audio: {str(e)}")
            return {
                "error": str(e),
                "detected_emotion": "unknown"
            }
    
    def get_audio_features(self, audio_data):
        """
        Extract audio features for more detailed analysis.
        
        Args:
            audio_data: Audio data or file path
            
        Returns:
            dict: Extracted features
        """
        try:
            # Handle FileStorage object from Flask
            if isinstance(audio_data, FileStorage):
                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    audio_data.save(temp_file)
                
                # Process the audio
                try:
                    features = extract_feature(temp_path, mfcc=True, chroma=True, mel=True)
                    return {"features": features.tolist() if features is not None else None}
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            # Process the path
            elif isinstance(audio_data, str) and audio_data.startswith('data:audio'):
                # Create temp file as in analyze_audio
                audio_content = audio_data.split(',')[1]
                decoded_audio = base64.b64decode(audio_content)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(decoded_audio)
                
                try:
                    # Get features
                    features = extract_feature(temp_path, mfcc=True, chroma=True, mel=True)
                    return {"features": features.tolist() if features is not None else None}
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                # Direct processing
                features = extract_feature(audio_data, mfcc=True, chroma=True, mel=True)
                return {"features": features.tolist() if features is not None else None}
        except Exception as e:
            print(f"Error extracting audio features: {str(e)}")
            return {"error": str(e)}

# For backward compatibility
AudioAnalyzer = SimpleAudioAnalyzer 