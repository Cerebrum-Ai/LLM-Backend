import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
import joblib
import os

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.encoder = None  # Will be set during training
        self.feature_names = [
            'gender', 'age', 'hypertension', 'heart_disease', 
            'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'
        ]
        
    def preprocess_input(self, data):
        """Preprocess the input data for prediction"""
        # Convert input data to DataFrame
        df = pd.DataFrame([data])
        
        # Rename hba1c_level to HbA1c_level if present
        if 'hba1c_level' in df.columns:
            df = df.rename(columns={'hba1c_level': 'HbA1c_level'})
        
        # Encode categorical variables if encoder is available
        categorical_cols = ['gender', 'smoking_history']
        if self.encoder is not None:
            df[categorical_cols] = self.encoder.transform(df[categorical_cols])
        else:
            # If no encoder is available, create a new one
            self.encoder = OrdinalEncoder()
            df[categorical_cols] = self.encoder.fit_transform(df[categorical_cols])
        
        return df[self.feature_names]
    
    def predict(self, data):
        """Make prediction for the given input data"""
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model first.")
            
        # Preprocess the input
        processed_data = self.preprocess_input(data)
        
        # Make prediction
        prediction = self.model.predict(processed_data)
        probability = self.model.predict_proba(processed_data)
        
        return {
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),
            'message': 'High risk of diabetes' if prediction[0] == 1 else 'Low risk of diabetes'
        }
    
    def load_model(self, model_path):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)
        
    def save_model(self, model_path):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump(self.model, model_path) 