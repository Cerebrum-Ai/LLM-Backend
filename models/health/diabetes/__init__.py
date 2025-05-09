# This file makes the models directory a Python package 

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from .diabetes_model import DiabetesPredictor

def train_diabetes_model():
    """Train the diabetes prediction model and save it"""
    try:
        # Load the dataset
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'diabetes_prediction_dataset.csv')
        if not os.path.exists(data_path):
            print(f"Dataset not found at {data_path}. Please download it from the original repository.")
            return None
        
        print("Loading diabetes dataset...")
        df = pd.read_csv(data_path)
        
        # Prepare features and target
        X = df.drop('diabetes', axis=1)
        y = df['diabetes']
        
        # Encode categorical variables
        categorical_cols = ['gender', 'smoking_history']
        encoder = OrdinalEncoder()
        X[categorical_cols] = encoder.fit_transform(X[categorical_cols])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training Random Forest model...")
        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Create predictor instance
        predictor = DiabetesPredictor()
        predictor.model = model
        predictor.encoder = encoder  # Save the encoder for future predictions
        
        # Save the model
        model_path = os.path.join(os.path.dirname(__file__), 'diabetes_model.joblib')
        predictor.save_model(model_path)
        
        # Calculate and print accuracy
        accuracy = model.score(X_test, y_test)
        print(f"Diabetes model trained successfully with accuracy: {accuracy:.2f}")
        
        return predictor
        
    except Exception as e:
        print(f"Error training diabetes model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_diabetes_predictor():
    """Get or initialize the diabetes predictor instance"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'diabetes_model.joblib')
        
        if not os.path.exists(model_path):
            print("Diabetes model not found. Training new model...")
            predictor = train_diabetes_model()
            if predictor is None:
                print("Failed to train diabetes model")
                return None
            return predictor
        
        print("Loading existing diabetes model...")
        predictor = DiabetesPredictor()
        predictor.load_model(model_path)
        return predictor
    except Exception as e:
        print(f"Error loading diabetes model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Initialize the model when the module is imported
print("\nInitializing diabetes prediction model...")
diabetes_predictor = get_diabetes_predictor()
if diabetes_predictor is None:
    print("Warning: Failed to initialize diabetes prediction model")
else:
    print("Diabetes prediction model initialized successfully") 