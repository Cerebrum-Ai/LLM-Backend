from .ecg_fm import load_ecg_fm_model, predict_ecg
from .preprocessing import preprocess_ecg, prepare_batch, generate_text_description
import numpy as np
import torch
import torch.nn.functional as F

class ECG_FM_Handler:
    def __init__(self, model_path=None, device='cuda'):
        """Initialize the ECG-FM model handler"""
        self.model = load_ecg_fm_model(model_path, device)
        self.device = device
        
    def process_ecg(self, ecg_signal, text_description):
        """Process ECG signal and return predictions"""
        # Move signal to device
        ecg_signal = ecg_signal.to(self.device)
        
        # Tokenize text description
        text_input = self.model.text_tokenizer(
            text_description,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(ecg_signal, text_input)
            probabilities = F.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        # Convert to numpy for easier handling
        prediction = prediction.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        
        return {
            'prediction': int(prediction[0]),
            'probabilities': probabilities[0].tolist(),
            'segment_predictions': [int(prediction[0])],  # Single prediction for now
            'segment_probabilities': [probabilities[0].tolist()]  # Single probability for now
        }
    
    def get_model_summary(self):
        """Get model architecture summary"""
        return str(self.model)
    
    def save_model(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
    
    @staticmethod
    def get_class_names():
        """Get class names for predictions"""
        return [
            'Normal',
            'Atrial Fibrillation',
            'First-degree AV Block',
            'Left Bundle Branch Block',
            'Right Bundle Branch Block'
        ] 