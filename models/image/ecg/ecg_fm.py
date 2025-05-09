import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np

class ECGFeatureExtractor(nn.Module):
    def __init__(self, input_channels=12, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, input_channels, sequence_length)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        return x

class ECG_FM(nn.Module):
    def __init__(self, 
                 input_channels=12,
                 hidden_dim=256,
                 num_classes=5,
                 model_name="microsoft/BiomedVLP-CXR-BERT-general"):
        super().__init__()
        
        # ECG feature extractor
        self.ecg_encoder = ECGFeatureExtractor(input_channels, hidden_dim)
        
        # Text encoder (using BiomedVLP-CXR-BERT)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + 768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, ecg_signal, text_input):
        # Process ECG signal
        ecg_features = self.ecg_encoder(ecg_signal)
        ecg_features = ecg_features.mean(dim=2)  # Global average pooling
        
        # Process text
        text_outputs = self.text_encoder(**text_input)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Feature fusion
        combined_features = torch.cat([ecg_features, text_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits
    
    def prepare_text_input(self, text):
        """Prepare text input for the model"""
        return self.text_tokenizer(text, 
                                 padding=True, 
                                 truncation=True, 
                                 max_length=512, 
                                 return_tensors="pt")

def load_ecg_fm_model(model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load the ECG-FM model"""
    model = ECG_FM()
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_ecg(model, ecg_signal, text_description, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Make prediction using the ECG-FM model"""
    model.eval()
    with torch.no_grad():
        # Prepare inputs
        ecg_signal = torch.FloatTensor(ecg_signal).to(device)
        if len(ecg_signal.shape) == 2:
            ecg_signal = ecg_signal.unsqueeze(0)  # Add batch dimension
            
        text_input = model.prepare_text_input(text_description)
        text_input = {k: v.to(device) for k, v in text_input.items()}
        
        # Get prediction
        logits = model(ecg_signal, text_input)
        probabilities = F.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        
        return {
            'prediction': prediction.item(),
            'probabilities': probabilities.cpu().numpy(),
            'logits': logits.cpu().numpy()
        } 