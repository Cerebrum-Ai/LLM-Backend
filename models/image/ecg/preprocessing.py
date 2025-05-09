import numpy as np
from scipy import signal
import torch

def normalize_ecg(ecg_signal):
    """Normalize ECG signal to zero mean and unit variance"""
    mean = np.mean(ecg_signal, axis=1, keepdims=True)
    std = np.std(ecg_signal, axis=1, keepdims=True)
    return (ecg_signal - mean) / (std + 1e-8)

def filter_ecg(ecg_signal, fs=500, lowcut=0.5, highcut=45):
    """Apply bandpass filter to ECG signal"""
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, ecg_signal, axis=1)

def segment_ecg(ecg_signal, segment_length=5000, overlap=0.5):
    """Segment ECG signal into fixed-length segments with overlap"""
    n_channels, n_samples = ecg_signal.shape
    step = int(segment_length * (1 - overlap))
    segments = []
    
    for start in range(0, n_samples - segment_length + 1, step):
        segment = ecg_signal[:, start:start + segment_length]
        segments.append(segment)
    
    return np.array(segments)

def preprocess_ecg(ecg_signal, fs=500, segment_length=5000):
    """Complete preprocessing pipeline for ECG signal"""
    # Ensure correct shape (channels, time)
    if len(ecg_signal.shape) == 1:
        ecg_signal = ecg_signal.reshape(1, -1)
    elif len(ecg_signal.shape) == 2 and ecg_signal.shape[0] > ecg_signal.shape[1]:
        ecg_signal = ecg_signal.T
    
    # Apply preprocessing steps
    ecg_filtered = filter_ecg(ecg_signal, fs=fs)
    ecg_normalized = normalize_ecg(ecg_filtered)
    ecg_segments = segment_ecg(ecg_normalized, segment_length=segment_length)
    
    return ecg_segments

def prepare_batch(ecg_segments, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Prepare batch of ECG segments for model input"""
    return torch.FloatTensor(ecg_segments).to(device)

def generate_text_description(ecg_metadata):
    """Generate text description from ECG metadata"""
    description = f"ECG recording with {ecg_metadata.get('num_leads', 12)} leads. "
    if 'patient_age' in ecg_metadata:
        description += f"Patient age: {ecg_metadata['patient_age']}. "
    if 'patient_gender' in ecg_metadata:
        description += f"Patient gender: {ecg_metadata['patient_gender']}. "
    if 'recording_duration' in ecg_metadata:
        description += f"Recording duration: {ecg_metadata['recording_duration']} seconds. "
    if 'sampling_rate' in ecg_metadata:
        description += f"Sampling rate: {ecg_metadata['sampling_rate']} Hz. "
    return description 