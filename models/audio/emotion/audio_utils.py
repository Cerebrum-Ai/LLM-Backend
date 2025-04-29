import soundfile
import librosa
import numpy as np
import os
import sys
from pathlib import Path

# Import convert_wavs from the same directory
from .convert_wavs import convert_audio

AVAILABLE_EMOTIONS = {
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "ps",  # pleasant surprised
    "boredom"
}

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    
    # Check if the file exists
    if not os.path.exists(file_name):
        print(f"Error: File does not exist: {file_name}")
        return None
        
    try:
        print(f"Attempting to open audio file: {file_name}")
        with soundfile.SoundFile(file_name) as sound_file:
            print(f"Successfully opened file: {file_name}, format: {sound_file.format}, sample rate: {sound_file.samplerate}")
            pass
    except Exception as e:
        # if the file format isn't supported by librosa
        # or the file has some other issues
        # convert the file to a compatible format
        try:
            print(f"[+] Converting {file_name} to a compatible format. Error was: {str(e)}")
            # Try to convert the file in place
            convert_audio(file_name)
            print(f"Conversion complete for {file_name}")
        except Exception as e:
            # if conversion still failed
            # then inform the user that the file is probably damaged
            print(f"[Error]: Failed to convert audio file: {str(e)}")
            return None

    try:
        with soundfile.SoundFile(file_name) as sound_file:
            print(f"Reading audio data from {file_name}, sample rate: {sound_file.samplerate}")
            X = sound_file.read(dtype="float32")
            
            # Convert stereo to mono if needed
            if X.ndim > 1:
                print(f"Converting stereo audio to mono (original shape: {X.shape})")
                X = np.mean(X, axis=1)
                print(f"Converted to mono shape: {X.shape}")
            
            sample_rate = sound_file.samplerate
            result = np.array([])
            if mfcc:
                print(f"Extracting MFCC features from {file_name}")
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result = np.hstack((result, mfccs))
            if chroma:
                print(f"Extracting chroma features from {file_name}")
                chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate).T, axis=0)
                result = np.hstack((result, chroma))
            if mel:
                print(f"Extracting MEL spectrogram features from {file_name}")
                # Fix for updated librosa version
                mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
                result = np.hstack((result, mel))
            if contrast:
                print(f"Extracting spectral contrast features from {file_name}")
                contrast = np.mean(librosa.feature.spectral_contrast(y=X, sr=sample_rate).T, axis=0)
                result = np.hstack((result, contrast))
            if tonnetz:
                print(f"Extracting tonnetz features from {file_name}")
                tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
                result = np.hstack((result, tonnetz))
        print(f"Successfully extracted features from {file_name}")
        return result
    except Exception as e:
        print(f"[Error]: Failed to extract features: {str(e)}")
        return None
