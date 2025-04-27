import soundfile
import librosa
import numpy as np
import pickle
import os
import sys

# Ensure the current directory is in the path for imports
module_dir = os.path.dirname(os.path.abspath(__file__))
if module_dir not in sys.path:
    sys.path.append(module_dir)
    
from convert_wavs import convert_audio


AVAILABLE_EMOTIONS = {
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "ps", # pleasant surprised
    "boredom"
}


def get_label(audio_config):
    """Returns label corresponding to which features are to be extracted
        e.g:
    audio_config = {'mfcc': True, 'chroma': True, 'contrast': False, 'tonnetz': False, 'mel': False}
    get_label(audio_config): 'mfcc-chroma'
    """
    features = ["mfcc", "chroma", "mel", "contrast", "tonnetz"]
    label = ""
    for feature in features:
        if audio_config[feature]:
            label += f"{feature}-"
    return label.rstrip("-")


def get_dropout_str(dropout, n_layers=3):
    if isinstance(dropout, list):
        return "_".join([ str(d) for d in dropout])
    elif isinstance(dropout, float):
        return "_".join([ str(dropout) for i in range(n_layers) ])


def get_first_letters(emotions):
    return "".join(sorted([ e[0].upper() for e in emotions ]))


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


def get_best_estimators(classification):
    """
    Loads the estimators that are pickled in `grid` folder
    Note that if you want to use different or more estimators,
    you can fine tune the parameters in `grid_search.py` script
    and run it again ( may take hours )
    """
    if classification:
        return pickle.load(open("grid/best_classifiers.pickle", "rb"))
    else:
        return pickle.load(open("grid/best_regressors.pickle", "rb"))


def get_audio_config(features_list):
    """
    Converts a list of features into a dictionary understandable by
    `data_extractor.AudioExtractor` class
    """
    audio_config = {'mfcc': False, 'chroma': False, 'mel': False, 'contrast': False, 'tonnetz': False}
    for feature in features_list:
        if feature not in audio_config:
            raise TypeError(f"Feature passed: {feature} is not recognized.")
        audio_config[feature] = True
    return audio_config