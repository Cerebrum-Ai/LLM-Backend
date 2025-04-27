import os
import numpy as np
from utils import extract_feature

# Create a simple function to test audio emotion extraction
def test_audio_features():
    print("Testing audio feature extraction...")
    
    # Find a WAV file to test with
    test_file = None
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith(".wav"):
                test_file = os.path.join(root, file)
                print(f"Found audio file: {test_file}")
                break
        if test_file:
            break
    
    if not test_file:
        print("No WAV files found in the data directory!")
        return
    
    # Extract features
    features = extract_feature(test_file, mfcc=True, chroma=True, mel=True)
    
    if features is not None:
        print(f"Successfully extracted features of shape: {features.shape}")
        print("Feature extraction works correctly!")
        return True
    else:
        print("Feature extraction failed!")
        return False

if __name__ == "__main__":
    test_audio_features() 