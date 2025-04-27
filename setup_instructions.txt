# Speech Emotion Recognition System - Setup & Testing Instructions

## 1. Environment Setup

### Prerequisites
- Python 3.6+ installed
- pip package manager
- Microphone for testing (if you want to record your own voice)

### Clone the Repository
```
git clone <repository-url>
cd emotion-recognition-using-speech
```

### Create a  Environment (Recommended)
using Conda preffered 
```
```


### Install Dependencies
```
pip install -r requirements.txt
pip install -r requirements
```

Note: Some dependencies might need additional system libraries:
- `pyaudio` requires PortAudio
  - macOS: `brew install portaudio`
  - Ubuntu: `sudo apt-get install python3-pyaudio`
  - Windows: May need Microsoft Visual C++ Build Tools

## 2. Dataset Preparation

The system uses several datasets:
- RAVDESS and TESS (English)
- EMO-DB (German)
- Custom datasets (your own recordings)

### Option 1: Use Pre-existing Datasets
1. Download the datasets:
   - RAVDESS: https://zenodo.org/record/1188976
   - TESS: https://tspace.library.utoronto.ca/handle/1807/24487
   - EMO-DB: http://emodb.bilderbar.info/download/

2. Place the audio files in the proper directories:
   - RAVDESS & TESS: `data/training/` and `data/validation/`
   - EMO-DB: `data/emodb/`
   - Custom: `data/train-custom/` and `data/test-custom/`

### Option 2: Use Your Own Audio
Record your own audio samples and place them in:
- `data/train-custom/` (for training)
- `data/test-custom/` (for testing)

Format: WAV files with naming convention `[emotion]_[number].wav`

## 3. Generate CSV Files

If using your own datasets, run:
```
python create_csv.py
```

This will create metadata CSV files for training and testing.

## 4. Training the Model

### Basic Training
```
python emotion_recognition.py
```

This will:
1. Load audio data
2. Extract features (MFCC, Chroma, MEL)
3. Train various machine learning models
4. Determine the best model

### Custom Training
You can customize training by modifying `emotion_recognition.py` parameters:
- Emotions to detect
- Features to extract
- Machine learning models to use
- Dataset balancing options

### Grid Search for Hyperparameters
```
python grid_search.py
```

This will perform grid search to find optimal hyperparameters for models.

## 5. Testing the System

### Test with Recorded Audio
```
python test.py
```

This will:
1. Load the trained model
2. Prompt you to speak
3. Record your voice
4. Predict the emotion

### Test with Custom Parameters
```
python test.py --emotions "happy,sad,angry" --model "GradientBoostingClassifier"
```

Available options:
- `--emotions`: Comma-separated list of emotions to detect
- `--model`: Machine learning model to use

Available emotions: "neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "ps" (pleasant surprise), "boredom"

Available models: BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, MLPClassifier, etc.

## 6. Troubleshooting

### Audio Recording Issues
- Check if your microphone is properly connected and recognized
- Try adjusting the `THRESHOLD` and `SILENCE` parameters in `test.py`

### Installation Problems
- For PyAudio issues:
  - Windows: Try `pip install pipwin` followed by `pipwin install pyaudio`
  - macOS: Try `brew install portaudio` before installing requirements

### Model Performance
- If accuracy is low, try:
  - Adding more training data
  - Adjusting the emotions to detect (fewer emotions usually means better accuracy)
  - Trying different models with `--model` parameter

## 7. Advanced Usage

### Using Deep Learning Models
The system also supports deep learning models:
```
python deep_emotion_recognition.py
```

This uses TensorFlow to build and train neural networks for emotion recognition.

### Feature Extraction Only
If you want to extract features without training:
```
from utils import extract_feature
features = extract_feature("path/to/audio.wav", mfcc=True, chroma=True, mel=True)
```

## 8. Additional Notes

- The system works best in quiet environments
- Short audio clips (2-5 seconds) work better than long ones
- Balanced datasets (similar number of samples per emotion) lead to better results
- Default emotions are "sad", "neutral", and "happy" as they're the most distinguishable

