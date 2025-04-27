# Complete Setup and Testing Guide for Speech Emotion Recognition with LLM-Backend

This guide walks you through setting up both the emotion recognition system and integrating it with the LLM-Backend application for audio analysis.

## Part 1: Setting Up the Emotion Recognition System

### 1. Environment Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate the environment
# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate

# Clone the emotion recognition repository
git clone https://github.com/your-repo/emotion-recognition-using-speech.git
cd emotion-recognition-using-speech

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Option A: Download pre-existing datasets
```bash
# Create necessary directories
mkdir -p data/training data/validation data/emodb data/train-custom data/test-custom

# Download datasets (examples)
# RAVDESS: https://zenodo.org/record/1188976
# TESS: https://tspace.library.utoronto.ca/handle/1807/24487
# EMO-DB: http://emodb.bilderbar.info/download/

# Place audio files in proper directories
# RAVDESS & TESS -> data/training/ and data/validation/
# EMO-DB -> data/emodb/
```

Option B: Use your own audio samples
```bash
# Record audio samples and save as WAV files
# Name format: [emotion]_[number].wav (e.g., happy_1.wav)
# Place in data/train-custom/ and data/test-custom/
```

### 3. Generate CSV Files
```bash
python create_csv.py
```

### 4. Training the Basic Model
```bash
python emotion_recognition.py
```

### 5. Testing with Your Voice
```bash
python test.py
# Follow prompts to speak and get prediction
```

## Part 2: Setting Up LLM-Backend with Audio Integration

### 1. Clone and Setup the LLM-Backend

```bash
# Clone the repository
git clone https://github.com/Cerebrum-Ai/LLM-Backend.git
cd LLM-Backend

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Required Models

```bash
# For macOS, you might need to install wget first
brew install wget

# Download the medical language model
wget https://huggingface.co/mradermacher/Bio-Medical-MultiModal-Llama-3-8B-V1-GGUF/resolve/main/Bio-Medical-MultiModal-Llama-3-8B-V1.Q4_K_M.gguf

# Download supplementary model
wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q5_K_M.gguf
```

### 3. Set Up Environment Variables

Create a .env file in the LLM-Backend directory:
```
NGROK_AUTH_TOKENS=your_ngrok_auth_token
NODE_HANDLER_URL=http://your_node_handler_url
```

### 4. Integrate Emotion Recognition

Ensure the emotion-recognition-using-speech directory is accessible to the LLM-Backend. You can either:

A. Copy necessary files to LLM-Backend:
```bash
cp -r /path/to/emotion-recognition-using-speech/utils.py /path/to/LLM-Backend/
cp -r /path/to/emotion-recognition-using-speech/data_extractor.py /path/to/LLM-Backend/
```

B. Or add the emotion-recognition-using-speech directory to your Python path.

### 5. Running the LLM-Backend Server

```bash
python main.py
```

This will:
1. Initialize the LLM, Vector Database, and Audio Analyzer
2. Start the Flask server on port 5050
3. Create an ngrok tunnel for external access

## Part 3: Testing the Integrated System

### 1. Testing Audio Analysis Endpoint

Using curl:
```bash
# Test with a local audio file
curl -X POST -F "audio=@/path/to/your/audio_sample.wav" http://localhost:5050/api/analyze_audio
```

Example response:
```json
{
  "status": "success",
  "analysis": {
    "detected_emotion": "happy",
    "probabilities": {
      "angry": 0.05,
      "sad": 0.1,
      "neutral": 0.15,
      "happy": 0.6,
      "fear": 0.1
    }
  }
}
```

### 2. Testing the Chat Endpoint with Audio

```bash
# Test the full chat endpoint with audio
curl -X POST \
  -F "question=How are you feeling today?" \
  -F "audio=@/path/to/your/audio_sample.wav" \
  http://localhost:5050/api/chat
```

Example response:
```json
{
  "status": "success",
  "analysis": {
    "initial_diagnosis": "Based on your question and the audio analysis...",
    "vectordb_results": "...",
    "final_analysis": "Your voice indicates happiness with 60% confidence...",
    "audio_analysis": {
      "detected_emotion": "happy",
      "probabilities": {
        "angry": 0.05,
        "sad": 0.1,
        "neutral": 0.15,
        "happy": 0.6,
        "fear": 0.1
      }
    }
  }
}
```

### 3. Real-World Example: Medical Diagnosis with Image and Audio

```bash
# Test with medical query, image, and audio emotion analysis
curl -X POST \
  https://[your-ngrok-url].ngrok-free.app/chat \
  -F "question=I've been having headaches and skin issues" \
  -F "image=https://dermnetnz.org/assets/Uploads/site-age-specific/lowerleg9.jpg" \
  -F "audio=@data/training/Actor_16/03-02-05-01-01-02-16_angry.wav"
```

Example response:
```json
{
  "analysis": {
    "audio_analysis": {
      "detected_emotion": "angry",
      "probabilities": {
        "angry": 0.95,
        "fear": 0.0,
        "happy": 0.01,
        "neutral": 0.02,
        "sad": 0.02
      }
    },
    "final_analysis": "Diagnosis: Hyperthyroidism\nSymptoms: fatigue, weight loss, rapid heartbeat, sweating, anxiety, irritability, tremors\nTreatment: medications to reduce thyroid hormone production, radioactive iodine therapy, surgery\nEmotional State: angry\n",
    "initial_diagnosis": " headache, acne, depression, anxiety disorder.",
    "vectordb_results": "Hyperthyroidism,\"weight loss, rapid heartbeat, sweating, anxiety, irritability, tremors\",\"medications to reduce thyroid hormone production, radioactive iodine therapy, surgery\"\nHypothyroidism,\"fatigue, weight gain, cold intolerance, dry skin, constipation\",\"thyroid hormone replacement medication\"\nMigraine,\"severe throbbing headache, often on one side, nausea, vomiting, sensitivity to light and sound\",\"pain relievers, triptans, preventive medications, lifestyle management\"\nTension Headache,\"dull, aching head pain, tightness or pressure across the forehead or sides of the head\",\"pain relievers, stress management\"\nCluster Headache,\"severe, intense pain around one eye, often with tearing, nasal congestion, and restlessness\",\"oxygen therapy, triptans, preventive medications\""
  },
  "status": "success"
}
```

This example demonstrates how the system incorporates:
1. Text input analysis (patient's symptoms)
2. Image analysis (skin condition image)
3. Audio emotion detection (detecting "angry" emotional state)
4. Vector database retrieval (finding similar medical conditions)
5. Final integrated analysis with diagnosis and treatment recommendations

### 4. Web Interface Testing

If you have a frontend application connected to the LLM-Backend:

1. Access the application using the ngrok URL that was generated
2. Navigate to the chat interface
3. Record audio or upload an audio file
4. Submit a question along with the audio
5. Observe the response, which should include emotion analysis

## Part 4: Troubleshooting

### Audio Analysis Issues

1. **Problem**: Audio file not recognized
   **Solution**: Ensure the audio is in WAV format and properly encoded

2. **Problem**: "Failed to extract features" error
   **Solution**: Check that the audio file has sufficient duration (>1s) and appropriate volume

3. **Problem**: Emotion detection accuracy is low
   **Solution**: 
   - Add more training samples to data/train-custom/
   - Run `python emotion_recognition.py` to retrain the model
   - Try recording in a quieter environment

### LLM-Backend Server Issues

1. **Problem**: Server fails to start
   **Solution**: Check .env file configuration and ensure all dependencies are installed

2. **Problem**: Ngrok tunnel fails
   **Solution**: Verify your ngrok auth token is valid and not expired

3. **Problem**: "Model is still initializing" response
   **Solution**: Wait a few minutes for all components to initialize, especially if running on lower-end hardware

## Part 5: Advanced Configuration

### Customizing Emotions

To change the detected emotions, modify the `_emotions` list in `audio_processor.py`:

```python
_emotions = ["angry", "sad", "neutral", "happy", "fear", "excited", "calm"]
```

### Improving Model Accuracy

1. Increase training data samples
2. Adjust model parameters in `audio_processor.py`:
```python
model = RandomForestClassifier(n_estimators=200, max_depth=10)
```

3. Try a different model entirely:
```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
```

### Integrating with External Services

You can also post audio analysis results to external services:

```bash
curl -X POST \
  -F "audio=@/path/to/your/audio_sample.wav" \
  http://localhost:5050/api/analyze_audio | \
  curl -X POST \
  -H "Content-Type: application/json" \
  -d @- \
  https://your-external-service.com/api/endpoint
``` 