# Cerebrum AI LLM Backend

This repository contains the backend code for Cerebrum AI's LLM-powered multimodal health analysis system.

## System Components

The system consists of two main components:
1. **Main Application**: Handles LLM processing, vector database lookups, and coordinates multimodal inputs
2. **Models Service**: Provides audio emotion analysis functionality

## Prerequisites

- Python 3.9+ (3.13 recommended)
- Conda environment manager
- GGUF format LLM models
- Audio dataset for emotion recognition (if retraining models)

## Setup Instructions

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Cerebrum-Ai/LLM-Backend.git
   cd LLM-Backend
   ```

2. Create and activate conda environment:
   ```bash
   conda create -n aventus python=3.13
   conda activate aventus
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install llama-cpp-python
   ```

4. Set environment variables in `.env` file:
   ```
   NGROK_AUTH_TOKENS=your_ngrok_auth_token1,your_ngrok_auth_token2
   NODE_HANDLER_URL=https://your-node-handler.ngrok-free.app
   ML_MODELS_URL=http://localhost:9000
   ```

### Starting the System

1. Start the Models Service (for audio processing):
   ```bash
   conda activate aventus
   cd models
   python app.py
   ```
   This will start the Models Service on port 9000.

2. In a separate terminal, start the Main Application:
   ```bash
   conda activate aventus
   # Make sure you're in the project root directory
   python main.py
   ```
   This will:
   - Initialize the Vector Database
   - Load the LLM models
   - Start the Flask server on port 5050
   - Set up ngrok tunneling
   - Register with the node handler

## API Endpoints

### 1. Status Endpoints

Check the status of the Main Application:
```bash
curl http://localhost:5050/status
```

Check the status of the Models Service:
```bash
curl http://localhost:9000/status
```

### 2. Chat Endpoint

Send a text-only query:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What is the meaning of life?"}' \
  http://localhost:5050/api/chat
```

### 3. Audio Analysis

#### Using the Models Service directly:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"data_type": "audio", "model": "emotion", "url": "/path/to/audio.wav"}' \
  http://localhost:9000/ml/process
```

#### Using the Main Application:
```bash
curl -X POST -F "audio=@/path/to/audio.wav" \
  http://localhost:5050/api/analyze_audio
```

### 4. Multimodal Analysis

Send text, image, and audio in one request:
```bash
curl -X POST \
  -F "question=I have a skin problem and I feel angry about it" \
  -F "image=https://dermnetnz.org/assets/Uploads/site-age-specific/lowerleg9.jpg" \
  -F "audio=@/path/to/audio.wav" \
  http://localhost:5050/api/chat
```

## Using the ngrok URL

If you want to access the API remotely, you can use the ngrok URL that's generated when you start the Main Application. Replace `localhost:5050` with the ngrok URL in any of the commands above.

Example:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the meaning of life?"}' \
  https://your-ngrok-url.ngrok-free.app/api/chat
```

## Troubleshooting

1. **LLM Initialization Failure**: 
   - Ensure you have installed llama-cpp-python: `pip install llama-cpp-python`
   - Verify the model paths in `singleton.py`

2. **Audio Processing Errors**:
   - Make sure the Models Service is running on port 9000
   - Ensure the `ML_MODELS_URL` environment variable is set correctly
   - Check that audio files are in WAV format

3. **Ngrok Errors**:
   - Verify your ngrok auth tokens are valid
   - Ensure you're not exceeding ngrok connection limits

4. **Memory Issues**:
   - If you're getting "Insufficient Memory" errors, consider reducing model sizes or context length
   - Close other memory-intensive applications

5. **Missing Module Errors**:
   - Run `pip install -r requirements.txt` to ensure all dependencies are installed

## Additional Information

For detailed information about the codebase organization and architecture, please refer to the project documentation. 