# Cerebrum AI LLM Backend

This repository contains the backend code for Cerebrum AI's LLM-powered multimodal health analysis system.

## System Components

The system consists of two main components:
1. **Main Application**: Handles LLM processing, vector database lookups, and coordinates multimodal inputs
2. **Models Service**: Provides audio emotion analysis functionality

## Prerequisites

- Python 3.9+ (3.13 recommended)
- Conda environment manager
- GGUF format LLM models (Bio-Medical-MultiModal-Llama-3-8B-V1.Q4_K_M.gguf)
- Audio dataset for emotion recognition (if retraining models)
- Minimum 16GB RAM (32GB recommended for optimal performance)
- At least 10GB free disk space

## Comprehensive Setup Guide

### 1. Clone and Environment Setup

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

### 2. Model Setup

1. Download the required LLM models:
   - If the models are not already present, you'll need to download:
     - `Bio-Medical-MultiModal-Llama-3-8B-V1.Q4_K_M.gguf` (for medical analysis)
     - `phi-2.Q5_K_M.gguf` (for general text processing)
   - Place them in the root directory of the project

2. Initialize the vector database and LLM models using the setup notebook:
   ```bash
   jupyter notebook setup.ipynb
   ```

3. Inside the notebook:
   - Run all cells to set up the vector database
   - This will process the medical data, create embeddings, and store them in `medical_data_embeddings.pkl`
   - It will also verify that the LLM models are working correctly

### 3. Environment Variables

Create a `.env` file in the project root with the following variables:

```
NGROK_AUTH_TOKENS=your_ngrok_auth_token1,your_ngrok_auth_token2
NODE_HANDLER_URL=https://your-node-handler.ngrok-free.app
ML_MODELS_URL=http://localhost:9000
```

- If you need ngrok auth tokens, sign up at [ngrok.com](https://ngrok.com)
- The NODE_HANDLER_URL should be provided by your system administrator
- The ML_MODELS_URL should be set to wherever your Models Service will run (default is localhost:9000)

### 4. Audio Emotion Analysis Setup

The audio emotion analysis model is pre-trained and available in the repository as `models/audio/emotion/audio_emotion_model.pkl`. If you need to retrain it:

1. Prepare your audio data:
   - Place WAV files in `models/audio/emotion/data/` directory
   - Files should be organized by emotion (e.g., "happy", "sad", "angry", "neutral", "fear")

2. Run the training script:
   ```bash
   cd models/audio/emotion
   python convert_wavs.py  # Convert audio files to the right format if needed
   ```

3. Test the audio analyzer:
   ```bash
   python test_audio_analyzer.py
   ```

### 5. Memory Optimization

For optimal performance on machines with limited RAM:

1. Adjust model context size in `singleton.py`:
   ```python
   # Reduce n_ctx for lower memory usage (default is 2048)
   n_ctx=1024  # Reduces memory usage but may affect model performance
   ```

2. Close other memory-intensive applications before running the system

3. If you encounter `Insufficient Memory` errors, try:
   ```bash
   # Add these environment variables before running the application
   export GGML_METAL_PATH_RESOURCES=.
   export GGML_METAL_NDEBUG=1
   ```

### 6. Starting the System

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

3. Check that both services are running:
   ```bash
   # Check main app
   curl http://localhost:5050/status
   
   # Check models service
   curl http://localhost:9000/status
   ```

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

### 4. Image Analysis

```bash
curl -X POST \
  -F "question=What is happening to my skin?" \
  -F "image=https://dermnetnz.org/assets/Uploads/site-age-specific/lowerleg9.jpg" \
  http://localhost:5050/api/chat
```

### 5. Multimodal Analysis

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

### Common Issues

1. **LLM Initialization Failure**: 
   - Ensure you have installed llama-cpp-python: `pip install llama-cpp-python`
   - Verify the model paths in `singleton.py`
   - If you see `No module named 'llama_cpp'` error, reinstall with: `pip uninstall llama-cpp-python && pip install llama-cpp-python`

2. **Audio Processing Errors**:
   - Make sure the Models Service is running on port 9000
   - Ensure the `ML_MODELS_URL` environment variable is set correctly
   - Check that audio files are in WAV format
   - For errors like "Failed to extract features", ensure your WAV file has a valid sample rate (8000Hz-48000Hz)

3. **Ngrok Errors**:
   - Verify your ngrok auth tokens are valid
   - If you see "Ngrok session limit reached", either wait for sessions to expire or use another token
   - When switching ngrok tokens, restart the main application

4. **Memory Issues**:
   - If you see `Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)` errors:
     - Reduce the model context size in `singleton.py`
     - Close other applications that use GPU/RAM
     - Consider using a smaller model variant
   - If the application is slow, add more RAM or reduce the size of your embeddings database

5. **"python-dotenv could not parse statement" warnings**:
   - These are usually benign and can be ignored
   - If they bother you, check your .env file for proper formatting

6. **Vector DB Loading Errors**:
   - If Chroma DB fails to load, re-run the `setup.ipynb` notebook
   - Make sure the `chroma_langchain_db` directory has proper permissions

### Emergency Restart

If the system becomes unresponsive:

1. Kill all Python processes:
   ```bash
   pkill -f "python"
   ```

2. Check for any zombie ngrok processes:
   ```bash
   ps aux | grep ngrok
   ```

3. Kill any remaining ngrok processes:
   ```bash
   pkill -f "ngrok"
   ```

4. Restart both services as described in the "Starting the System" section

## Additional Information

For detailed information about the codebase organization and architecture, please refer to the project documentation or contact the development team. 