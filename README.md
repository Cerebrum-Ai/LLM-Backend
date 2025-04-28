# Cerebrum AI LLM Backend

This repository contains the backend code for Cerebrum AI's LLM-powered multimodal health analysis system.

## System Components

The system consists of two main components:
1. **Main Application (main.py)**: Handles LLM processing, vector database lookups, and coordinates multimodal inputs
2. **Models Service (models.py)**: Provides machine learning capabilities including audio emotion analysis and keystroke pattern analysis

## Prerequisites

- Python 3.9+ (3.13 recommended)
- Conda environment manager (recommended)
- GGUF format LLM models (Bio-Medical-MultiModal-Llama-3-8B-V1.Q4_K_M.gguf)
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
   ```

4. Install the LLama-cpp-python library with specific optimizations:
   - For Mac with Metal support:
     ```bash
     CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
     ```
   - For Windows with CUDA support:
     ```bash
     CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_FORCE_CUBLAS=on -DLLAVA_BUILD=off -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
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

### 4. Memory Optimization

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

### 5. Starting the System

1. Start the Models Service:
   ```bash
   conda activate aventus
   # Make sure you're in the project root directory
   python models.py
   ```
   This will:
   - Initialize all ML models
   - Start the Flask server for ML models on port 9000
   - Set up ngrok tunneling for remote access
   - Register with the node handler

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

3. Both services will display their ngrok URLs when they start up, which you can use to access them remotely.

4. To run the services in the background:
   ```bash
   # Start ML models service in background
   python models.py &
   
   # Start main application in background
   python main.py &
   ```

## API Endpoints

### 1. Health Check Endpoints

Check the status of the Models Service:
```bash
curl https://your-models-service-ngrok-url.ngrok-free.app/
```

### 2. Chat Endpoint

Send a text-only query:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of diabetes?"}' \
  https://your-llm-service-ngrok-url.ngrok-free.app/api/chat
```

### 3. Audio Analysis

#### Using the Models Service directly:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"data_type": "audio", "model": "emotion", "url": "/path/to/audio.wav"}' \
  https://your-models-service-ngrok-url.ngrok-free.app/ml/process
```

#### Using the Main Application:
```bash
curl -X POST -F "audio=@/path/to/audio.wav" \
  https://your-llm-service-ngrok-url.ngrok-free.app/api/analyze_audio
```

### 4. Typing Analysis

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"keystrokes": [{"key": "a", "timestamp": 1680000000, "event": "keydown"}, {"key": "b", "timestamp": 1680000100, "event": "keydown"}, ...]}' \
  https://your-llm-service-ngrok-url.ngrok-free.app/api/analyze_typing
```

### 5. Multimodal Analysis

Send text, image, and audio in one request:
```bash
curl -X POST \
  -F "question=I have a skin problem and I feel angry about it" \
  -F "image=@/path/to/image.jpg" \
  -F "audio=@/path/to/audio.wav" \
  https://your-llm-service-ngrok-url.ngrok-free.app/api/chat
```

## Troubleshooting

### Common Issues

1. **LLM Initialization Failure**: 
   - Ensure you have installed llama-cpp-python with the correct optimizations
   - Verify the model paths in `singleton.py`
   - If you see `No module named 'llama_cpp'` error, reinstall with the correct CMAKE_ARGS

2. **Audio Processing Errors**:
   - Make sure the Models Service is running
   - Ensure the `ML_MODELS_URL` environment variable is set correctly
   - Check that audio files are in WAV format

3. **Ngrok Errors**:
   - Verify your ngrok auth tokens are valid
   - If you see "Ngrok session limit reached", either wait for sessions to expire or use another token
   - When switching ngrok tokens, restart the services

4. **"Connection refused" errors**:
   - Check that both services are running
   - Verify that the ML_MODELS_URL is correctly set in your .env file
   - Ensure the ports (5050 for main, 9000 for models) are not being used by other applications

5. **Vector DB Loading Errors**:
   - If Chroma DB fails to load, re-run the `setup.ipynb` notebook
   - Make sure the `chroma_langchain_db` directory has proper permissions

### Testing Endpoints

Use the included `test_endpoints.py` script to verify all endpoints are working:

```bash
python test_endpoints.py <llm-service-url> <ml-models-url>
```

For example:
```bash
python test_endpoints.py https://c061-104-28-219-95.ngrok-free.app https://4f76-104-28-219-94.ngrok-free.app
```

### Emergency Restart

If the system becomes unresponsive:

1. Kill the Python processes:
   ```bash
   pkill -f "python models.py"
   pkill -f "python main.py"
   ```

2. Check for any remaining processes:
   ```bash
   ps aux | grep python
   ```

3. Restart both services as described in the "Starting the System" section

## Additional Information

For detailed information about the codebase organization and architecture, please refer to the project documentation or contact the development team. 