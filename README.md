# Cerebrum AI LLM Backend

This repository contains the backend code for Cerebrum AI's LLM-powered multimodal health analysis system, which integrates large language models with specialized machine learning models for health assessment.

## System Overview

The system consists of two main components that work together:

1. **Main LLM Service (main.py)**: 
   - Handles large language model processing
   - Provides vector database lookups for medical knowledge
   - Coordinates multimodal inputs (text, audio, typing patterns)
   - Routes ML analysis requests to the Models Service
   - Provides a comprehensive API for health analysis

2. **ML Models Service (models.py)**:
   - Provides specialized machine learning capabilities
   - Currently includes:
     - Audio emotion analysis model
     - Keystroke pattern analysis for neurological screening

3. **Performance Monitoring System (monitor.py)**:
   - Monitors the health and performance of all services
   - Tracks API endpoint response times
   - Monitors system resource usage
   - Provides a real-time dashboard
   - Logs issues and errors for troubleshooting

## Prerequisites

- Python 3.9+ (3.13 recommended)
- Conda environment manager (recommended)
- GGUF format LLM models:
  - `Bio-Medical-MultiModal-Llama-3-8B-V1.Q4_K_M.gguf` (for medical analysis)
  - `phi-2.Q5_K_M.gguf` (for general text processing)
- Minimum 16GB RAM (32GB recommended for optimal performance)
- At least 10GB free disk space
- FFmpeg installed (for audio processing)

## Complete Setup Guide

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

3. Install basic dependencies:
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

5. Install ngrok:
   ```bash
   pip install ngrok
   ```

### 2. Model Setup

1. Download the required LLM models and place them in the project root:
   - `Bio-Medical-MultiModal-Llama-3-8B-V1.Q4_K_M.gguf` (~7.4GB)
   - `phi-2.Q5_K_M.gguf` (~2.2GB)

   You can download these from Hugging Face or other model repositories.

2. Initialize the vector database:
   ```bash
   jupyter notebook setup.ipynb
   ```
   Run all cells to:
   - Process the medical data from `medical_data.csv`
   - Create embeddings stored in `medical_data_embeddings.pkl`
   - Verify LLM models are working correctly

### 3. Environment Variables

Create a `.env` file in the project root with the following variables:

```
# Ngrok auth tokens for creating public tunnel URLs
NGROK_AUTH_TOKENS=your_ngrok_auth_token1,your_ngrok_auth_token2

# URL of the node handler service
NODE_HANDLER_URL=https://your-node-handler.ngrok-free.app

# URL of the ML Models service
ML_MODELS_URL=http://localhost:9000

# Optional: Port for the Main LLM service (default is 5050)
MAIN_PORT=5050
```

You can use the `env_example` file as a template:
```bash
cp env_example .env
```

Then edit the `.env` file to add your specific tokens and URLs.

### 4. Starting the System

We've included convenience scripts to start and stop the services:

1. Make the scripts executable:
   ```bash
   chmod +x start_services.sh stop_services.sh
   ```

2. Start both services with a single command:
   ```bash
   ./start_services.sh
   ```

   This will:
   - Start the ML Models Service in the background
   - Wait for it to initialize
   - Start the Main LLM Service in the background
   - Display the ngrok URLs for both services

3. To stop all services:
   ```bash
   ./stop_services.sh
   ```

### 5. Alternative: Manual Startup

If you prefer to start the services manually:

1. Start the ML Models Service:
   ```bash
   conda activate aventus
   python models.py
   ```

2. In a separate terminal, start the Main LLM Service:
   ```bash
   conda activate aventus
   python main.py
   ```

## Detailed Model Documentation

### 1. Large Language Models

The system uses two primary LLM models:

1. **Bio-Medical-MultiModal-Llama-3-8B-V1**: 
   - Specialized for medical knowledge and diagnostics
   - Used for the primary health analysis
   - Processes both text and image inputs

2. **Phi-2**:
   - Used for general text processing
   - Handles initial query understanding and routing

### 2. Audio Emotion Analysis Model

The audio emotion model analyzes voice recordings to detect emotional states:

- **Implementation**: `SimpleAudioAnalyzer` class in `models/audio/emotion/audio_analyzer.py`
- **Model File**: `audio_emotion_model.pkl`
- **Features Analyzed**:
  - Mel-frequency cepstral coefficients (MFCCs)
  - Spectral features
  - Tonal features
  - Rhythm patterns
- **Emotions Detected**:
  - Happy
  - Sad
  - Angry
  - Neutral
  - Fear
- **Input Format**: WAV audio files 

### 3. Keystroke Pattern Analysis Model

The keystroke pattern model analyzes typing patterns to screen for potential neurological conditions:

- **Implementation**: `KeystrokeProcessor` class in `models/typing/pattern/typing_processor.py`
- **Features Analyzed**:
  - Key press duration
  - Time between keystrokes
  - Error rates and corrections
  - Typing rhythm patterns
  - Typing speed
  - Pause patterns
- **Conditions Screened**:
  - Parkinson's disease
  - Essential tremor
  - Carpal tunnel syndrome
  - Multiple sclerosis
  - Normal typing patterns
- **Requirements**: Minimum 10 keystrokes for analysis
- **Input Format**: JSON keystroke data with timestamps

## API Reference

### Main LLM Service Endpoints

#### 1. Chat Endpoint

Processes natural language queries with optional multimodal inputs.

**Endpoint**: `/api/chat`  
**Method**: POST  
**Content-Type**: `application/json` or `multipart/form-data`

**JSON Request Format**:
```json
{
  "question": "What are the symptoms of diabetes?",
  "session_id": "optional_session_id",
  "image": "optional_base64_encoded_image",
  "audio": "optional_base64_encoded_audio",
  "typing": {
    "keystrokes": [
      {"key": "a", "timeDown": 1620136589000, "timeUp": 1620136589080},
      {"key": "b", "timeDown": 1620136589200, "timeUp": 1620136589270},
      ...
    ]
  }
}
```

**Form Data Request Format**:
```
question: What are the symptoms of diabetes?
image: [file upload]
audio: [file upload]
```

**Response Format**:
```json
{
  "status": "success",
  "analysis": {
    "initial_diagnosis": "Initial assessment based on query",
    "vectordb_results": "Relevant medical data from knowledge base",
    "final_analysis": "Comprehensive assessment including symptoms, treatments, and multimodal insights"
  }
}
```

**Example Curl Command**:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of diabetes?"}' \
  https://your-llm-service-ngrok-url.ngrok-free.app/api/chat
```

#### 2. Audio Analysis Endpoint

Analyzes audio for emotional content.

**Endpoint**: `/api/analyze_audio`  
**Method**: POST  

**Form Data Request Format**:
```
audio: [file upload]
```

**Response Format**:
```json
{
  "status": "success",
  "analysis": {
    "detected_emotion": "happy|sad|angry|neutral|fear",
    "confidence": 0.85,
    "emotions": {
      "happy": 0.85,
      "sad": 0.05,
      "angry": 0.03,
      "neutral": 0.05,
      "fear": 0.02
    }
  }
}
```

**Example Curl Command**:
```bash
curl -X POST \
  -F "audio=@/path/to/audio.wav" \
  https://your-llm-service-ngrok-url.ngrok-free.app/api/analyze_audio
```

#### 3. Typing Analysis Endpoint

Analyzes keystroke patterns for potential neurological conditions.

**Endpoint**: `/api/analyze_typing`  
**Method**: POST  
**Content-Type**: `application/json`

**Request Format**:
```json
{
  "keystrokes": [
    {"key": "a", "timeDown": 1620136589000, "timeUp": 1620136589080},
    {"key": "b", "timeDown": 1620136589200, "timeUp": 1620136589270},
    ...
  ]
}
```

**Response Format**:
```json
{
  "status": "success",
  "analysis": {
    "detected_condition": "parkinsons|essential_tremor|carpal_tunnel|multiple_sclerosis|normal",
    "probabilities": {
      "parkinsons": 0.15,
      "essential_tremor": 0.05,
      "carpal_tunnel": 0.75,
      "multiple_sclerosis": 0.03,
      "normal": 0.02
    },
    "features": {
      "key_press_duration": 130.5,
      "duration_variability": 25.3,
      "time_between_keys": 200.7,
      "rhythm_variability": 50.2,
      "error_rate": 0.03,
      "typing_speed": 240.5,
      "rhythm_consistency": 0.85,
      "pause_frequency": 4.2
    }
  }
}
```

**Example Curl Command**:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"keystrokes": [{"key": "a", "timeDown": 1620136589000, "timeUp": 1620136589080}, {"key": "b", "timeDown": 1620136589200, "timeUp": 1620136589270}]}' \
  https://your-llm-service-ngrok-url.ngrok-free.app/api/analyze_typing
```

### ML Models Service Endpoints

#### 1. Health Check Endpoint

**Endpoint**: `/`  
**Method**: GET  

**Response Format**:
```json
{
  "status": "ok",
  "service": "ml_models",
  "message": "ML models service is running"
}
```

**Example Curl Command**:
```bash
curl https://your-models-service-ngrok-url.ngrok-free.app/
```

#### 2. ML Process Endpoint

Generic endpoint for all machine learning models.

**Endpoint**: `/ml/process`  
**Method**: POST  
**Content-Type**: `application/json`

**Request Format**:
```json
{
  "url": "path_to_data_or_json_string",
  "data_type": "audio|typing",
  "model": "emotion|pattern"
}
```

**Response Format** (for audio/emotion):
```json
{
  "detected_emotion": "happy|sad|angry|neutral|fear",
  "confidence": 0.85,
  "emotions": {
    "happy": 0.85,
    "sad": 0.05,
    "angry": 0.03,
    "neutral": 0.05,
    "fear": 0.02
  }
}
```

**Response Format** (for typing/pattern):
```json
{
  "detected_condition": "parkinsons|essential_tremor|carpal_tunnel|multiple_sclerosis|normal",
  "probabilities": {
    "parkinsons": 0.15,
    "essential_tremor": 0.05,
    "carpal_tunnel": 0.75,
    "multiple_sclerosis": 0.03,
    "normal": 0.02
  },
  "features": {
    "key_press_duration": 130.5,
    "typing_speed": 240.5,
    "rhythm_consistency": 0.85
  }
}
```

**Example Curl Commands**:

For audio analysis:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"url": "/path/to/audio.wav", "data_type": "audio", "model": "emotion"}' \
  https://your-models-service-ngrok-url.ngrok-free.app/ml/process
```

For typing pattern analysis:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"url": "{\"keystrokes\": [{\"key\": \"a\", \"timeDown\": 1620136589000, \"timeUp\": 1620136589080}, {\"key\": \"b\", \"timeDown\": 1620136589200, \"timeUp\": 1620136589270}]}", "data_type": "typing", "model": "pattern"}' \
  https://your-models-service-ngrok-url.ngrok-free.app/ml/process
```

## Testing

### Automatic Testing

We've included a comprehensive test script that verifies all endpoints:

```bash
python test_endpoints.py <llm-service-url> <ml-models-url>
```

Example:
```bash
python test_endpoints.py https://c061-104-28-219-95.ngrok-free.app https://4f76-104-28-219-94.ngrok-free.app
```

This will:
- Test the health of both services
- Test the chat endpoint with a medical query
- Test the typing analysis endpoint
- Test the ML models process endpoint directly

### Manual Testing

You can also manually test each endpoint using the curl commands provided in the API Reference section.

## Memory Management

For systems with limited RAM:

1. Adjust model context size in `singleton.py`:
   ```python
   n_ctx=1024  # Reduces memory usage but may affect model performance
   ```

2. Set environment variables before running:
   ```bash
   export GGML_METAL_PATH_RESOURCES=.
   export GGML_METAL_NDEBUG=1
   ```

## Troubleshooting

### Common Issues

1. **LLM Initialization Failure**: 
   - Ensure you have installed llama-cpp-python with the correct optimizations
   - Verify the model paths in `singleton.py`
   - Check that models are properly downloaded

2. **Audio Processing Errors**:
   - Make sure FFmpeg is installed: `brew install ffmpeg` (Mac) or `apt install ffmpeg` (Linux)
   - Ensure audio files are in WAV format with valid sample rates

3. **"Connection refused" errors**:
   - Verify both services are running
   - Check ML_MODELS_URL in .env matches the actual port (default: 9000)
   - Ensure ports aren't being used by other applications

4. **Ngrok Issues**:
   - Verify ngrok auth tokens are valid
   - Install ngrok if missing: `pip install ngrok`
   - Check for multiple tokens in .env if you see "session limit reached" errors

5. **Vector DB Issues**:
   - Re-run `setup.ipynb` notebook cells if database doesn't load
   - Check permissions on the `chroma_langchain_db` directory

### Logs and Debugging

The services output logs to the console. To capture logs in files:

```bash
python models.py > models.log 2>&1 &
python main.py > main.log 2>&1 &
```

### Emergency Restart

If services become unresponsive:

```bash
./stop_services.sh
./start_services.sh
```

Or manually:

```bash
pkill -f "python models.py"
pkill -f "python main.py"
```

## File Structure

```
LLM-Backend/
├── main.py                   # Main LLM service
├── models.py                 # ML Models service
├── singleton.py              # LLM initialization code
├── utils.py                  # Utility functions
├── requirements.txt          # Core dependencies
├── test_endpoints.py         # Endpoint testing script
├── start_services.sh         # Service starter script
├── stop_services.sh          # Service stopper script
├── monitor.py                # Performance monitoring system
├── env_example               # Example .env file
├── models/                   # ML model implementations
│   ├── audio/                # Audio analysis models
│   │   └── emotion/          # Emotion detection
│   └── typing/               # Typing analysis models
│       └── pattern/          # Pattern detection
├── Bio-Medical-MultiModal-Llama-3-8B-V1.Q4_K_M.gguf  # Medical LLM model
└── phi-2.Q5_K_M.gguf         # General purpose LLM model
```

## Performance Monitoring

The system includes a comprehensive performance monitoring tool (`monitor.py`) that provides real-time visibility into system health, API endpoint performance, and resource usage.

### Features

- **Real-time Dashboard**: Visual display of service status and performance metrics
- **API Endpoint Monitoring**: Tests all endpoints periodically and reports response times
- **Service Health Checks**: Verifies both LLM and ML services are running properly
- **Resource Tracking**: Monitors CPU and memory usage
- **Process Monitoring**: Checks if all required processes are running
- **Error Logging**: Records errors for troubleshooting

### Running the Monitor

1. Make sure the monitoring script is executable:
   ```bash
   chmod +x monitor.py
   chmod +x monitor_wrapper.sh
   ```

2. Start the monitor:
   
   **Interactive Mode** (shows real-time dashboard):
   ```bash
   ./monitor.py
   ```
   
   **Background Mode** (runs in the background):
   ```bash
   ./monitor_wrapper.sh start
   ```

3. Managing the monitor in background mode:
   ```bash
   # Check monitor status
   ./monitor_wrapper.sh status
   
   # Stop the monitor
   ./monitor_wrapper.sh stop
   
   # Restart the monitor
   ./monitor_wrapper.sh restart
   ```

4. Optional command-line arguments:
   ```bash
   # Monitor with a custom interval (in seconds)
   ./monitor.py --interval=30
   # or in background mode:
   ./monitor_wrapper.sh start --interval=30
   
   # Monitor specific service URLs
   ./monitor.py --llm-url=https://your-llm-service-url.ngrok-free.app --ml-url=https://your-ml-models-url.ngrok-free.app
   
   # Save logs to a file
   ./monitor.py --log-file=monitor.log
   ```

5. Logs for the background monitor are automatically saved to the `logs/` directory.

6. To exit the interactive monitor, press `Ctrl+C`.

### Interpreting the Dashboard

The dashboard shows:

- **Services Status**: Online/offline status of each service with response times
- **System Resources**: CPU and memory usage percentages
- **Processes**: Status of critical processes (main.py and models.py)
- **LLM Endpoints**: Status and performance of all LLM service endpoints
- **ML Endpoints**: Status and performance of all ML Models service endpoints

Red indicators show problems that need attention, while green indicates healthy operation.

### Alert Configuration

The monitoring system includes a configurable alerting system that can notify you when services are unhealthy or performance thresholds are exceeded.

1. Configure alerts by editing `alert_config.json`:
   ```json
   {
     "alerts": {
       "service_down": {
         "enabled": true,
         "threshold": 3,
         "cooldown_minutes": 10
       },
       "high_latency": {
         "enabled": true,
         "threshold_ms": 2000,
         "consecutive_violations": 3
       }
       // ... more alert types
     },
     "notification_channels": {
       "console": { "enabled": true },
       "log_file": { 
         "enabled": true,
         "path": "logs/alerts.log"
       },
       "email": { "enabled": false },
       "slack": { "enabled": false }
     }
   }
   ```

2. Available alert types:
   - **service_down**: Triggers when a service goes down
   - **high_latency**: Triggers when endpoint response times exceed threshold
   - **resource_usage**: Triggers when CPU or memory usage exceeds threshold
   - **process_missing**: Triggers when a critical process is not running

3. Available notification channels:
   - **console**: Displays alerts in the console
   - **log_file**: Writes alerts to a log file
   - **email**: Sends email notifications (requires configuration)
   - **slack**: Sends Slack notifications (requires webhook URL)

4. To enable email or Slack notifications, update their configurations in `alert_config.json` and set `enabled` to `true`.

## Contributing

Please refer to the development team for contribution guidelines.

## License

Proprietary - All rights reserved. 