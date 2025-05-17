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
     - Image classification/analysis models for medical imaging
     - Diabetes prediction model for health risk assessment
     - ECG analysis model for heart condition detection

3. **Performance Monitoring System (monitor.py)**:

   - Monitors the health and performance of all services (LLM, ML models, node handler)
   - Tracks API endpoint response times and verifies correct payloads for each route
   - Monitors system resource usage
   - Provides a real-time dashboard
   - Logs issues and errors for troubleshooting
   - Automatically manages service startup/shutdown and triggers alerts for endpoint failures or threshold violations

4. **Workflow Models**:

   - **ER Triage Model (er_triage.py)**:

     - Patient assessment and prioritization system
     - Assigns triage levels (RESUSCITATION to NON-URGENT)
     - Analyzes vital signs and symptoms
     - Generates critical findings and recommendations
     - Integrates with alert system for urgent cases

   - **Lab Analysis Model (lab_analysis.py)**:

     - Comprehensive laboratory result analysis
     - Reference range validation for common tests
     - Trend analysis for sequential results
     - Critical value detection and alerting
     - Automated recommendations based on findings

   - **Alert System (alert_system.py)**:
     - Multi-channel alert management (console, email, Slack, pager)
     - Configurable alert levels and priorities
     - Rate limiting and cooldown periods
     - Alert history tracking and filtering
     - Integration with other workflow models

## Prerequisites

- Python 3.12.5 (3.12 recommended)
- Conda environment manager / venv environment manager (recommended)
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

2. Create and activate conda environment or using python virtual environment:

- Using conda:

```bash
conda create -n aventus python=3.11
conda activate aventus
```

- Using python virtual environment:

```bash
python -m venv aventus
source aventus/bin/activate  # On Windows use `aventus\Scripts\activate`
```

3. Install basic dependencies:

   ```bash
   pip install -r requirements_main.txt
   ```

4. Install the LLama-cpp-python library with specific optimizations:
   - For Mac with Metal support:
     ```bash
     CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
     ```
   - For Windows and Linux with CUDA support:
     ```bash
     CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_FORCE_CUBLAS=on -DLLAVA_BUILD=off -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
     ```

- For Linux / Windows without CUDA support:
  ```bash
  pip install llama-cpp-python
  ```

5. Install ngrok:
   ```bash
   pip install ngrok
   ```

### 2. Model Setup

1. Download the required LLM models and place them in the project root:

   - `Bio-Medical-MultiModal-Llama-3-8B-V1.Q4_K_M.gguf` (~4.8GB)
     'https://huggingface.co/TheBloke/Bio-Medical-MultiModal-Llama-3-8B-V1-Q4_K_M-GGUF/resolve/main/Bio-Medical-MultiModal-Llama-3-8B-V1.Q4_K_M.gguf'
   - `phi-2.Q5_K_M.gguf` (~2.2GB)
     'https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q5_K_M.gguf'
     You can download these from Hugging Face and then place in the folder created

2. Initialize the vector database:

   ```bash

   ```

   Run all cells to:

   - Process the medical data from `medical_data.csv`
   - Create embeddings stored in `medical_data_embeddings.pkl`
   - Verify LLM models are working correctly

## Example API Input/Output

### Example Request (JSON)

```
POST /api/external/chat
Content-Type: application/json

{
  "question": "What disease do I have",
  "image": "https://odgfmdbnjroqktddkgkz.supabase.co/storage/v1/object/public/test//test.png",
  "audio": "https://odgfmdbnjroqktddkgkz.supabase.co/storage/v1/object/public/test//output10.wav"
}
```

### Example Response (JSON)

```
{
  "analysis": {
    "audio_analysis": {
      "detected_emotion": "angry",
      "probabilities": {
        "angry": 0.85,
        "fear": 0.0,
        "happy": 0.0,
        "neutral": 0.0,
        "sad": 0.15
      }
    },
    "final_analysis": "\nDiagnosis: Coronary Artery Disease\nSymptoms: chest pain, shortness of breath, fatigue\nTreatment: medication, lifestyle changes\nEmotional State: angry\n",
    "image_analysis": {
      "breastmnist": {"predicted_label": "malignant", "probability": 0.9964368343353271},
      "chestmnist": {"predicted_label": "Lung Opacity", "probability": 0.2963375449180603},
      "dermamnist": {"predicted_label": "Melanoma", "probability": 0.4841499626636505},
      "octmnist": {"predicted_label": "DME", "probability": 0.3850083649158478},
      "pathmnist": {"predicted_label": "mucus", "probability": 0.9718669056892395},
      "pneumoniamnist": {"predicted_label": "pneumonia", "probability": 0.9292548298835754}
    },
    "initial_diagnosis": "1. Diabetes, 2. Hypertension, 3. Hyperlipidemia, 4. Obesity, 5. Coronary Artery Disease",
    "vectordb_results": "Coronary Atherosclerosis,\"Chest pain or discomfort (angina), shortness of breath, fatigue, irregular heartbeat, dizziness, nausea, sweating, jaw or arm pain (in some cases), heart attack (when a coronary artery becomes completely blocked)\",\"Lifestyle changes (such as a heart-healthy diet, regular exercise, weight management, smoking cessation), medication (such as aspirin, cholesterol-lowering medications, beta-blockers, calcium channel blockers, nitroglycerin), procedures (such as angioplasty and stenting, coronary artery bypass grafting), cardiac rehabilitation, management of other risk factors (such as high blood pressure, diabetes)\""
  },
  "status": "success"
}
```

### Diabetes Prediction API

The system includes a diabetes prediction model that can assess diabetes risk based on various health parameters.

#### Example Request

```bash
curl -X POST http://localhost:9000/ml/process \
-H "Content-Type: application/json" \
-d '{
  "data_type": "health",
  "model": "diabetes",
  "url": {
    "gender": "Male",
    "age": 45,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 25.5,
    "HbA1c_level": 5.7,
    "blood_glucose_level": 120
  }
}'
```

#### Required Parameters

- `gender`: String ("Male" or "Female")
- `age`: Integer (age in years)
- `hypertension`: Integer (0 or 1)
- `heart_disease`: Integer (0 or 1)
- `smoking_history`: String (e.g., "never", "former", "current")
- `bmi`: Float (Body Mass Index)
- `HbA1c_level`: Float (Glycated hemoglobin level)
- `blood_glucose_level`: Integer (Blood glucose level in mg/dL)

#### Example Response

```json
{
  "prediction": 0,
  "probability": 0.15,
  "risk_level": "Low"
}
```

The response includes:

- `prediction`: Binary outcome (0 for no diabetes, 1 for diabetes)
- `probability`: Probability of diabetes (0 to 1)
- `risk_level`: Categorical risk assessment ("Low", "Medium", "High")

### 3. Environment Variables

Create a `.env` file in the project root with the following variables:

```
# Ngrok auth tokens for creating public tunnel URLs
NGROK_AUTH_TOKENS=your_ngrok_auth_token1,your_ngrok_auth_token2,....
NGROK_AUTH_TOKENS=your_ngrok_auth_token_for_node_handler
# URL of the node handler service
NODE_HANDLER_URL=https://your-node-handler.ngrok-free.app
NGROK_HANDLER_URL=your-node-handler.ngrok-free.app # same ad NODE_HANDLER_URL just without https
# URL of the ML Models service
ML_MODELS_URL=http://localhost:9000



# URL for supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_BUCKET=your_supabase_bucket #for storing images for processing
SUPABASE_ADMIN_KEY=your_supabase_admin_key

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
   cd runner
   python monitor.py --auto-start #might show typing_service is failing but its redundant
   ```

2. To stop all services:
   Close monitor.py and run
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
3. In a third terminal, start the Node Handler Service:
   ```bash
   conda activate aventus
   python node_handler.py
   ```
4. In a fourth terminal, start the Chatbot Service:
   ```bash
   conda activate aventus
   python initial_chatbot.py
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

### 4. ECG Analysis Model

The ECG analysis model processes electrocardiogram data to detect various heart conditions:

- **Implementation**: `ECGAnalyzer` class in `models/health/ecg/ecg_analyzer.py`
- **Features Analyzed**:
  - Heart rate and rhythm
  - P-wave morphology
  - QRS complex characteristics
  - ST segment changes
  - T-wave abnormalities
- **Conditions Detected**:
  - Normal sinus rhythm
  - Atrial fibrillation
  - Ventricular tachycardia
  - Myocardial infarction
  - Bundle branch blocks
- **Input Format**:
  - Raw ECG signal data (numpy array)
  - Standard 12-lead ECG recordings
  - Single-lead ECG data
- **Output Format**:
  ```json
  {
    "prediction": "atrial_fibrillation",
    "probability": 0.85,
    "confidence": 0.92,
    "rhythm_analysis": {
      "heart_rate": 85,
      "rhythm_type": "irregular",
      "p_wave_present": false
    },
    "waveform_analysis": {
      "qrs_duration": 0.08,
      "st_elevation": 0.0,
      "t_wave_inversion": false
    }
  }
  ```

## API Reference

### Main LLM Service Endpoints

#### 1. Chat Endpoint for LLM

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

Generic endpoint for all machine learning models (audio, typing, and soon image).

**Endpoint**: `/ml/process`  
**Method**: POST  
**Content-Type**: `application/json`

**Request Format**:

```json
{
  "url": "path_to_data_or_json_string_or_base64",
  "data_type": "audio|typing|image",
  "model": "emotion|pattern|classification"
}
```

- For audio: `url` can be a file path or downloadable URL.
- For typing: `url` is a JSON string or object with `{"keystrokes": [...]}`.
- For image: `url` will be a URL, file path, or base64 string (see upcoming image model section).

**Response Format** (audio/emotion):

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

**Response Format** (typing/pattern):

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

**Response Format** (image/classification, coming soon):

```json
{
  "detected_condition": "covid|pneumonia|normal|other",
  "probabilities": {
    "covid": 0.12,
    "pneumonia": 0.08,
    "normal": 0.75,
    "other": 0.05
  },
  "features": {
    "area": 12345.6,
    "confidence": 0.91
  }
}
```

**Example Curl Command (image, coming soon):**

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.png", "data_type": "image", "model": "classification"}' \
  https://your-models-service-ngrok-url.ngrok-free.app/ml/process
```

### Workflow Models API Reference

#### 1. ER Triage Endpoint

Processes patient data for emergency room triage assessment.

**Endpoint**: `/api/er/triage`  
**Method**: POST  
**Content-Type**: `application/json`

**Request Format**:

```json
{
  "vitals": {
    "heart_rate": 120,
    "blood_pressure_systolic": 140,
    "blood_pressure_diastolic": 90,
    "oxygen_saturation": 95,
    "temperature": 37.2
  },
  "symptoms": ["chest pain", "shortness of breath", "dizziness"],
  "medical_history": {
    "previous_conditions": ["hypertension", "diabetes"],
    "medications": ["aspirin", "metformin"],
    "allergies": ["penicillin"]
  }
}
```

**Response Format**:

```json
{
  "triage_level": "URGENT",
  "severity_score": 6.5,
  "recommendations": [
    "Immediate medical attention required",
    "Prepare ECG and cardiac monitoring",
    "Monitor vital signs every 15 minutes"
  ],
  "timestamp": "2024-03-14T10:30:00Z",
  "critical_findings": ["Possible acute coronary syndrome"]
}
```

**Example Curl Command**:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "vitals": {
      "heart_rate": 120,
      "blood_pressure_systolic": 140,
      "oxygen_saturation": 95
    },
    "symptoms": ["chest pain", "shortness of breath"]
  }' \
  https://your-llm-service-ngrok-url.ngrok-free.app/api/er/triage
```

#### 2. Lab Analysis Endpoint

Analyzes laboratory test results and provides interpretation.

**Endpoint**: `/api/lab/analyze`  
**Method**: POST  
**Content-Type**: `application/json`

**Request Format**:

```json
{
  "results": {
    "WBC": 11.5,
    "RBC": 4.8,
    "HGB": 14.2,
    "PLT": 250,
    "NA": 140,
    "K": 4.0,
    "GLU": 95,
    "TROP": 0.02,
    "BNP": 80
  },
  "previous_results": {
    "WBC": [
      { "value": 10.2, "timestamp": "2024-03-01T10:00:00Z" },
      { "value": 9.8, "timestamp": "2024-02-15T10:00:00Z" }
    ],
    "HGB": [
      { "value": 13.8, "timestamp": "2024-03-01T10:00:00Z" },
      { "value": 13.5, "timestamp": "2024-02-15T10:00:00Z" }
    ]
  }
}
```

**Response Format**:

```json
{
  "current_analysis": {
    "WBC": {
      "value": 11.5,
      "unit": "10^9/L",
      "reference_range": "4.5-11.0",
      "status": "HIGH"
    },
    "HGB": {
      "value": 14.2,
      "unit": "g/dL",
      "reference_range": "13.5-17.5",
      "status": "NORMAL"
    }
  },
  "trend_analysis": {
    "WBC": {
      "current_value": 11.5,
      "previous_value": 10.2,
      "trend": "INCREASING",
      "percent_change": 12.7
    }
  },
  "critical_findings": [],
  "recommendations": [
    "Consider evaluation for possible infection due to elevated WBC"
  ],
  "timestamp": "2024-03-14T10:30:00Z"
}
```

**Example Curl Command**:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "results": {
      "WBC": 11.5,
      "HGB": 14.2,
      "TROP": 0.02
    }
  }' \
  https://your-llm-service-ngrok-url.ngrok-free.app/api/lab/analyze
```

#### 3. Alerts Endpoint

Retrieves and manages system alerts.

**Endpoint**: `/api/alerts`  
**Method**: GET  
**Query Parameters**:

- `level`: Filter by alert level (CRITICAL, HIGH, MEDIUM, LOW)
- `source`: Filter by alert source
- `start_time`: Filter by start time (ISO format)
- `end_time`: Filter by end time (ISO format)

**Response Format**:

```json
{
  "alerts": [
    {
      "message": "Severe tachycardia detected",
      "level": "CRITICAL",
      "source": "ER_TRIAGE",
      "timestamp": "2024-03-14T10:30:00Z",
      "data": {
        "vitals": {
          "heart_rate": 150
        }
      }
    }
  ],
  "count": 1
}
```

**Example Curl Command**:

```bash
curl "https://your-llm-service-ngrok-url.ngrok-free.app/api/alerts?level=CRITICAL&source=ER_TRIAGE"
```

#### 4. Clear Alerts Endpoint

Clears the alert history.

**Endpoint**: `/api/alerts/clear`  
**Method**: POST

**Response Format**:

```json
{
  "message": "Alert history cleared"
}
```

**Example Curl Command**:

```bash
curl -X POST https://your-llm-service-ngrok-url.ngrok-free.app/api/alerts/clear
```

### 3. Direct ML Call (ML Forwarding)

Forwards ML requests (audio, typing, image) to the ML node.

- **Endpoint:** `/ml/forward`
- **Method:** POST
- **Content-Type:** `application/json`
- **Request Example:**
  ```json
  {
    "url": "https://example.com/image.png",
    "data_type": "image",
    "model": "classification"
  }
  ```
- **Curl Example:**
  ```bash
  curl -X POST -H "Content-Type: application/json" \
    -d '{"url": "https://example.com/image.png", "data_type": "image", "model": "classification"}' \
    https://your-node-handler.ngrok-free.app/ml/forward
  ```

### 4. Workflow Models Forwarding

Forwards requests to various workflow models (ER Triage, Lab Analysis, ECG Analysis, Diabetes Prediction).

- **Endpoint:** `/workflow/forward`
- **Method:** POST
- **Content-Type:** `application/json`
- **Request Format:**
  ```json
  {
    "model": "er_triage|lab_analysis|ecg_analysis|diabetes",
    "data": {
      // Model-specific data (see examples below)
    }
  }
  ```

#### ER Triage Example:

```json
{
  "model": "er_triage",
  "data": {
    "vitals": {
      "heart_rate": 120,
      "blood_pressure_systolic": 140,
      "oxygen_saturation": 95
    },
    "symptoms": ["chest pain", "shortness of breath"]
  }
}
```

#### Lab Analysis Example:

```json
{
  "model": "lab_analysis",
  "data": {
    "results": {
      "WBC": 11.5,
      "HGB": 14.2,
      "TROP": 0.02
    }
  }
}
```

#### ECG Analysis Example:

```json
{
  "model": "ecg_analysis",
  "data": {
    "ecg_signal": "base64_encoded_ecg_data",
    "sampling_rate": 500,
    "leads": ["II", "V1", "V2", "V3", "V4", "V5", "V6"]
  }
}
```

#### Diabetes Prediction Example:

```json
{
  "model": "diabetes",
  "data": {
    "gender": "Male",
    "age": 45,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 25.5,
    "HbA1c_level": 5.7,
    "blood_glucose_level": 120
  }
}
```

- **Curl Example (ER Triage):**

  ```bash
  curl -X POST \
    -H "Content-Type: application/json" \
    -d '{
      "model": "er_triage",
      "data": {
        "vitals": {
          "heart_rate": 120,
          "blood_pressure_systolic": 140,
          "oxygen_saturation": 95
        },
        "symptoms": ["chest pain", "shortness of breath"]
      }
    }' \
    https://your-node-handler.ngrok-free.app/workflow/forward
  ```

- **Response Format:**
  ```json
  {
    "status": "success",
    "model": "er_triage",
    "result": {
      // Model-specific response (same as direct API calls)
    }
  }
  ```

### 5. Status Call (Node Handler Health)

**Endpoint**: `/status`  
**Method**: GET

**Response Format**:

```json
{
  "status": "ok",
  "message": "Node handler service is running"
}
```

**Example Curl Command**:

```bash
curl https://your-node-handler.ngrok-free.app/status
```

## Testing

### Automatic Testing

We've included a comprehensive test script that verifies all endpoints:

```bash
python test_endpoints.py
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
pkill -f "python node_handler.py"
pkill -f "python initial_chatbot.py"
```

## File Structure

```
LLM-Backend/
├── main.py                   # Main LLM service
├── models.py                 # ML Models service
├── singleton.py              # LLM initialization code
├── requirements.txt          # Core dependencies
├── requirements_main.txt     # Main service dependencies
├── test_endpoints.py         # Endpoint testing script
├── start_services.sh         # Service starter script
├── stop_services.sh          # Service stopper script
├── monitor.py                # Performance monitoring system
├── env_example               # Example .env file
├── runner/                   # Scripts and service management
│   ├── monitor_wrapper.sh    # Monitor management script
│   ├── start_services.sh     # Service starter script
│   ├── stop_services.sh      # Service stopper script
│   └── alert_config.json     # Alert configuration for monitor
├── models/                   # ML model implementations (Python package)
│   ├── __init__.py           # Package marker
│   ├── audio/                # Audio analysis models
│   │   ├── __init__.py
│   │   └── emotion/          # Emotion detection
│   │       ├── __init__.py
│   │       └── audio_utils.py
│   ├── typing/               # Typing analysis models
│   │   ├── __init__.py
│   │   └── pattern/          # Pattern detection
│   │       ├── __init__.py
│   │       └── typing_processor.py
│   └── image/                # [In progress] Image analysis models
│       ├── __init__.py
│       └── classification/   # [Planned] Image classification
│           └── __init__.py
├── Bio-Medical-MultiModal-Llama-3-8B-V1.Q4_K_M.gguf  # Medical LLM model
└── phi-2.Q5_K_M.gguf         # General purpose LLM model

```

## Performance Monitoring

The system includes a comprehensive performance monitoring tool (`monitor.py` (inside runner)) that provides real-time visibility into system health, API endpoint performance, and resource usage.

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
