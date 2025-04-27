# Model Download Instructions

This document explains how to download the necessary model files for the application.

## Required Model Files

1. **LLM Models**
   - `phi-2.Q5_K_M.gguf` (2.2GB)
   - `Bio-Medical-MultiModal-Llama-3-8B-V1.Q4_K_M.gguf` (7.4GB)

2. **Other Model Files**
   - `audio_emotion_model.pkl` (Generated automatically on first run)
   - `medical_llm.pkl`
   - `multimodal_llm.pkl`
   - `medical_data_documents.pkl`
   - `medical_data_embeddings.pkl`

## Download Instructions

### For LLM Models

1. **Phi-2 Model**:
   ```
   wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q5_K_M.gguf
   ```

2. **Bio-Medical-MultiModal-Llama Model**:
   ```
   wget https://huggingface.co/TheBloke/Bio-Medical-MultiModal-Llama-3-8B-V1-GGUF/resolve/main/Bio-Medical-MultiModal-Llama-3-8B-V1.Q4_K_M.gguf
   ```

### For PKL Files

Some PKL files are generated when you first run the application:

1. **Audio Emotion Model** (`audio_emotion_model.pkl`):
   - The system automatically trains this model on first run using a RandomForestClassifier
   - It uses audio samples from the "data" directory with emotion labels in the filenames
   - If no training samples are found, it creates a basic dummy model
   - No download needed - just ensure you have audio samples in the "data" directory for better results
   - Training happens only once; the model is saved for future use

2. **Medical Data Embeddings** (`medical_data_embeddings.pkl`):
   ```
   # Generated from medical_data.csv on first run
   ```

## Note on Git Storage

Due to size constraints, the large model files (.gguf files) and PKL files should NOT be committed to Git. The .gitignore file already excludes these:

```
# Already in .gitignore
*.gguf
*.pkl
```

Instead, users should download the required files separately using the instructions above.

## Initial Setup

1. Clone the repository
2. Download the LLM model files (.gguf) as instructed above
3. Install dependencies: `pip install -r requirements.txt`
4. Ensure you have audio samples in the "data" directory (for better audio emotion recognition)
5. Run the application: `python main.py` 
