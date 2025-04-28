from flask import Flask, request, jsonify
import os
import sys
import tempfile
import base64
import json
from audio.emotion.audio_processor import SimpleAudioAnalyzer

app = Flask(__name__)
audio_analyzer = SimpleAudioAnalyzer.get_instance()

@app.route('/status', methods=['GET']) 
def status():
    return jsonify({"status": "ok"})

@app.route('/ml/process', methods=['POST'])
def process_ml():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        data_type = data.get("data_type")
        model_type = data.get("model")
        
        if data_type == "audio" and model_type == "emotion":
            # Get the audio path
            file_path = data.get("url")
            if not file_path or not os.path.exists(file_path):
                return jsonify({"error": f"File not found: {file_path}"}), 400
            
            # Analyze audio
            result = audio_analyzer.analyze_audio(file_path)
            return jsonify({"emotion": result})
        else:
            return jsonify({"error": f"Unsupported data_type or model: {data_type}/{model_type}"}), 400
    
    except Exception as e:
        print(f"Error processing ML request: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting ML Models Service on port 9000...")
    app.run(host="0.0.0.0", port=9000) 


