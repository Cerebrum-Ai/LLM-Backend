#!/bin/bash
# Test curl command for audio analysis

# Set base URL (update as needed)
LLM_URL="http://localhost:5050"

echo "Testing Cerebrum AI Audio Analysis API..."

# Check if test file exists or create it
if [ ! -f "tests/test_audio.wav" ]; then
  echo "Test audio file not found. Attempting to create a sample audio file..."
  if command -v ffmpeg &> /dev/null; then
    echo "Creating sample audio file with ffmpeg..."
    ffmpeg -f lavfi -i "sine=frequency=1000:duration=5" tests/test_audio.wav -y
  else
    echo "ffmpeg not found. Please install ffmpeg or provide a test_audio.wav file."
    exit 1
  fi
fi

# Test the endpoint
echo "Sending audio file to ${LLM_URL}/api/analyze_audio"
curl -X POST "${LLM_URL}/api/analyze_audio" \
  -F "audio=@tests/test_audio.wav" 