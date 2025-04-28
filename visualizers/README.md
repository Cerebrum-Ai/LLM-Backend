# Cerebrum AI Typing Analysis Tools

This suite of tools allows you to analyze typing patterns for potential neurological conditions using the Cerebrum AI LLM Backend.

## Features

- **Real-time capture and analysis** of keystroke data
- **Visualization** of typing patterns and features
- **Comparison** of typing patterns across multiple subjects
- **ML-powered analysis** to detect potential conditions:
  - Parkinson's Disease
  - Essential Tremor
  - Carpal Tunnel Syndrome
  - Multiple Sclerosis
  - Normal typing patterns

## Tools Included

### 1. Typing Test Script (`test_typing.py`)

A simple script to test the typing analysis functionality by sending generated typing data to the API.

```bash
# Test with default parameters (normal typing)
./test_typing.py

# Simulate Parkinson's typing pattern
./test_typing.py --condition parkinsons

# Simulate Essential Tremor typing pattern
./test_typing.py --condition essential_tremor

# Use a custom endpoint URL
./test_typing.py --url http://custom-server:5050
```

### 2. Typing Visualizer (`typing_visualizer.py`)

A real-time visualization tool that captures your typing and displays patterns as you type.

```bash
# Start live recording and visualization
./typing_visualizer.py

# Load and visualize saved typing data
./typing_visualizer.py --mode file --file typing_data_1234567890.json

# Connect to a different ML service
./typing_visualizer.py --ml-url http://custom-ml-service:9000
```

**Controls:**
- Press `a` to analyze current typing data with the ML service
- Press `s` to save the current recording
- Press `q` to quit the visualizer

### 3. Typing Analysis Dashboard (`typing_dashboard.py`)

A web-based dashboard for comparing typing patterns across multiple subjects/recordings.

```bash
# Start the dashboard
./typing_dashboard.py
```

Then visit http://127.0.0.1:8050/ in your browser.

## Testing with Curl Commands

For direct testing of the API endpoints, use these curl commands:

### 1. Keystroke Analysis Test

```bash
curl -X POST "http://localhost:5050/api/analyze_typing" \
  -H "Content-Type: application/json" \
  -d '{
  "keystrokes": [
    {"key": "a", "timeDown": 1620136589000, "timeUp": 1620136589080, "pressure": 0.8},
    {"key": "b", "timeDown": 1620136589200, "timeUp": 1620136589270, "pressure": 0.7},
    {"key": "c", "timeDown": 1620136589400, "timeUp": 1620136589460, "pressure": 0.9},
    {"key": "d", "timeDown": 1620136589600, "timeUp": 1620136589650, "pressure": 0.6},
    {"key": "e", "timeDown": 1620136589800, "timeUp": 1620136589860, "pressure": 0.8},
    {"key": "f", "timeDown": 1620136590000, "timeUp": 1620136590070, "pressure": 0.7},
    {"key": "g", "timeDown": 1620136590200, "timeUp": 1620136590260, "pressure": 0.9},
    {"key": "h", "timeDown": 1620136590400, "timeUp": 1620136590450, "pressure": 0.6},
    {"key": "i", "timeDown": 1620136590600, "timeUp": 1620136590660, "pressure": 0.8},
    {"key": "j", "timeDown": 1620136590800, "timeUp": 1620136590870, "pressure": 0.7}
  ],
  "text": "abcdefghij"
}'
```

### 2. Audio Analysis Test

```bash
# First, create a sample audio file if you don't have one:
ffmpeg -f lavfi -i "sine=frequency=1000:duration=5" test_audio.wav

# Then, send it to the API:
curl -X POST "http://localhost:5050/api/analyze_audio" \
  -F "audio=@test_audio.wav"
```

### 3. Direct ML Service Test

```bash
# Image analysis:
curl -X POST "http://localhost:9000/ml/process" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/sample-image.jpg", "data_type": "image", "model": "diagnosis"}'

# Audio analysis:
curl -X POST "http://localhost:9000/ml/process" \
  -H "Content-Type: application/json" \
  -d '{"url": "file:///path/to/test_audio.wav", "data_type": "audio", "model": "emotion"}'

# Typing analysis:
curl -X POST "http://localhost:9000/ml/process" \
  -H "Content-Type: application/json" \
  -d '{"url": {"keystrokes": [...], "text": "sample"}, "data_type": "typing", "model": "pattern"}'
```

### 4. Chat API Test

```bash
curl -X POST "http://localhost:5050/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Parkinson\'s disease?", "context": ""}'
```

## Understanding the Analysis

The system analyzes several key features from typing patterns:

| Feature | Description | Relevance |
|---------|-------------|-----------|
| Key Press Duration | How long keys are held down | Longer in Parkinson's |
| Duration Variability | Consistency of key press time | Higher in neurological conditions |
| Time Between Keys | Time between consecutive keypresses | Longer in motor disorders |
| Rhythm Variability | Consistency of typing rhythm | Less consistent in disorders |
| Error Rate | Frequency of typing errors | Higher in many conditions |
| Typing Speed | Characters per minute | Slower in motor disorders |
| Rhythm Consistency | Pattern stability over time | Lower in neurological disorders |
| Pause Frequency | How often typing pauses occur | More pauses in some conditions |

## API Endpoints

The typing analysis functionality uses two main endpoints:

1. **LLM Service Endpoint**: `/api/analyze_typing`
   - Accepts keystroke data
   - Forwards to ML service for processing
   - Returns analysis results

2. **ML Service Endpoint**: `/ml/process`
   - Parameters: `url` (keystroke data), `data_type: "typing"`, `model: "pattern"`
   - Processes keystroke patterns
   - Returns condition detection and probabilities

## Data Format

The keystroke data should be structured as follows:

```json
{
  "keystrokes": [
    {
      "key": "a",
      "timeDown": 1620136589000,
      "timeUp": 1620136589080,
      "pressure": 0.8
    },
    {
      "key": "b",
      "timeDown": 1620136589200,
      "timeUp": 1620136589270,
      "pressure": 0.7
    }
  ],
  "text": "ab"
}
```

## Requirements

- Python 3.7+
- The following Python packages:
  - requests
  - matplotlib
  - numpy
  - keyboard
  - dash
  - dash-bootstrap-components
  - plotly
  - pandas
  
```bash
pip install requests matplotlib numpy keyboard dash dash-bootstrap-components plotly pandas
```

## Integration

These tools work with the Cerebrum AI LLM Backend. Make sure both the LLM service and ML Models service are running before using these tools.

## Privacy Notice

Keystroke data can contain sensitive information. The tools are designed to:
- Only store data locally unless explicitly sent to the ML service
- Only analyze typing patterns, not the content of what is typed
- Allow you to disable data collection at any time 