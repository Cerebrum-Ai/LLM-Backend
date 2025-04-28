#!/bin/bash

# Base URL - replace with your actual ngrok URL
BASE_URL="https://8e9b-104-28-219-95.ngrok-free.app"

# Test typing analysis endpoint
echo "Testing typing analysis endpoint..."
curl -X POST "${BASE_URL}/analyze/typing" \
  -H "Content-Type: application/json" \
  -d '{
    "typing_data": {
      "keystrokes": [
        {"key": "h", "press_time": 100, "release_time": 200},
        {"key": "e", "press_time": 300, "release_time": 400},
        {"key": "l", "press_time": 500, "release_time": 600},
        {"key": "l", "press_time": 700, "release_time": 800},
        {"key": "o", "press_time": 900, "release_time": 1000}
      ],
      "text": "hello"
    }
  }'

echo -e "\n\nTesting typing pattern endpoint..."
curl -X POST "${BASE_URL}/analyze/typing/pattern" \
  -H "Content-Type: application/json" \
  -d '{
    "typing_data": {
      "keystrokes": [
        {"key": "t", "press_time": 100, "release_time": 200},
        {"key": "e", "press_time": 300, "release_time": 400},
        {"key": "s", "press_time": 500, "release_time": 600},
        {"key": "t", "press_time": 700, "release_time": 800}
      ],
      "text": "test"
    }
  }'

echo -e "\n\nTesting typing speed endpoint..."
curl -X POST "${BASE_URL}/analyze/typing/speed" \
  -H "Content-Type: application/json" \
  -d '{
    "typing_data": {
      "keystrokes": [
        {"key": "q", "press_time": 100, "release_time": 200},
        {"key": "u", "press_time": 300, "release_time": 400},
        {"key": "i", "press_time": 500, "release_time": 600},
        {"key": "c", "press_time": 700, "release_time": 800},
        {"key": "k", "press_time": 900, "release_time": 1000}
      ],
      "text": "quick"
    }
  }'

echo -e "\n\nTesting typing rhythm endpoint..."
curl -X POST "${BASE_URL}/analyze/typing/rhythm" \
  -H "Content-Type: application/json" \
  -d '{
    "typing_data": {
      "keystrokes": [
        {"key": "r", "press_time": 100, "release_time": 200},
        {"key": "h", "press_time": 300, "release_time": 400},
        {"key": "y", "press_time": 500, "release_time": 600},
        {"key": "t", "press_time": 700, "release_time": 800},
        {"key": "h", "press_time": 900, "release_time": 1000},
        {"key": "m", "press_time": 1100, "release_time": 1200}
      ],
      "text": "rhythm"
    }
  }' 