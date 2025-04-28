#!/bin/bash
# Test curl command for keystroke analysis

# Set base URL (update as needed)
LLM_URL="http://localhost:5050"

echo "Testing Cerebrum AI Keystroke Analysis API..."
echo "Note: Run this script from the project root directory"
echo

echo "Sending request to ${LLM_URL}/api/analyze_typing"

curl -X POST "${LLM_URL}/api/analyze_typing" \
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