#!/bin/bash
# Test curl commands for Cerebrum AI API endpoints

# Set base URL (update as needed)
LLM_URL="http://localhost:5050"
ML_URL="http://localhost:9000"

echo "=== Cerebrum AI API Test Commands ==="
echo

# 1. Test Keystroke Analysis
echo "=== Testing Keystroke Analysis ==="
echo "curl -X POST ${LLM_URL}/api/analyze_typing -H \"Content-Type: application/json\" -d @- << 'EOF'
{
  \"keystrokes\": [
    {\"key\": \"a\", \"timeDown\": $(date +%s000), \"timeUp\": $(($(date +%s000) + 80)), \"pressure\": 0.8},
    {\"key\": \"b\", \"timeDown\": $(($(date +%s000) + 200)), \"timeUp\": $(($(date +%s000) + 270)), \"pressure\": 0.7},
    {\"key\": \"c\", \"timeDown\": $(($(date +%s000) + 400)), \"timeUp\": $(($(date +%s000) + 460)), \"pressure\": 0.9},
    {\"key\": \"d\", \"timeDown\": $(($(date +%s000) + 600)), \"timeUp\": $(($(date +%s000) + 650)), \"pressure\": 0.6},
    {\"key\": \"e\", \"timeDown\": $(($(date +%s000) + 800)), \"timeUp\": $(($(date +%s000) + 860)), \"pressure\": 0.8},
    {\"key\": \"f\", \"timeDown\": $(($(date +%s000) + 1000)), \"timeUp\": $(($(date +%s000) + 1070)), \"pressure\": 0.7},
    {\"key\": \"g\", \"timeDown\": $(($(date +%s000) + 1200)), \"timeUp\": $(($(date +%s000) + 1260)), \"pressure\": 0.9},
    {\"key\": \"h\", \"timeDown\": $(($(date +%s000) + 1400)), \"timeUp\": $(($(date +%s000) + 1450)), \"pressure\": 0.6},
    {\"key\": \"i\", \"timeDown\": $(($(date +%s000) + 1600)), \"timeUp\": $(($(date +%s000) + 1660)), \"pressure\": 0.8},
    {\"key\": \"j\", \"timeDown\": $(($(date +%s000) + 1800)), \"timeUp\": $(($(date +%s000) + 1870)), \"pressure\": 0.7}
  ],
  \"text\": \"abcdefghij\"
}
EOF"
echo
echo

# 2. Test Audio Analysis (using a sample test file)
echo "=== Testing Audio Analysis ==="
echo "# First, create a sample audio file if you don't have one:"
echo "# ffmpeg -f lavfi -i \"sine=frequency=1000:duration=5\" test_audio.wav"
echo
echo "# Then, send it to the API:"
echo "curl -X POST ${LLM_URL}/api/analyze_audio \\
  -F \"audio=@test_audio.wav\""
echo
echo "# Or send directly to ML service:"
echo "curl -X POST ${ML_URL}/ml/process \\
  -H \"Content-Type: application/json\" \\
  -d '{\"url\": \"file:///path/to/test_audio.wav\", \"data_type\": \"audio\", \"model\": \"emotion\"}'"
echo
echo

# 3. Test Image Analysis
echo "=== Testing Image Analysis ==="
echo "# Using a sample test image:"
echo "curl -X POST ${ML_URL}/ml/process \\
  -H \"Content-Type: application/json\" \\
  -d '{\"url\": \"https://example.com/sample-image.jpg\", \"data_type\": \"image\", \"model\": \"diagnosis\"}'"
echo
echo

# 4. Test API directly
echo "=== Testing LLM Chat API ==="
echo "curl -X POST ${LLM_URL}/api/chat \\
  -H \"Content-Type: application/json\" \\
  -d '{\"question\": \"What is Parkinson's disease?\", \"context\": \"\"}'"
echo
echo

echo "=== Run Commands ==="
echo "Copy-paste any of the above commands to test the respective endpoint."
echo "For keystroke test, use this simplified command:"

# Provide a ready-to-use keystroke test command
echo
echo "# Ready-to-use keystroke test command:"
echo "curl -X POST ${LLM_URL}/api/analyze_typing -H \"Content-Type: application/json\" -d '{
  \"keystrokes\": [
    {\"key\": \"a\", \"timeDown\": 1620136589000, \"timeUp\": 1620136589080, \"pressure\": 0.8},
    {\"key\": \"b\", \"timeDown\": 1620136589200, \"timeUp\": 1620136589270, \"pressure\": 0.7},
    {\"key\": \"c\", \"timeDown\": 1620136589400, \"timeUp\": 1620136589460, \"pressure\": 0.9},
    {\"key\": \"d\", \"timeDown\": 1620136589600, \"timeUp\": 1620136589650, \"pressure\": 0.6},
    {\"key\": \"e\", \"timeDown\": 1620136589800, \"timeUp\": 1620136589860, \"pressure\": 0.8},
    {\"key\": \"f\", \"timeDown\": 1620136590000, \"timeUp\": 1620136590070, \"pressure\": 0.7},
    {\"key\": \"g\", \"timeDown\": 1620136590200, \"timeUp\": 1620136590260, \"pressure\": 0.9},
    {\"key\": \"h\", \"timeDown\": 1620136590400, \"timeUp\": 1620136590450, \"pressure\": 0.6},
    {\"key\": \"i\", \"timeDown\": 1620136590600, \"timeUp\": 1620136590660, \"pressure\": 0.8},
    {\"key\": \"j\", \"timeDown\": 1620136590800, \"timeUp\": 1620136590870, \"pressure\": 0.7}
  ],
  \"text\": \"abcdefghij\"
}'" 