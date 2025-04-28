#!/bin/bash
# Test curl command for image analysis

# Set base URL (update as needed)
ML_URL="http://localhost:9000"

echo "Testing Cerebrum AI Image Analysis API..."
echo "Note: Run this script from the project root directory"
echo

# Option 1: Use a sample URL
SAMPLE_URL="https://www.example.com/sample-image.jpg"
echo "Option 1: Testing with a sample URL (${SAMPLE_URL})"
echo "Sending request to ${ML_URL}/ml/process"
curl -X POST "${ML_URL}/ml/process" \
  -H "Content-Type: application/json" \
  -d "{
    \"url\": \"${SAMPLE_URL}\",
    \"data_type\": \"image\",
    \"model\": \"diagnosis\"
  }"

echo -e "\n\n"

# Option 2: Upload a local file
echo "Option 2: To test with a local image file, run:"
echo "curl -X POST \"${ML_URL}/ml/process\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"url\": \"file:///path/to/tests/test_image.jpg\", \"data_type\": \"image\", \"model\": \"diagnosis\"}'" 