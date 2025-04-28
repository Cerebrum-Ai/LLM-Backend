#!/bin/bash
#
# Cerebrum AI LLM Backend - Service Starter Script
# Usage: ./start_services.sh
#
# This script starts both the ML Models Service and the Main LLM Service
# in the background.

# Ensure we're in the project root directory
cd "$(dirname "$0")"

echo "🧠 Cerebrum AI LLM Backend Service Starter"
echo "=========================================="
echo

# Check if Python and conda are available
if ! command -v python &>/dev/null; then
    echo "❌ Python not found. Please install Python 3.9+ and try again."
    exit 1
fi

# Check if services are already running
if pgrep -f "python models.py" &>/dev/null; then
    echo "⚠️  ML Models service is already running!"
else
    echo "🚀 Starting ML Models Service..."
    python models.py &
    # Store the PID for potential later use
    MODELS_PID=$!
    echo "✅ ML Models Service started with PID: $MODELS_PID"
fi

# Give the models service time to initialize
echo "⏳ Waiting for ML Models Service to initialize (5 seconds)..."
sleep 5

# Check if main service is already running
if pgrep -f "python main.py" &>/dev/null; then
    echo "⚠️  Main LLM Service is already running!"
else
    echo "🚀 Starting Main LLM Service..."
    python main.py &
    # Store the PID for potential later use
    MAIN_PID=$!
    echo "✅ Main LLM Service started with PID: $MAIN_PID"
fi

echo
echo "✨ Both services have been started in the background."
echo "📋 To check service status, run:"
echo "   python test_endpoints.py https://your-llm-service-url.ngrok-free.app https://your-ml-models-url.ngrok-free.app"
echo
echo "📝 Log information will be displayed in the terminal."
echo "💡 To stop the services, run: ./stop_services.sh"
echo 