#!/bin/bash
#
# Cerebrum AI LLM Backend - Service Stopper Script
# Usage: ./stop_services.sh
#
# This script stops both the ML Models Service and the Main LLM Service.

echo "🛑 Cerebrum AI LLM Backend Service Stopper"
echo "=========================================="
echo

# Check for running ML Models Service
if pgrep -f "python models.py" &>/dev/null; then
    echo "🔄 Stopping ML Models Service..."
    pkill -f "python models.py"
    
    # Check if it was successfully stopped
    if ! pgrep -f "python models.py" &>/dev/null; then
        echo "✅ ML Models Service stopped successfully."
    else
        echo "❌ Failed to stop ML Models Service. Try manually with:"
        echo "   pkill -9 -f \"python models.py\""
    fi
else
    echo "ℹ️  ML Models Service is not running."
fi

# Check for running Main LLM Service
if pgrep -f "python main.py" &>/dev/null; then
    echo "🔄 Stopping Main LLM Service..."
    pkill -f "python main.py"
    
    # Check if it was successfully stopped
    if ! pgrep -f "python main.py" &>/dev/null; then
        echo "✅ Main LLM Service stopped successfully."
    else
        echo "❌ Failed to stop Main LLM Service. Try manually with:"
        echo "   pkill -9 -f \"python main.py\""
    fi
else
    echo "ℹ️  Main LLM Service is not running."
fi

# Check for any zombie ngrok processes
if pgrep -f "ngrok" &>/dev/null; then
    echo "🧟 Found zombie ngrok processes. Stopping them..."
    pkill -f "ngrok"
    
    # Check if they were successfully stopped
    if ! pgrep -f "ngrok" &>/dev/null; then
        echo "✅ Ngrok processes stopped successfully."
    else
        echo "❌ Failed to stop ngrok processes. Try manually with:"
        echo "   pkill -9 -f \"ngrok\""
    fi
fi

echo
echo "🎉 All services have been stopped."
echo "To restart the services, run: ./start_services.sh"
echo 