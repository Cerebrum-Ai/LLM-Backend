#!/bin/bash

# Installation script for Cerebrum AI Typing Analysis Tools

echo "Installing Cerebrum AI Typing Analysis Tools..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed or not in your PATH"
    echo "Please install pip and try again"
    exit 1
fi

# Install required packages
echo "Installing required Python packages..."
pip install -r typing_analysis_requirements.txt

# Make scripts executable
echo "Making scripts executable..."
chmod +x test_typing.py
chmod +x typing_visualizer.py
chmod +x typing_dashboard.py

# Check if required Python modules are available
echo "Verifying installation..."
python -c "import requests, matplotlib, numpy, keyboard, dash, dash_bootstrap_components, plotly, pandas" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ All dependencies successfully installed!"
    echo "
    Typing Analysis Tools are ready to use!
    
    Available tools:
      - ./test_typing.py - Basic API testing tool
      - ./typing_visualizer.py - Real-time typing pattern visualization
      - ./typing_dashboard.py - Web dashboard for analysis
    
    For more information, see TYPING_ANALYSIS_README.md
    "
else
    echo "❌ Some dependencies could not be installed."
    echo "Please check the error messages above and try installing manually:"
    echo "pip install requests matplotlib numpy keyboard dash dash-bootstrap-components plotly pandas"
fi 