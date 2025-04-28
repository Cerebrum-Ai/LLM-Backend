#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
    fi
}

# Function to check if a service is running
check_service() {
    curl -s $1 > /dev/null
    return $?
}

# Function to test LLM service
test_llm_service() {
    echo -e "${YELLOW}Testing LLM Service...${NC}"
    
    # Test health endpoint
    curl -s http://localhost:5050/api/chat > /dev/null
    print_status $? "LLM Service Health Check"
    
    # Test chat endpoint
    response=$(curl -s -X POST http://localhost:5050/api/chat \
        -H "Content-Type: application/json" \
        -d '{"question": "Hello, how are you?", "context": ""}')
    
    if [[ $response == *"response"* ]]; then
        print_status 0 "LLM Chat Endpoint"
    else
        print_status 1 "LLM Chat Endpoint"
    fi
}

# Function to test ML Models service
test_ml_service() {
    echo -e "${YELLOW}Testing ML Models Service...${NC}"
    
    # Test health endpoint
    curl -s http://localhost:9000/ > /dev/null
    print_status $? "ML Service Health Check"
    
    # Test typing analysis endpoint
    response=$(curl -s -X POST http://localhost:9000/ml/process \
        -H "Content-Type: application/json" \
        -d '{
            "url": {
                "keystrokes": [
                    {"key": "a", "timeDown": 1620136589000, "timeUp": 1620136589080, "pressure": 0.8},
                    {"key": "b", "timeDown": 1620136589200, "timeUp": 1620136589270, "pressure": 0.7}
                ],
                "text": "ab"
            },
            "data_type": "typing",
            "model": "pattern"
        }')
    
    if [[ $response == *"result"* ]]; then
        print_status 0 "ML Typing Analysis Endpoint"
    else
        print_status 1 "ML Typing Analysis Endpoint"
    fi
}

# Function to test monitor script
test_monitor() {
    echo -e "${YELLOW}Testing Monitor Script...${NC}"
    
    # Test if monitor script exists and is executable
    if [ -x "../monitor.py" ]; then
        print_status 0 "Monitor Script Exists"
        
        # Test monitor wrapper
        if [ -x "../monitor_wrapper.sh" ]; then
            print_status 0 "Monitor Wrapper Exists"
            
            # Test monitor status
            ../monitor_wrapper.sh status > /dev/null
            print_status $? "Monitor Status Check"
        else
            print_status 1 "Monitor Wrapper Exists"
        fi
    else
        print_status 1 "Monitor Script Exists"
    fi
}

# Main test execution
echo -e "${YELLOW}Starting Comprehensive Tests...${NC}"

# Check if services are running
echo -e "${YELLOW}Checking Service Status...${NC}"
check_service "http://localhost:5050/api/chat"
print_status $? "LLM Service Running"

check_service "http://localhost:9000/"
print_status $? "ML Service Running"

# Run component tests
test_llm_service
test_ml_service
test_monitor

echo -e "${YELLOW}Tests Completed${NC}" 