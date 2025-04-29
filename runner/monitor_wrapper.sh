#!/bin/bash
#
# Cerebrum AI LLM Backend - Monitor Wrapper Script
# This script runs the monitoring system in the background
# Usage: ./monitor_wrapper.sh [start|stop|status]
#
# It can start the monitor in the background, stop it, and check its status

# Function to check if monitor is running
check_monitor() {
    if pgrep -f "python monitor.py" > /dev/null; then
        return 0  # Running
    else
        return 1  # Not running
    fi
}

# Get monitor PID if running
get_monitor_pid() {
    pgrep -f "python monitor.py"
}

# Start the monitor
start_monitor() {
    if check_monitor; then
        echo "📊 Monitor is already running with PID $(get_monitor_pid)"
        return 0
    fi
    
    echo "🚀 Starting Cerebrum AI Monitor..."
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Start the monitor in the background, redirecting output to log file
    python monitor.py --log-file=logs/monitor_$(date +%Y%m%d_%H%M%S).log "$@" > logs/monitor_stdout.log 2>&1 &
    
    # Save PID to file
    echo $! > .monitor.pid
    
    echo "✅ Monitor started with PID $!"
    echo "📝 Logs are being saved to logs/monitor_stdout.log"
}

# Stop the monitor
stop_monitor() {
    if ! check_monitor; then
        echo "❌ Monitor is not running"
        return 1
    fi
    
    PID=$(get_monitor_pid)
    echo "🛑 Stopping monitor with PID $PID..."
    kill $PID
    
    # Wait to see if it exited
    sleep 2
    
    if check_monitor; then
        echo "⚠️  Monitor didn't stop gracefully, forcing termination..."
        kill -9 $PID
        sleep 1
    fi
    
    if ! check_monitor; then
        echo "✅ Monitor stopped"
        # Remove PID file if it exists
        [ -f .monitor.pid ] && rm .monitor.pid
    else
        echo "❌ Failed to stop monitor"
        return 1
    fi
}

# Check monitor status
monitor_status() {
    if check_monitor; then
        PID=$(get_monitor_pid)
        echo "✅ Monitor is running with PID $PID"
        echo "📊 Runtime: $(ps -o etime= -p $PID)"
        echo "💻 CPU usage: $(ps -o %cpu= -p $PID)%"
        echo "🧠 Memory usage: $(ps -o %mem= -p $PID)%"
    else
        echo "❌ Monitor is not running"
        return 1
    fi
}

# Main script logic
case "$1" in
    start)
        shift  # Remove 'start' from arguments
        start_monitor "$@"
        ;;
    stop)
        stop_monitor
        ;;
    restart)
        stop_monitor
        shift  # Remove 'restart' from arguments
        start_monitor "$@"
        ;;
    status)
        monitor_status
        ;;
    *)
        echo "Usage: $0 [start|stop|restart|status] [monitor options]"
        echo
        echo "Examples:"
        echo "  $0 start                    # Start monitor with default settings"
        echo "  $0 start --interval=30      # Start monitor with 30-second intervals"
        echo "  $0 stop                     # Stop the running monitor"
        echo "  $0 restart                  # Restart the monitor"
        echo "  $0 status                   # Check if monitor is running"
        exit 1
        ;;
esac

exit 0 