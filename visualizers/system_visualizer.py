#!/usr/bin/env python3

"""
System Visualizer for Cerebrum AI LLM Backend
A real-time monitoring dashboard that displays system metrics and service health.
"""

import time
import psutil
import requests
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from datetime import datetime

# Configuration
UPDATE_INTERVAL = 1  # seconds
LLM_SERVICE_URL = "http://localhost:5050"
ML_SERVICE_URL = "http://localhost:9000"

console = Console()

def get_system_metrics():
    """Get current system metrics."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        'cpu': cpu_percent,
        'memory_percent': memory.percent,
        'memory_used': memory.used / (1024**3),  # GB
        'memory_total': memory.total / (1024**3),  # GB
        'disk_percent': disk.percent,
        'disk_used': disk.used / (1024**3),  # GB
        'disk_total': disk.total / (1024**3),  # GB
    }

def check_service_health(url):
    """Check if a service is healthy."""
    try:
        response = requests.get(url, timeout=2)
        return response.status_code == 200
    except:
        return False

def create_metrics_table(metrics):
    """Create a table with system metrics."""
    table = Table(title="System Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("CPU Usage", f"{metrics['cpu']}%")
    table.add_row("Memory Usage", f"{metrics['memory_percent']}% ({metrics['memory_used']:.1f}GB / {metrics['memory_total']:.1f}GB)")
    table.add_row("Disk Usage", f"{metrics['disk_percent']}% ({metrics['disk_used']:.1f}GB / {metrics['disk_total']:.1f}GB)")
    
    return table

def create_service_status_table():
    """Create a table with service health status."""
    table = Table(title="Service Status")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="green")
    
    llm_status = "🟢 Running" if check_service_health(LLM_SERVICE_URL) else "🔴 Down"
    ml_status = "🟢 Running" if check_service_health(ML_SERVICE_URL) else "🔴 Down"
    
    table.add_row("LLM Service", llm_status)
    table.add_row("ML Service", ml_status)
    
    return table

def main():
    """Main function to run the visualizer."""
    console.clear()
    console.print(Panel.fit("Cerebrum AI LLM Backend - System Visualizer", style="bold blue"))
    
    with Live(console=console, refresh_per_second=1) as live:
        while True:
            # Get current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Get metrics and create tables
            metrics = get_system_metrics()
            metrics_table = create_metrics_table(metrics)
            service_table = create_service_status_table()
            
            # Create layout
            layout = Layout()
            layout.split_column(
                Layout(Panel(Text(current_time, style="bold yellow"), title="Current Time")),
                Layout(service_table),
                Layout(metrics_table)
            )
            
            # Update display
            live.update(layout)
            time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Visualizer stopped by user.[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]") 