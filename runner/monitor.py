#!/usr/bin/env python3
"""
Cerebrum AI LLM Backend - Performance Monitoring System

This script monitors the health and performance of all Cerebrum AI LLM backend services,
including the Main LLM service, ML Models service, and their API endpoints.

Features:
- Service availability checks
- API endpoint health monitoring
- Response time measurements
- Resource usage tracking (CPU, memory)
- Error detection and reporting
- Periodic testing of all major endpoints
- Dashboard output for system status

Usage:
  python monitor.py [--interval=60] [--llm-url=URL] [--ml-url=URL] [--log-file=FILE]
"""

import argparse
import json
import os
import platform
import psutil
import requests
import sys
import time
import datetime
import logging
import subprocess
import threading
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from dotenv import load_dotenv

# Load alert configuration
ALERT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'alert_config.json')
if os.path.exists(ALERT_CONFIG_PATH):
    with open(ALERT_CONFIG_PATH, 'r') as f:
        ALERT_CONFIG = json.load(f)
else:
    ALERT_CONFIG = {}

# Helper for alerting
class AlertManager:
    def __init__(self, config):
        self.config = config
        self.console_enabled = config.get('notification_channels', {}).get('console', {}).get('enabled', True)
        self.log_enabled = config.get('notification_channels', {}).get('log_file', {}).get('enabled', False)
        self.log_path = config.get('notification_channels', {}).get('log_file', {}).get('path', 'logs/alerts.log')
        self.email_enabled = config.get('notification_channels', {}).get('email', {}).get('enabled', False)
        self.slack_enabled = config.get('notification_channels', {}).get('slack', {}).get('enabled', False)
        if self.log_enabled and not os.path.exists(os.path.dirname(self.log_path)):
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def alert(self, message, level='warning'):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = f"[{timestamp}] [{level.upper()}] {message}"
        if self.console_enabled:
            print(msg)
        if self.log_enabled:
            with open(self.log_path, 'a') as f:
                f.write(msg + '\n')
        # Email/Slack integration can be added here

# Default URLs - will be overridden by command line args or .env file
LLM_SERVICE_URL = "http://localhost:5050"
ML_MODELS_URL = "http://localhost:9000"

# Load environment variables
load_dotenv()
if os.getenv("LLM_SERVICE_URL"):
    LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL")
if os.getenv("ML_MODELS_URL"):
    ML_MODELS_URL = os.getenv("ML_MODELS_URL")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("cerebrum-monitor")

# Global state for monitoring data
monitoring_data = {
    "llm_service": {
        "online": False,
        "last_checked": None,
        "response_time": None,
        "errors": [],
    },
    "ml_service": {
        "online": False,
        "last_checked": None,
        "response_time": None,
        "errors": [],
    },
    "endpoints": {},
    "system": {
        "cpu_percent": 0,
        "memory_percent": 0,
        "processes": {
            "main.py": False,
            "models.py": False,
        }
    }
}

class CerebrumMonitor:
    """Main monitoring class for Cerebrum AI LLM Backend"""
    
    def __init__(self, llm_url, ml_url, interval=60, log_file=None):
        """Initialize the monitor"""
        self.llm_url = llm_url
        self.ml_url = ml_url
        self.interval = interval  # seconds
        self.log_file = log_file
        self.console = Console()
        self.layout = self._setup_layout()
        
        # If log file provided, add file handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
    
    def _setup_layout(self):
        """Configure the rich layout for the monitoring dashboard"""
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1)
        )
        layout["main"].split_row(
            Layout(name="services", ratio=1),
            Layout(name="endpoints", ratio=2)
        )
        layout["main"]["endpoints"].split(
            Layout(name="llm_endpoints", ratio=1),
            Layout(name="ml_endpoints", ratio=1)
        )
        return layout
    
    def update_dashboard(self):
        """Update the dashboard with current monitoring data"""
        # Header
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = Panel(
            Text(f"🧠 Cerebrum AI LLM Backend Monitor - {current_time}", style="bold white"), 
            style="blue"
        )
        self.layout["header"].update(header)
        
        # Services Status
        services_table = Table(title="Services Status", show_header=True, header_style="bold")
        services_table.add_column("Service", style="dim")
        services_table.add_column("Status")
        services_table.add_column("Response Time")
        services_table.add_column("Last Checked")
        
        # LLM Service
        llm_status = "✅ Online" if monitoring_data["llm_service"]["online"] else "❌ Offline"
        llm_style = "green" if monitoring_data["llm_service"]["online"] else "red"
        llm_time = f"{monitoring_data['llm_service']['response_time']:.2f}ms" if monitoring_data["llm_service"]["response_time"] else "N/A"
        llm_checked = monitoring_data["llm_service"]["last_checked"].strftime("%H:%M:%S") if monitoring_data["llm_service"]["last_checked"] else "Never"
        
        services_table.add_row("LLM Service", f"[{llm_style}]{llm_status}[/{llm_style}]", llm_time, llm_checked)
        
        # ML Service
        ml_status = "✅ Online" if monitoring_data["ml_service"]["online"] else "❌ Offline"
        ml_style = "green" if monitoring_data["ml_service"]["online"] else "red"
        ml_time = f"{monitoring_data['ml_service']['response_time']:.2f}ms" if monitoring_data["ml_service"]["response_time"] else "N/A"
        ml_checked = monitoring_data["ml_service"]["last_checked"].strftime("%H:%M:%S") if monitoring_data["ml_service"]["last_checked"] else "Never"
        
        services_table.add_row("ML Models Service", f"[{ml_style}]{ml_status}[/{ml_style}]", ml_time, ml_checked)
        
        # System Info
        services_table.add_section()
        services_table.add_row("System CPU", f"{monitoring_data['system']['cpu_percent']}%", "", "")
        services_table.add_row("System Memory", f"{monitoring_data['system']['memory_percent']}%", "", "")
        
        # Processes
        services_table.add_section()
        for process_name, running in monitoring_data["system"]["processes"].items():
            status = "✅ Running" if running else "❌ Stopped"
            style = "green" if running else "red"
            services_table.add_row(f"Process: {process_name}", f"[{style}]{status}[/{style}]", "", "")
        
        self.layout["main"]["services"].update(services_table)
        
        # LLM Endpoints Status
        llm_table = Table(title="LLM Service Endpoints", show_header=True, header_style="bold")
        llm_table.add_column("Endpoint", style="dim")
        llm_table.add_column("Status")
        llm_table.add_column("Response Time")
        llm_table.add_column("Last Tested")
        
        for endpoint, data in monitoring_data["endpoints"].items():
            if endpoint.startswith("llm_"):
                status = "✅ OK" if data.get("status") == "ok" else "❌ Error"
                style = "green" if data.get("status") == "ok" else "red"
                resp_time = f"{data.get('response_time', 0):.2f}ms" if data.get('response_time') else "N/A"
                last_tested = data.get("last_tested").strftime("%H:%M:%S") if data.get("last_tested") else "Never"
                
                llm_table.add_row(
                    endpoint.replace("llm_", ""),
                    f"[{style}]{status}[/{style}]",
                    resp_time,
                    last_tested
                )
        
        self.layout["main"]["endpoints"]["llm_endpoints"].update(llm_table)
        
        # ML Endpoints Status
        ml_table = Table(title="ML Models Service Endpoints", show_header=True, header_style="bold")
        ml_table.add_column("Endpoint", style="dim")
        ml_table.add_column("Status")
        ml_table.add_column("Response Time")
        ml_table.add_column("Last Tested")
        
        for endpoint, data in monitoring_data["endpoints"].items():
            if endpoint.startswith("ml_"):
                status = "✅ OK" if data.get("status") == "ok" else "❌ Error"
                style = "green" if data.get("status") == "ok" else "red"
                resp_time = f"{data.get('response_time', 0):.2f}ms" if data.get('response_time') else "N/A"
                last_tested = data.get("last_tested").strftime("%H:%M:%S") if data.get("last_tested") else "Never"
                
                ml_table.add_row(
                    endpoint.replace("ml_", ""),
                    f"[{style}]{status}[/{style}]",
                    resp_time,
                    last_tested
                )
        
        self.layout["main"]["endpoints"]["ml_endpoints"].update(ml_table)
    
    def check_llm_service(self):
        """Check if the LLM service is available"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.llm_url}/api/chat", timeout=5)
            elapsed = (time.time() - start_time) * 1000  # Convert to ms
            
            monitoring_data["llm_service"]["online"] = True
            monitoring_data["llm_service"]["response_time"] = elapsed
            monitoring_data["llm_service"]["last_checked"] = datetime.datetime.now()
            logger.info(f"LLM service is online. Response time: {elapsed:.2f}ms")
            
            return True
        except requests.RequestException as e:
            monitoring_data["llm_service"]["online"] = False
            monitoring_data["llm_service"]["last_checked"] = datetime.datetime.now()
            monitoring_data["llm_service"]["errors"].append(str(e))
            logger.error(f"LLM service check failed: {str(e)}")
            
            # Keep only the last 5 errors
            if len(monitoring_data["llm_service"]["errors"]) > 5:
                monitoring_data["llm_service"]["errors"] = monitoring_data["llm_service"]["errors"][-5:]
            
            return False
    
    def check_ml_service(self):
        """Check if the ML Models service is available"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.ml_url}/", timeout=5)
            elapsed = (time.time() - start_time) * 1000  # Convert to ms
            
            monitoring_data["ml_service"]["online"] = True
            monitoring_data["ml_service"]["response_time"] = elapsed
            monitoring_data["ml_service"]["last_checked"] = datetime.datetime.now()
            logger.info(f"ML Models service is online. Response time: {elapsed:.2f}ms")
            
            return True
        except requests.RequestException as e:
            monitoring_data["ml_service"]["online"] = False
            monitoring_data["ml_service"]["last_checked"] = datetime.datetime.now()
            monitoring_data["ml_service"]["errors"].append(str(e))
            logger.error(f"ML Models service check failed: {str(e)}")
            
            # Keep only the last 5 errors
            if len(monitoring_data["ml_service"]["errors"]) > 5:
                monitoring_data["ml_service"]["errors"] = monitoring_data["ml_service"]["errors"][-5:]
            
            return False
    
    def check_system_resources(self):
        """Check system CPU and memory usage"""
        try:
            monitoring_data["system"]["cpu_percent"] = psutil.cpu_percent()
            monitoring_data["system"]["memory_percent"] = psutil.virtual_memory().percent
            
            # Check if processes are running
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'python' in cmdline:
                        if 'main.py' in cmdline:
                            monitoring_data["system"]["processes"]["main.py"] = True
                        elif 'models.py' in cmdline:
                            monitoring_data["system"]["processes"]["models.py"] = True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            logger.info(f"System: CPU: {monitoring_data['system']['cpu_percent']}%, "
                       f"Memory: {monitoring_data['system']['memory_percent']}%")
        except Exception as e:
            logger.error(f"Error checking system resources: {str(e)}")
    
    def test_llm_chat_endpoint(self):
        """Test the LLM chat endpoint with a simple query"""
        endpoint_key = "llm_chat"
        if endpoint_key not in monitoring_data["endpoints"]:
            monitoring_data["endpoints"][endpoint_key] = {}
        
        try:
            start_time = time.time()
            payload = {"question": "What is a test query?", "session_id": f"monitor_{int(time.time())}"}
            response = requests.post(f"{self.llm_url}/api/chat", json=payload, timeout=10)
            elapsed = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                monitoring_data["endpoints"][endpoint_key]["status"] = "ok"
            else:
                monitoring_data["endpoints"][endpoint_key]["status"] = "error"
                monitoring_data["endpoints"][endpoint_key]["error"] = f"HTTP {response.status_code}"
            
            monitoring_data["endpoints"][endpoint_key]["response_time"] = elapsed
            monitoring_data["endpoints"][endpoint_key]["last_tested"] = datetime.datetime.now()
            
            logger.info(f"LLM Chat endpoint test: status={monitoring_data['endpoints'][endpoint_key]['status']}, "
                       f"time={elapsed:.2f}ms")
        except requests.RequestException as e:
            monitoring_data["endpoints"][endpoint_key]["status"] = "error"
            monitoring_data["endpoints"][endpoint_key]["error"] = str(e)
            monitoring_data["endpoints"][endpoint_key]["last_tested"] = datetime.datetime.now()
            logger.error(f"LLM Chat endpoint test failed: {str(e)}")
    
    def test_llm_typing_analysis_endpoint(self):
        """Test the LLM typing analysis endpoint"""
        endpoint_key = "llm_typing_analysis"
        if endpoint_key not in monitoring_data["endpoints"]:
            monitoring_data["endpoints"][endpoint_key] = {}
        
        try:
            start_time = time.time()
            
            # Create typing data with enough keystrokes
            keystrokes = []
            for i in range(20):
                keystrokes.append({
                    "key": chr(97 + (i % 26)),  # a-z characters
                    "timeDown": 1680000000 + (i * 100),
                    "timeUp": 1680000000 + (i * 100) + 50,
                    "event": "keydown"
                })
            
            payload = {
                "url": json.dumps({"keystrokes": keystrokes}),
                "data_type": "typing",
                "model": "pattern"
            }
            response = requests.post(f"{self.ml_url}/ml/process", json=payload, timeout=10)
            elapsed = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code in [0,1000]:
                monitoring_data["endpoints"][endpoint_key]["status"] = "ok"
            else:
                monitoring_data["endpoints"][endpoint_key]["status"] = "error"
                monitoring_data["endpoints"][endpoint_key]["error"] = f"HTTP {response.status_code}"
            
            monitoring_data["endpoints"][endpoint_key]["response_time"] = elapsed
            monitoring_data["endpoints"][endpoint_key]["last_tested"] = datetime.datetime.now()
            
            logger.info(f"LLM Typing Analysis endpoint test: status={monitoring_data['endpoints'][endpoint_key]['status']}, "
                       f"time={elapsed:.2f}ms")
        except requests.RequestException as e:
            monitoring_data["endpoints"][endpoint_key]["status"] = "error"
            monitoring_data["endpoints"][endpoint_key]["error"] = str(e)
            monitoring_data["endpoints"][endpoint_key]["last_tested"] = datetime.datetime.now()
            logger.error(f"LLM Typing Analysis endpoint test failed: {str(e)}")
    
    def test_ml_process_endpoint(self):
        """Test the ML process endpoint with typing data"""
        endpoint_key = "ml_process"
        if endpoint_key not in monitoring_data["endpoints"]:
            monitoring_data["endpoints"][endpoint_key] = {}
        
        try:
            start_time = time.time()
            
            # Create typing data with enough keystrokes
            keystrokes = []
            for i in range(20):
                keystrokes.append({
                    "key": chr(97 + (i % 26)),  # a-z characters
                    "timeDown": 1680000000 + (i * 100),
                    "timeUp": 1680000000 + (i * 100) + 50,
                    "event": "keydown"
                })
            
            payload = {
                "url": json.dumps({"keystrokes": keystrokes}),
                "data_type": "typing",
                "model": "pattern"
            }
            
            response = requests.post(f"{self.ml_url}/ml/process", json=payload, timeout=10)
            elapsed = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code in [200, 400]:  # Even 400 means the endpoint is working
                monitoring_data["endpoints"][endpoint_key]["status"] = "ok"
            else:
                monitoring_data["endpoints"][endpoint_key]["status"] = "error"
                monitoring_data["endpoints"][endpoint_key]["error"] = f"HTTP {response.status_code}"
            
            monitoring_data["endpoints"][endpoint_key]["response_time"] = elapsed
            monitoring_data["endpoints"][endpoint_key]["last_tested"] = datetime.datetime.now()
            
            logger.info(f"ML Process endpoint test: status={monitoring_data['endpoints'][endpoint_key]['status']}, "
                       f"time={elapsed:.2f}ms")
        except requests.RequestException as e:
            monitoring_data["endpoints"][endpoint_key]["status"] = "error"
            monitoring_data["endpoints"][endpoint_key]["error"] = str(e)
            monitoring_data["endpoints"][endpoint_key]["last_tested"] = datetime.datetime.now()
            logger.error(f"ML Process endpoint test failed: {str(e)}")
    
    def test_llm_audio_analysis_endpoint(self):
        """Test the LLM audio analysis endpoint (dummy test as we can't easily send real audio)"""
        endpoint_key = "llm_audio_analysis"
        if endpoint_key not in monitoring_data["endpoints"]:
            monitoring_data["endpoints"][endpoint_key] = {}
        
        try:
            # Just check if endpoint exists by sending an OPTIONS request
            start_time = time.time()
            response = requests.options(f"{self.llm_url}/api/analyze_audio", timeout=5)
            elapsed = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code < 500:  # Even 4xx is OK for OPTIONS as we're just checking availability
                monitoring_data["endpoints"][endpoint_key]["status"] = "ok"
            else:
                monitoring_data["endpoints"][endpoint_key]["status"] = "error"
                monitoring_data["endpoints"][endpoint_key]["error"] = f"HTTP {response.status_code}"
            
            monitoring_data["endpoints"][endpoint_key]["response_time"] = elapsed
            monitoring_data["endpoints"][endpoint_key]["last_tested"] = datetime.datetime.now()
            
            logger.info(f"LLM Audio Analysis endpoint check: status={monitoring_data['endpoints'][endpoint_key]['status']}, "
                       f"time={elapsed:.2f}ms")
        except requests.RequestException as e:
            monitoring_data["endpoints"][endpoint_key]["status"] = "error"
            monitoring_data["endpoints"][endpoint_key]["error"] = str(e)
            monitoring_data["endpoints"][endpoint_key]["last_tested"] = datetime.datetime.now()
            logger.error(f"LLM Audio Analysis endpoint check failed: {str(e)}")
    
    def test_ml_audio_emotion_endpoint(self):
        """Test the ML audio emotion endpoint (dummy test as we can't easily send real audio)"""
        endpoint_key = "ml_audio_emotion"
        if endpoint_key not in monitoring_data["endpoints"]:
            monitoring_data["endpoints"][endpoint_key] = {}
        
        try:
            # Send a basic request to check endpoint health
            start_time = time.time()
            payload = {
                "url": "test_data",  # This will fail but we just want to check the endpoint
                "data_type": "audio",
                "model": "emotion"
            }
            response = requests.post(f"{self.ml_url}/ml/process", json=payload, timeout=5)
            elapsed = (time.time() - start_time) * 1000  # Convert to ms
            
            # 400 is also OK as it means the endpoint exists but rejected our dummy data
            if response.status_code in [200, 400]:
                monitoring_data["endpoints"][endpoint_key]["status"] = "ok"
            else:
                monitoring_data["endpoints"][endpoint_key]["status"] = "error"
                monitoring_data["endpoints"][endpoint_key]["error"] = f"HTTP {response.status_code}"
            
            monitoring_data["endpoints"][endpoint_key]["response_time"] = elapsed
            monitoring_data["endpoints"][endpoint_key]["last_tested"] = datetime.datetime.now()
            
            logger.info(f"ML Audio Emotion endpoint check: status={monitoring_data['endpoints'][endpoint_key]['status']}, "
                       f"time={elapsed:.2f}ms")
        except requests.RequestException as e:
            monitoring_data["endpoints"][endpoint_key]["status"] = "error"
            monitoring_data["endpoints"][endpoint_key]["error"] = str(e)
            monitoring_data["endpoints"][endpoint_key]["last_tested"] = datetime.datetime.now()
            logger.error(f"ML Audio Emotion endpoint check failed: {str(e)}")
    
    def run_tests(self):
        """Run all tests for services and endpoints based on alert_config.json"""
        logger.info("Starting test cycle...")
        alert_mgr = AlertManager(ALERT_CONFIG)

        # Check process status
        # Reset process status before checking
        for proc in monitoring_data['system']['processes']:
            monitoring_data['system']['processes'][proc] = False
        self.check_system_resources()
        for proc, required in monitoring_data['system']['processes'].items():
            if not required:
                alert_mgr.alert(f"Process {proc} is not running!", level='critical')

        # Check resource thresholds
        resource_cfg = ALERT_CONFIG.get('alerts', {}).get('resource_usage', {})
        cpu = monitoring_data['system']['cpu_percent']
        mem = monitoring_data['system']['memory_percent']
        if cpu > resource_cfg.get('cpu_threshold_percent', 100):
            alert_mgr.alert(f"CPU usage high: {cpu}%", level='warning')
        if mem > resource_cfg.get('memory_threshold_percent', 100):
            alert_mgr.alert(f"Memory usage high: {mem}%", level='warning')

        # Service checks
        llm_ok = self.check_llm_service()
        ml_ok = self.check_ml_service()
        if not llm_ok:
            alert_mgr.alert("LLM Service is DOWN!", level='critical')
        if not ml_ok:
            alert_mgr.alert("ML Models Service is DOWN!", level='critical')

        # Endpoint checks from config
        endpoints_cfg = ALERT_CONFIG.get('monitoring', {}).get('endpoints', {})
        # Reset endpoint status at the start of each test cycle
        for endpoint in endpoints_cfg:
            monitoring_data['endpoints'][endpoint] = {}
        for endpoint, cfg in endpoints_cfg.items():
            timeout = cfg.get('timeout_seconds', 10)
            expected_codes = cfg.get('expected_status_code', 200)
            if not isinstance(expected_codes, list):
                expected_codes = [expected_codes]
            # Determine URL and payload
            if endpoint.startswith('llm_'):
                base_url = self.llm_url
            else:
                base_url = self.ml_url
            if endpoint == 'llm_chat':
                url = f"{base_url}/api/chat"
                payload = cfg.get('test_payload', {"question": "What is a test query?", "session_id": "monitor_test"})
                method = 'post'
            elif endpoint == 'llm_typing_analysis':
                url = f"{base_url}/ml/forward"
                keystrokes = [{"key": chr(97 + (i % 26)), "timeDown": 1680000000 + (i * 100), "timeUp": 1680000000 + (i * 100) + 50, "event": "keydown"} for i in range(20)]
                payload = {"keystrokes": keystrokes}
                method = 'post'
            elif endpoint == 'llm_audio_analysis':
                url = f"{base_url}/api/analyze_audio"
                payload = None
                method = 'options'
            elif endpoint == 'ml_process':
                url = f"{base_url}/ml/process"
                keystrokes = [{"key": chr(97 + (i % 26)), "timeDown": 1680000000 + (i * 100), "timeUp": 1680000000 + (i * 100) + 50, "event": "keydown"} for i in range(20)]
                payload = {"url": json.dumps({"keystrokes": keystrokes}), "data_type": "typing", "model": "pattern"}
                method = 'post'
            elif endpoint == 'ml_audio_emotion':
                url = f"{base_url}/ml/process"
                payload = {"url": "test_data", "data_type": "audio", "model": "emotion"}
                method = 'post'
            else:
                continue
            try:
                start_time = time.time()
                if method == 'post':
                    resp = requests.post(url, json=payload, timeout=timeout)
                elif method == 'options':
                    resp = requests.options(url, timeout=timeout)
                else:
                    resp = requests.get(url, timeout=timeout)
                elapsed = (time.time() - start_time) * 1000
                status_ok = resp.status_code in expected_codes if endpoint != 'llm_audio_analysis' else resp.status_code < 500
                monitoring_data['endpoints'][endpoint]['response_time'] = elapsed
                monitoring_data['endpoints'][endpoint]['last_tested'] = datetime.datetime.now()
                if status_ok:
                    monitoring_data['endpoints'][endpoint]['status'] = 'ok'
                else:
                    monitoring_data['endpoints'][endpoint]['status'] = 'error'
                    monitoring_data['endpoints'][endpoint]['error'] = f"HTTP {resp.status_code}"
                    alert_mgr.alert(f"Endpoint {endpoint} failed with status {resp.status_code}", level='warning')
            except Exception as e:
                monitoring_data['endpoints'][endpoint]['status'] = 'error'
                monitoring_data['endpoints'][endpoint]['error'] = str(e)
                monitoring_data['endpoints'][endpoint]['last_tested'] = datetime.datetime.now()
                alert_mgr.alert(f"Endpoint {endpoint} check failed: {e}", level='critical')

        logger.info("Test cycle completed")
    
    def run_monitoring_loop(self):
        """Main monitoring loop"""
        logger.info(f"Starting Cerebrum AI LLM Backend Monitor")
        logger.info(f"LLM Service URL: {self.llm_url}")
        logger.info(f"ML Models URL: {self.ml_url}")
        logger.info(f"Monitoring interval: {self.interval} seconds")
        
        try:
            with Live(self.layout, refresh_per_second=1, screen=True):
                while True:
                    self.run_tests()
                    self.update_dashboard()
                    time.sleep(self.interval)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
            raise

import signal
import atexit

# --- Process Management: Start/Stop Services ---
SERVICE_PROCS = {}

def is_process_running(script_name):
    """Check if a process with the given script name is running."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and script_name in ' '.join(proc.info['cmdline']):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

def start_service(script_name):
    """Start a Python service as a subprocess if not running."""
    if is_process_running(script_name):
        logger.info(f"{script_name} is already running.")
        return None
    logger.info(f"Starting {script_name}...")
    proc = subprocess.Popen([sys.executable, script_name], cwd=os.path.dirname(os.path.abspath(__file__)) + '/..')
    SERVICE_PROCS[script_name] = proc
    logger.info(f"{script_name} started with PID {proc.pid}")
    return proc

def stop_services():
    """Terminate all started service subprocesses."""
    for name, proc in SERVICE_PROCS.items():
        if proc and proc.poll() is None:
            logger.info(f"Terminating {name} (PID {proc.pid})...")
            try:
                proc.terminate()
                proc.wait(timeout=10)
                logger.info(f"{name} terminated.")
            except Exception as e:
                logger.warning(f"Failed to terminate {name}: {e}")

# Register cleanup
atexit.register(stop_services)

def handle_signal(sig, frame):
    logger.info(f"Received signal {sig}. Stopping services...")
    stop_services()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Cerebrum AI LLM Backend Monitor")
    parser.add_argument("--interval", type=int, default=60, help="Monitoring interval in seconds")
    parser.add_argument("--llm-url", type=str, default=LLM_SERVICE_URL, help="LLM Service URL")
    parser.add_argument("--ml-url", type=str, default=ML_MODELS_URL, help="ML Models Service URL")
    parser.add_argument("--log-file", type=str, help="Log file path")
    parser.add_argument("--auto-start", action='store_true', help="Automatically start backend services if not running")
    args = parser.parse_args()

    # Optionally start backend services
    if args.auto_start:
        # Start ML Models first, then Main LLM
        start_service('models.py')
        time.sleep(5)  # Allow ML models to initialize
        start_service('main.py')
        time.sleep(3)  # Allow LLM to initialize

    monitor = CerebrumMonitor(
        llm_url=args.llm_url,
        ml_url=args.ml_url,
        interval=args.interval,
        log_file=args.log_file
    )
    
    monitor.run_monitoring_loop()

if __name__ == "__main__":
    main() 