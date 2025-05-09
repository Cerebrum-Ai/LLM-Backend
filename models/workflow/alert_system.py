from typing import Dict, List, Optional
from datetime import datetime
import json
import requests
from threading import Thread
import time

class AlertSystem:
    """
    Alert System for critical findings and emergency notifications
    """
    def __init__(self, config_path: str = 'alert_config.json'):
        self.config = self._load_config(config_path)
        self.alert_history = []
        self.alert_threads = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load alert configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                'alert_channels': {
                    'console': {'enabled': True},
                    'email': {'enabled': False},
                    'slack': {'enabled': False},
                    'pager': {'enabled': False}
                },
                'alert_levels': {
                    'CRITICAL': {'color': 'red', 'priority': 1},
                    'HIGH': {'color': 'orange', 'priority': 2},
                    'MEDIUM': {'color': 'yellow', 'priority': 3},
                    'LOW': {'color': 'blue', 'priority': 4}
                },
                'notification_cooldown': 300,  # 5 minutes
                'max_alerts_per_hour': 10
            }
            
    def send_alert(self, 
                  message: str,
                  level: str = 'HIGH',
                  source: str = 'SYSTEM',
                  data: Optional[Dict] = None) -> bool:
        """
        Send an alert through configured channels
        
        Args:
            message: Alert message
            level: Alert level (CRITICAL, HIGH, MEDIUM, LOW)
            source: Source of the alert
            data: Optional additional data
            
        Returns:
            bool: True if alert was sent successfully
        """
        if not self._should_send_alert(level):
            return False
            
        alert = {
            'message': message,
            'level': level,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'data': data or {}
        }
        
        # Add to history
        self.alert_history.append(alert)
        
        # Send through all enabled channels
        success = True
        for channel, config in self.config['alert_channels'].items():
            if config.get('enabled', False):
                if not self._send_to_channel(channel, alert):
                    success = False
                    
        return success
    
    def _should_send_alert(self, level: str) -> bool:
        """Check if alert should be sent based on rate limiting and cooldown"""
        # Check rate limiting
        recent_alerts = [a for a in self.alert_history 
                        if (datetime.now() - datetime.fromisoformat(a['timestamp'])).total_seconds() < 3600]
        if len(recent_alerts) >= self.config['max_alerts_per_hour']:
            return False
            
        # Check cooldown for same level
        if self.alert_history:
            last_alert = self.alert_history[-1]
            if (last_alert['level'] == level and 
                (datetime.now() - datetime.fromisoformat(last_alert['timestamp'])).total_seconds() < 
                self.config['notification_cooldown']):
                return False
                
        return True
    
    def _send_to_channel(self, channel: str, alert: Dict) -> bool:
        """Send alert to specific channel"""
        try:
            if channel == 'console':
                return self._send_to_console(alert)
            elif channel == 'email':
                return self._send_to_email(alert)
            elif channel == 'slack':
                return self._send_to_slack(alert)
            elif channel == 'pager':
                return self._send_to_pager(alert)
            return False
        except Exception as e:
            print(f"Error sending alert to {channel}: {str(e)}")
            return False
    
    def _send_to_console(self, alert: Dict) -> bool:
        """Send alert to console"""
        try:
            level_color = self.config['alert_levels'][alert['level']]['color']
            print(f"\n[{alert['timestamp']}] {alert['level']} ALERT from {alert['source']}:")
            print(f"Message: {alert['message']}")
            if alert['data']:
                print(f"Data: {json.dumps(alert['data'], indent=2)}")
            return True
        except Exception as e:
            print(f"Error sending to console: {str(e)}")
            return False
    
    def _send_to_email(self, alert: Dict) -> bool:
        """Send alert via email"""
        # Implement email sending logic here
        # This would typically use SMTP or an email service API
        return False
    
    def _send_to_slack(self, alert: Dict) -> bool:
        """Send alert to Slack"""
        if 'webhook_url' not in self.config['alert_channels']['slack']:
            return False
            
        try:
            webhook_url = self.config['alert_channels']['slack']['webhook_url']
            payload = {
                'text': f"*{alert['level']} ALERT* from {alert['source']}\n{alert['message']}",
                'attachments': [{
                    'color': self.config['alert_levels'][alert['level']]['color'],
                    'fields': [
                        {
                            'title': 'Timestamp',
                            'value': alert['timestamp'],
                            'short': True
                        },
                        {
                            'title': 'Source',
                            'value': alert['source'],
                            'short': True
                        }
                    ]
                }]
            }
            
            if alert['data']:
                payload['attachments'][0]['fields'].append({
                    'title': 'Additional Data',
                    'value': json.dumps(alert['data'], indent=2),
                    'short': False
                })
                
            response = requests.post(webhook_url, json=payload)
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending to Slack: {str(e)}")
            return False
    
    def _send_to_pager(self, alert: Dict) -> bool:
        """Send alert to pager system"""
        # Implement pager system integration here
        # This would typically use a pager service API
        return False
    
    def get_alert_history(self, 
                         level: Optional[str] = None,
                         source: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Dict]:
        """Get filtered alert history"""
        filtered = self.alert_history
        
        if level:
            filtered = [a for a in filtered if a['level'] == level]
        if source:
            filtered = [a for a in filtered if a['source'] == source]
        if start_time:
            filtered = [a for a in filtered if datetime.fromisoformat(a['timestamp']) >= start_time]
        if end_time:
            filtered = [a for a in filtered if datetime.fromisoformat(a['timestamp']) <= end_time]
            
        return filtered
    
    def clear_alert_history(self):
        """Clear alert history"""
        self.alert_history = [] 