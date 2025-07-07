"""
Configuration settings for the schedule notifier.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration manager for the schedule notifier."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return self._get_default_config()
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "whatsapp": {
                "default_group_id": "",
                "default_phone_number": "",
                "send_time": "08:00",
                "wait_time": 10,
                "auto_send": False
            },
            "schedule": {
                "weekend_notifications": False,
                "reminder_days": 7,
                "data_file": "data/schedule.json"
            },
            "display": {
                "use_rich_formatting": True,
                "show_emojis": True,
                "table_style": "cyan"
            },
            "notifications": {
                "daily_schedule": True,
                "exam_reminders": True,
                "assignment_reminders": True,
                "class_reminders": True
            }
        }
    
    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        current = self.data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    # Convenience properties
    @property
    def whatsapp_group_id(self) -> str:
        return self.get("whatsapp.default_group_id", "")
    
    @property
    def whatsapp_phone_number(self) -> str:
        return self.get("whatsapp.default_phone_number", "")
    
    @property
    def send_time(self) -> str:
        return self.get("whatsapp.send_time", "08:00")
    
    @property
    def auto_send(self) -> bool:
        return self.get("whatsapp.auto_send", False)
    
    @property
    def data_file(self) -> str:
        return self.get("schedule.data_file", "data/schedule.json")
    
    @property
    def reminder_days(self) -> int:
        return self.get("schedule.reminder_days", 7)
    
    @property
    def weekend_notifications(self) -> bool:
        return self.get("schedule.weekend_notifications", False)


class DataManager:
    """Manages loading and saving schedule data."""
    
    def __init__(self, data_file: str = "data/schedule.json"):
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(exist_ok=True)
    
    def load_schedule_data(self) -> Dict[str, Any]:
        """Load schedule data from JSON file."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return self._get_default_schedule_data()
        else:
            return self._get_default_schedule_data()
    
    def save_schedule_data(self, data: Dict[str, Any]):
        """Save schedule data to JSON file."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving schedule data: {e}")
    
    def _get_default_schedule_data(self) -> Dict[str, Any]:
        """Get default schedule data structure."""
        return {
            "subjects": [],
            "exams": [],
            "assignments": [],
            "last_updated": None
        }
    
    def backup_data(self):
        """Create a backup of current data."""
        if self.data_file.exists():
            backup_file = self.data_file.with_suffix('.backup.json')
            try:
                import shutil
                shutil.copy2(self.data_file, backup_file)
                print(f"Backup created: {backup_file}")
            except Exception as e:
                print(f"Error creating backup: {e}")


# Environment variable helpers
def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with optional default."""
    return os.getenv(name, default)


def is_development() -> bool:
    """Check if running in development mode."""
    return get_env_var("ENVIRONMENT", "production").lower() == "development"


def get_log_level() -> str:
    """Get logging level from environment."""
    return get_env_var("LOG_LEVEL", "INFO").upper()


# Global config instance
config = Config()
data_manager = DataManager(config.data_file)