from typing import Any, Dict
from inference.dizel_ui.logic.config_manager import ConfigManager

class ConfigService:
    """Service wrapper for application configuration"""
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get the current application configuration"""
        return ConfigManager.load()
        
    @staticmethod
    def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the application configuration with partial changes.
        Returns the updated configuration.
        """
        config = ConfigManager.load()
        
        # Merge updates
        for key, val in updates.items():
            if isinstance(val, dict) and key in config and isinstance(config[key], dict):
                config[key].update(val)
            else:
                config[key] = val
                
        ConfigManager.save(config)
        return config
