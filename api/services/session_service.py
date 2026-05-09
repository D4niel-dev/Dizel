from typing import Dict, List, Optional
from inference.dizel_ui.logic import history_manager

class SessionService:
    """Service wrapper for session and history management"""
    
    @staticmethod
    def list_sessions() -> List[Dict]:
        """List all available sessions"""
        return history_manager.list_sessions()
        
    @staticmethod
    def get_session(session_id: str) -> Optional[Dict]:
        """Get a specific session by ID"""
        return history_manager.load_session(session_id)
        
    @staticmethod
    def create_or_update_session(messages: List[Dict], session_id: Optional[str] = None, title: Optional[str] = None) -> str:
        """Save or update a session"""
        return history_manager.save_session(messages, session_id=session_id, title=title)
        
    @staticmethod
    def delete_session(session_id: str) -> bool:
        """Delete a session"""
        return history_manager.delete_session(session_id)
        
    @staticmethod
    def toggle_pin(session_id: str) -> bool:
        """Toggle the pin status of a session"""
        return history_manager.toggle_pin_session(session_id)
