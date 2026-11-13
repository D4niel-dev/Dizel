from typing import Dict, List, Optional
from dizel_core import history_manager

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

    @staticmethod
    def rename_session(session_id: str, title: str) -> bool:
        """Rename a saved session."""
        return history_manager.rename_session(session_id, title)

    @staticmethod
    def toggle_archive(session_id: str) -> bool:
        """Toggle the archived flag of a session."""
        return history_manager.toggle_archive_session(session_id)

    @staticmethod
    def truncate(session_id: str, keep: int) -> bool:
        """Keep only the first `keep` messages."""
        return history_manager.truncate_session(session_id, keep)

    @staticmethod
    def delete_all() -> int:
        """Delete every saved session. Returns how many were removed."""
        return history_manager.delete_all_sessions()
