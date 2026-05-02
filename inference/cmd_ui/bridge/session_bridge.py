from typing import List, Dict, Optional
from textual.app import App
from inference.dizel_ui.logic.history_manager import (
    save_session, load_session, list_sessions, delete_session, toggle_pin_session, new_session_id
)

class SessionBridge:
    def __init__(self, app: App):
        self.app = app
        self.current_messages: List[Dict] = []
        
    def create(self) -> str:
        self.app.session_id = new_session_id()
        self.current_messages = []
        return self.app.session_id
        
    def add_message(self, role: str, content: str) -> None:
        self.current_messages.append({"role": role, "content": content})
        self.save()
        
    def save(self, title: str = None) -> str:
        if not self.current_messages:
            return ""
        if not self.app.session_id:
            self.app.session_id = new_session_id()
        save_session(self.current_messages, session_id=self.app.session_id, title=title)
        return self.app.session_id
        
    def load(self, session_id: str) -> bool:
        data = load_session(session_id)
        if data:
            self.app.session_id = session_id
            self.current_messages = data.get("messages", [])
            return True
        return False
        
    def get_all(self, query: str = None) -> List[Dict]:
        sessions = list_sessions()
        if query:
            q = query.lower()
            sessions = [s for s in sessions if q in s["title"].lower() or q in s["preview"].lower()]
        return sessions
        
    def delete(self, session_id: str) -> bool:
        if self.app.session_id == session_id:
            self.app.session_id = ""
            self.current_messages = []
        return delete_session(session_id)
        
    def pin(self, session_id: str) -> bool:
        return toggle_pin_session(session_id)
