from typing import List, Dict, Optional
import time
from textual.app import App
from dizel_core.history_manager import (
    save_session, load_session, list_sessions, delete_session, toggle_pin_session,
    rename_session, new_session_id
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
        self.current_messages.append({"role": role, "content": content, "ts": time.time()})
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
            sessions = [s for s in sessions if q in s.get("title", "").lower() or q in s.get("preview", "").lower() or any(q in t.lower() for t in s.get("tags", []))]
        return sessions
        
    def delete(self, session_id: str) -> bool:
        if self.app.session_id == session_id:
            self.app.session_id = ""
            self.current_messages = []
        return delete_session(session_id)
        
    def pin(self, session_id: str) -> bool:
        return toggle_pin_session(session_id)

    def rename(self, session_id: str, title: str) -> bool:
        return rename_session(session_id, title)
