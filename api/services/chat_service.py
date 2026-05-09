import threading
import queue
from typing import Dict, Any, Generator, Optional
from inference.dizel_ui.logic.chat_manager import ChatManager
from fastapi import HTTPException

class ChatService:
    """Service wrapper for ChatManager"""
    
    _manager = ChatManager()

    @classmethod
    def get_manager(cls) -> ChatManager:
        return cls._manager

    @classmethod
    def load_model(cls, checkpoint_path: str, device: Optional[str] = None) -> Dict[str, str]:
        """Load a local PyTorch model checkpoint"""
        try:
            # Note: The UI runs this in a background thread and uses a callback.
            # In a synchronous REST endpoint, we can block or run it in a thread.
            # For simplicity, we'll run it synchronously here so the client knows when it's done.
            cls._manager.load_model(checkpoint_path, device=device)
            return {"status": "success", "message": f"Model {checkpoint_path} loaded successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    @classmethod
    def switch_provider(cls) -> Dict[str, str]:
        """Reload the provider from config (if it was changed)"""
        cls._manager.reload_provider()
        return {"status": "success", "active_provider": cls._manager.active_provider_slug}

    @classmethod
    def apply_profile(cls, model_name: str, mode_name: str) -> Dict[str, str]:
        """Apply a generation profile"""
        cls._manager.apply_profile(model_name, mode_name)
        return {"status": "success", "profile": f"{model_name} / {mode_name}"}

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Get the current state of the ChatManager"""
        return {
            "is_ready": cls._manager.is_ready,
            "is_generating": cls._manager.is_generating,
            "provider": cls._manager.active_provider_slug,
            "api_model": cls._manager.active_api_model,
            "model_info": cls._manager.model_info
        }

    @classmethod
    def stream_chat(cls, user_text: str, session_id: Optional[str] = None, attachments: Optional[list] = None) -> Generator[str, None, None]:
        """
        Generate a response and yield tokens as Server-Sent Events (SSE).
        """
        if not cls._manager.is_ready:
            yield "data: {\"error\": \"Model not loaded or configured.\"}\n\n"
            return
            
        if cls._manager.is_generating:
            yield "data: {\"error\": \"Already generating.\"}\n\n"
            return

        # Restore history if a session is provided
        if session_id:
            from api.services.session_service import SessionService
            session = SessionService.get_session(session_id)
            if session and "messages" in session:
                cls._manager.load_history(session["messages"])
        else:
            cls._manager.new_session()
            session_id = cls._manager.session_id

        # We need a thread-safe way to get tokens from the ChatManager's background thread
        # into this generator.
        token_queue = queue.Queue()
        
        # Sentinels
        DONE = object()
        ERROR = object()

        def on_token(token: str):
            token_queue.put(token)
            
        def on_done(full_text: str):
            token_queue.put(DONE)
            # Save the session automatically
            from api.services.session_service import SessionService
            new_id = SessionService.create_or_update_session(cls._manager.get_history(), session_id=session_id)
            if not cls._manager.session_id:
                cls._manager.session_id = new_id
            
        def on_error(err_msg: str):
            token_queue.put((ERROR, err_msg))

        # Start generation
        cls._manager.send_message(
            user_text=user_text,
            attachments=attachments or [],
            on_token=on_token,
            on_done=on_done,
            on_error=on_error
        )

        import json
        
        # Consume queue
        while True:
            item = token_queue.get()
            if item is DONE:
                break
            elif isinstance(item, tuple) and item[0] is ERROR:
                error_msg = item[1]
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                break
            else:
                # Normal token
                yield f"data: {json.dumps({'token': item})}\n\n"
                
        # Send an explicit done event
        yield "data: {\"done\": true}\n\n"
