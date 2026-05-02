from textual.message import Message
from inference.dizel_ui.logic.chat_manager import ChatManager
from inference.dizel_ui.logic.token_budget import allocate_token_budget, classify_task
from textual.app import App
from textual.app import App

class ChatBridge:
    class TokenReceived(Message):
        def __init__(self, token: str):
            self.token = token
            super().__init__()

    class GenerationDone(Message):
        def __init__(self, full_text: str):
            self.full_text = full_text
            super().__init__()

    class GenerationError(Message):
        def __init__(self, error_msg: str):
            self.error_msg = error_msg
            super().__init__()
            
    class SystemLog(Message):
        def __init__(self, message: str):
            self.message = message
            super().__init__()

    def __init__(self, app: App):
        self.app = app
        self.manager = ChatManager()

    def send(self, text: str) -> None:
        workspace = self.app.query_one("WorkspacePanel")
        
        # 1. Budgeting
        has_tools = False
        if hasattr(self.app, "_tool_states"):
            has_tools = any(self.app._tool_states.values())
        
        task_type = classify_task(text, has_tools)
        input_tokens = len(text.split()) * 2  # rough estimate
        max_cap = int(self.app.usage_manager.max_capacity) if hasattr(self.app, 'usage_manager') else 4096
        
        budget = allocate_token_budget(
            task_type=task_type,
            input_token_count=input_tokens,
            context_tokens=self.app.context_tokens,
            model_ctx_length=max_cap
        )
        self.app.budget_tokens = budget
        
        self.app.call_from_thread(workspace.post_message, self.SystemLog(f"Routing to {self.app.active_provider} · {self.app.active_model}"))
        self.app.call_from_thread(workspace.post_message, self.SystemLog(f"Task: {task_type.value} | Budget: {budget} | Context: {self.app.context_tokens}/{max_cap}"))
        
        def on_token(token: str) -> None:
            # Update context token usage safely from main thread
            def _update_tokens():
                self.app.context_tokens += 1
                if hasattr(self.app, "usage_manager"):
                    cost = 1.0 if self.app.active_mode == "Planning" else 0.5
                    self.app.usage_manager.add_usage(cost)
            
            self.app.call_from_thread(_update_tokens)
                
            workspace = self.app.query_one("WorkspacePanel")
            self.app.call_from_thread(workspace.post_message, self.TokenReceived(token))

        def on_done(full_text: str) -> None:
            workspace = self.app.query_one("WorkspacePanel")
            self.app.call_from_thread(workspace.post_message, self.GenerationDone(full_text))

        def on_error(error_msg: str) -> None:
            workspace = self.app.query_one("WorkspacePanel")
            self.app.call_from_thread(workspace.post_message, self.GenerationError(error_msg))

        self.manager.send_message(
            user_text=text,
            attachments=[],
            on_token=on_token,
            on_done=on_done,
            on_error=on_error,
        )

    def stop(self) -> None:
        self.manager.stop_generation()
