from dataclasses import dataclass

@dataclass
class CMDState:
    active_model: str = "Dizel Lite"
    active_mode: str = "Fast"
    active_provider: str = "local"
    generation_state: str = "IDLE"
    session_id: str = ""
    context_tokens: int = 0
    budget_tokens: int = 0
