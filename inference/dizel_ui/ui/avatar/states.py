from enum import Enum, auto

class AvatarState(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    RESPONDING = auto()
    ERROR = auto()
