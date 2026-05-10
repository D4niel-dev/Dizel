from PySide6.QtCore import QObject
from .states import AvatarState
from . import animations

class AvatarController(QObject):
    """
    Manages the AvatarWidget's state machine, triggering the appropriate QPropertyAnimations
    and ensuring cleanup between transitions.
    """
    def __init__(self, avatar_widget, parent=None):
        super().__init__(parent)
        self.widget = avatar_widget
        self.current_state = None
        self._active_group = None
        self._cleanup_group = None

        # Start default state without triggering visual snap
        self.set_state(AvatarState.IDLE, instant=True)

    def play_initialization(self):
        """
        Special function to trigger the startup fade in.
        Returns to IDLE automatically when complete.
        """
        self._stop_current()
        self._active_group = animations.make_fade_in(self.widget)
        self._active_group.finished.connect(lambda: self.set_state(AvatarState.IDLE))
        self._active_group.start()

    def _stop_current(self):
        if self._active_group and self._active_group.state() == self._active_group.Running:
            self._active_group.stop()
            self._active_group.deleteLater()
            self._active_group = None

    def set_state(self, state: AvatarState, instant=False):
        if self.current_state == state:
            return
            
        self.current_state = state
        self._stop_current()
        
        if instant:
            # Snap parameters hard
            self.widget.set_scale(1.0)
            self.widget.set_glow(0.0)
            self.widget.set_ring_op(0.0)
            self.widget.set_opacity(1.0)
        else:
            # We first transition softly to neutral, then trigger the new target state's animation.
            # To avoid delays, we simply launch the new animation simultaneously.
            # Only untouched properties will reset to neutral correctly.
            pass

        # Dispatch the correct visual routine
        if state == AvatarState.IDLE:
            # If moving to IDLE, we just need a cleanup followed by the breathing loop
            # To handle cleanup safely: run cleanup, then when finished, run breathing
            self._cleanup_group = animations.make_cleanup(self.widget)
            self._cleanup_group.finished.connect(self._start_idle_loop)
            self._cleanup_group.start()
            
        elif state == AvatarState.LISTENING:
            self._cleanup_group = animations.make_cleanup(self.widget)
            self._cleanup_group.start()
            self._active_group = animations.make_listening(self.widget)
            self._active_group.start()
            
        elif state == AvatarState.THINKING:
            self._active_group = animations.make_thinking(self.widget)
            self._active_group.start()
            
        elif state == AvatarState.RESPONDING:
            self._cleanup_group = animations.make_cleanup(self.widget)
            self._cleanup_group.start()
            self._active_group = animations.make_responding(self.widget)
            self._active_group.start()
            
        elif state == AvatarState.ERROR:
            self._cleanup_group = animations.make_cleanup(self.widget)
            self._cleanup_group.start()
            self._active_group = animations.make_error(self.widget)
            self._active_group.finished.connect(lambda: self.set_state(AvatarState.IDLE))
            self._active_group.start()

    def _start_idle_loop(self):
        # Callback wrapper to prevent python garbage collection on lambda references
        if self.current_state == AvatarState.IDLE:
            self._active_group = animations.make_idle(self.widget)
            self._active_group.start()
