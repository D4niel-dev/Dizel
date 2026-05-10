# dizel_ui/utils/signals.py

from PySide6.QtCore import QObject, Signal, QRunnable, Slot

class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    Supported signals are:
    
    finished
        No payload
    
    error
        `str` error message
        
    result
        `object` data returned from processing, anything
        
    progress
        `str` indicating progress string or token
    """
    finished = Signal()
    error = Signal(str)
    result = Signal(object)
    progress = Signal(str)

class GenerationWorker(QRunnable):
    """
    Worker thread for offloading chat generation tasks to avoid UI freezing.
    """
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            # We can pass the signals object to the function so it can emit progress if needed
            result = self.fn(self.signals, *self.args, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()
