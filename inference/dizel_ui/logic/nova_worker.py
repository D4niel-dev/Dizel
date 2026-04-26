# dizel_ui/logic/nova_worker.py

import queue
import time
import numpy as np
from PySide6.QtCore import QThread, Signal

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import whisper
    import torch
except ImportError:
    whisper = None

class NovaWorker(QThread):
    partial_text = Signal(str)   # Emitted every ~2s with current transcription
    final_text = Signal(str)     # Emitted when recording stops (or auto-timeout)
    error = Signal(str)          # Emitted on mic/model error
    amplitude = Signal(float)    # Emitted for waveform visualization (0.0-1.0)
    status_update = Signal(str)  # Emitted for UI status (loading, listening, etc)

    def __init__(self, model_size="base", language="auto", silence_timeout=5, parent=None):
        super().__init__(parent)
        self.model_size = model_size
        self.language = language
        self.silence_timeout = silence_timeout
        self.running = False
        self.q = queue.Queue()

    def run(self):
        if sd is None:
            self.error.emit("sounddevice module not found. Check requirements.")
            return
        if whisper is None:
            self.error.emit("openai-whisper model not found. Check requirements.")
            return

        self.status_update.emit("Loading speech model...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # load_model downloads if not present. This might block.
            model = whisper.load_model(self.model_size, device=device)
        except Exception as e:
            self.error.emit(f"Failed to load model: {str(e)}")
            return

        def callback(indata, frames, time_info, status):
            if status:
                pass
            self.q.put(indata.copy())

        self.running = True
        self.status_update.emit("Listening...")
        sample_rate = 16000
        buffer = []
        last_transcribe_time = time.time()
        last_speech_time = time.time()
        silence_threshold = 0.005  # RMS amplitude threshold for speech

        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', callback=callback):
                while self.running:
                    # Drain queue
                    chunk_added = False
                    while not self.q.empty():
                        data = self.q.get_nowait()
                        buffer.append(data)
                        chunk_added = True
                        
                        amp = float(np.sqrt(np.mean(data**2)))
                        # Scale amplitude for UI (0.0 to 1.0 roughly)
                        ui_amp = min(1.0, amp * 10.0) 
                        self.amplitude.emit(ui_amp)
                        
                        if amp > silence_threshold:
                            last_speech_time = time.time()
                    
                    # Transcribe periodically if we have enough new data
                    if chunk_added and time.time() - last_transcribe_time > 1.5:
                        audio_data = np.concatenate(buffer).flatten()
                        if len(audio_data) > sample_rate * 1: # at least 1s of audio
                            args = {}
                            if self.language != "auto":
                                args["language"] = self.language
                            
                            # this is a blocking call, but audio cap keeps running in background
                            result = model.transcribe(audio_data, fp16=torch.cuda.is_available(), **args)
                            text = result["text"].strip()
                            if text:
                                self.partial_text.emit(text)
                            
                        last_transcribe_time = time.time()
                        
                    # Auto stop on silence
                    if time.time() - last_speech_time > self.silence_timeout and len(buffer) > 0:
                        break

                    time.sleep(0.05)

        except Exception as e:
            self.error.emit(f"Microphone error: {str(e)}")
            self.running = False
            return

        # Final transcription pass
        self.running = False
        if len(buffer) > 0:
            self.status_update.emit("Processing...")
            audio_data = np.concatenate(buffer).flatten()
            if len(audio_data) > sample_rate * 0.5: # at least 0.5s audio
                args = {}
                if self.language != "auto":
                    args["language"] = self.language
                result = model.transcribe(audio_data, fp16=torch.cuda.is_available(), **args)
                self.final_text.emit(result["text"].strip())
            else:
                self.final_text.emit("")
        else:
            self.final_text.emit("")

    def stop(self):
        self.running = False
