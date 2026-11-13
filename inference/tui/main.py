import argparse
import contextlib
import faulthandler
import os
import re
import shutil
import sys
import threading
import time
import traceback

from inference.tui.app import DizelCMDApp


def _headless(prompt: str, checkpoint: str, device: str) -> int:
    """Scriptable mode: tokens on stdout, everything else on stderr."""
    from dizel_core.chat_manager import ChatManager

    manager = ChatManager()
    out = sys.stdout
    status = {"error": None}
    done = threading.Event()

    with contextlib.redirect_stdout(sys.stderr):
        if checkpoint:
            manager.load_model(checkpoint, device, on_progress=lambda msg: print(msg, file=sys.stderr))
        elif not manager.is_ready:
            print(
                "No engine ready. Pass --checkpoint, or configure a provider in Dizel's settings.",
                file=sys.stderr,
            )
            return 2

        out.write("\n")
        out.flush()

        def on_token(token: str) -> None:
            out.write(token)
            out.flush()

        def on_done(full_text: str) -> None:
            out.write("\n")
            out.flush()
            done.set()

        def on_error(message: str) -> None:
            status["error"] = message
            done.set()

        manager.send_message(
            user_text=prompt,
            attachments=[],
            on_token=on_token,
            on_done=on_done,
            on_error=on_error,
        )

        if not done.wait(timeout=900):
            manager.stop_generation()
            print("Timed out after 900s.", file=sys.stderr)
            return 3

    if status["error"]:
        print(f"[error] {status['error']}", file=sys.stderr)
        return 1
    return 0


def _warm_local_runtime(checkpoint_arg: str, repo_root: str) -> None:
    """Import heavy local-inference modules on the main thread before Textual.

    Importing torch from the loader thread can stall the TUI on Windows before
    any progress is emitted. Importing here is deterministic and lets the
    loader thread focus on the checkpoint itself.
    """
    try:
        from dizel_core.config_manager import ConfigManager

        cfg = ConfigManager.load()
        api_cfg = cfg.get("api_router", {})
        provider = api_cfg.get("provider", "local") if isinstance(api_cfg, dict) else "local"
        if provider != "local":
            return

        path = checkpoint_arg or str(cfg.get("checkpoint", "") or "")
        if not path:
            ckpt_dir = os.path.join(repo_root, "checkpoints")
            if os.path.isdir(ckpt_dir):
                candidates = [
                    name
                    for name in os.listdir(ckpt_dir)
                    if os.path.splitext(name)[1].lower()
                    in (".pt", ".pth", ".gguf", ".safetensors", ".bin", ".ckpt")
                ]
                if candidates:
                    path = os.path.join(ckpt_dir, candidates[0])

        ext = os.path.splitext(path)[1].lower()

        import importlib
        import importlib.util

        # Before Textual starts there is no UI at all, so this phase looks like a
        # dead terminal. Trace it — importing torch or the llama.cpp DLL is the
        # slowest thing in the whole startup on Windows.
        trace = open(
            os.path.join(repo_root, "logs", "tui_load.log"), "a", encoding="utf-8", buffering=1
        )
        try:
            label = os.path.basename(path) if path else "(no checkpoint)"
            trace.write(f"\n--- warm-up · {label} ---\n")
            trace.write("importing dizel_core.model_backend\n")
            trace.flush()
            importlib.import_module("dizel_core.model_backend")
            trace.write("model_backend imported\n")
            trace.flush()
            # A checkpoint can be chosen after startup (/load), so warm
            # llama.cpp whenever it is installed instead of only when the
            # current path happens to be a GGUF.
            if ext == ".gguf" or importlib.util.find_spec("llama_cpp") is not None:
                trace.write("importing llama_cpp\n")
                trace.flush()
                importlib.import_module("llama_cpp")
                trace.write("llama_cpp imported\n")
                trace.flush()
        finally:
            trace.close()
    except Exception:
        # Let the normal loader surface a useful error in the TUI.
        pass


# ── Pre-UI output ────────────────────────────────────────────────────────
# This runs before Textual exists, so there is no stylesheet — just ANSI. Kept
# short on purpose: it is a loading screen, not a log.
_C = {
    "mark": "\x1b[38;5;75m",
    "dim": "\x1b[90m",
    "text": "\x1b[97m",
    "warn": "\x1b[38;5;214m",
    "ok": "\x1b[38;5;78m",
    "off": "\x1b[0m",
}


def _paint(text: str, colour: str) -> str:
    stream = sys.__stderr__
    if not (stream and getattr(stream, "isatty", lambda: False)()):
        return text
    return f"{_C[colour]}{text}{_C['off']}"


def _shorten(value: str, limit: int) -> str:
    """Keep both ends of a long path; the filename matters more than the root."""
    text = " ".join(str(value).split())
    if len(text) <= limit or limit <= 1:
        return text if len(text) <= limit else "…"
    head = (limit - 1) // 2
    return f"{text[:head]}…{text[-(limit - 1 - head):]}"


class _StepSpinner:
    """One animated line for the newest load step.

    Progress arrives on the loading thread; the animation runs on a daemon
    thread. Writes are serialised so a finalised step never collides with a
    frame. Non-TTY runs skip the animation and keep the plain step lines.
    """

    def __init__(self, stream=None) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._gutter = ""
        self._message = ""
        self._active = False
        self._phase = 0
        self._stream = stream if stream is not None else sys.__stderr__
        stream = self._stream
        self._tty = bool(stream and getattr(stream, "isatty", lambda: False)())
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        try:
            frames.encode(getattr(stream, "encoding", None) or "utf-8")
        except Exception:
            frames = "|/-\\"
        self._frames = frames
        try:
            self._width = shutil.get_terminal_size((80, 24)).columns
        except Exception:
            self._width = 80

    @property
    def tty(self) -> bool:
        return self._tty

    def begin(self, gutter: str, message: str) -> None:
        if not self._tty:
            return
        text = _shorten(message, max(20, self._width - 14))
        with self._lock:
            self._gutter = gutter
            self._message = text
            self._active = True
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="dizel-load-spinner"
            )
            self._thread.start()

    def _run(self) -> None:
        stream = self._stream
        while not self._stop.wait(0.09):
            with self._lock:
                if not self._active:
                    continue
                frame = self._frames[self._phase % len(self._frames)]
                self._phase += 1
                line = (
                    f"\r  {self._gutter}  {_paint(frame, 'mark')}  "
                    f"{_paint(self._message, 'text')}\x1b[K"
                )
            try:
                stream.write(line)
                stream.flush()
            except Exception:
                return

    def finish(self, marker: str = "·", colour: str = "dim") -> None:
        if not self._tty:
            return
        stream = self._stream
        with self._lock:
            if not self._active:
                return
            self._active = False
            gutter, message = self._gutter, self._message
            try:
                stream.write("\r\x1b[K")
                stream.write(
                    f"  {_paint(gutter, 'dim')}  {_paint(marker, colour)}  "
                    f"{_paint(message, 'text')}\n"
                )
                stream.flush()
            except Exception:
                pass

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.3)
            self._thread = None
        if self._tty:
            try:
                self._stream.write("\r\x1b[K")
                self._stream.flush()
            except Exception:
                pass


def _say(marker: str, message: str, colour: str = "dim", step: int | None = None) -> None:
    """One line of the load report: optional step number, marker, message.

    The number is what turns a wall of identical arrows into a sequence you can
    follow — it also tells you how far in you are when a step hangs.
    """
    gutter = f"{step:02d}" if step is not None else "  "
    print(
        f"  {_paint(gutter, 'dim')}  {_paint(marker, colour)}  {_paint(message, 'text')}",
        file=sys.__stderr__,
        flush=True,
    )


def _preload_local_model(checkpoint_arg: str, device: str):
    """Load the checkpoint before Textual starts, with plain terminal output.

    A 1 GB native load can block the whole process for a long time on some
    machines. Doing that inside the TUI freezes its event loop — no repaint, no
    timers, nothing. Before the UI exists there is no event loop to freeze, so a
    slow load just looks like a slow start.

    Returns a ready ChatManager, or None if there is nothing to preload.
    """
    spinner: _StepSpinner | None = None
    try:
        from dizel_core.chat_manager import ChatManager
        from dizel_core.config_manager import ConfigManager

        cfg = ConfigManager.load()
        router = cfg.get("api_router") or {}
        if (router.get("provider") or "local") != "local":
            return None

        path = checkpoint_arg or str(cfg.get("checkpoint") or "")
        if not path or not os.path.exists(path):
            return None

        if not checkpoint_arg:
            saved = cfg.get("device", "cpu")
            if saved in ("cpu", "cuda"):
                device = saved

        name = os.path.basename(path)
        width = 62
        label_width = 10
        started = time.monotonic()
        counter = {"n": 0}
        spinner = _StepSpinner()

        try:
            from dizel_core.version import __version__
        except Exception:
            __version__ = ""

        def rule() -> str:
            return f"  {_paint('─' * width, 'dim')}"

        def row(label: str, value: str) -> str:
            return (
                f"  {_paint(label.ljust(label_width), 'dim')}  "
                f"{_paint(value, 'text')}"
            )

        title = f"Dizel {__version__}".rstrip()
        left = f"◆  {title}"
        gap = max(1, width - len(left) - len("terminal"))

        print(file=sys.__stderr__)
        print(
            f"  {_paint(left, 'mark')}{' ' * gap}{_paint('terminal', 'dim')}",
            file=sys.__stderr__,
        )
        print(rule(), file=sys.__stderr__)
        print(file=sys.__stderr__)
        print(row("checkpoint", _shorten(name, width - label_width - 4)), file=sys.__stderr__)
        print(row("device", device), file=sys.__stderr__)
        print(file=sys.__stderr__, flush=True)

        def report(message: str) -> None:
            lowered = message.lower()

            # The header already names the file and the path is long enough to
            # wrap, so this step is dropped rather than repeated.
            if lowered.startswith("loading gguf model from"):
                return
            if "no gpu offload" in lowered:
                spinner.finish()
                counter["n"] += 1
                _say("!", "No GPU offload in this build — using CPU", "warn", counter["n"])
                return
            if "failed" in lowered or "error" in lowered:
                spinner.finish()
                counter["n"] += 1
                _say("!", message, "warn", counter["n"])
                return
            if lowered.startswith(("done loading", "model ready")):
                if lowered.startswith("done loading"):
                    size = re.search(r"\(([\d.]+ MB)\)", message)
                    if size:
                        counter["size"] = size.group(1)
                return

            counter["n"] += 1
            if spinner.tty:
                # The row above is finished; the new step animates until the
                # provider reports the next one.
                spinner.finish()
                spinner.begin(f"{counter['n']:02d}", message)
            else:
                _say("·", message, "dim", counter["n"])

        manager = ChatManager()
        manager.load_model(path, device, on_progress=report)

        spinner.finish()
        elapsed = time.monotonic() - started
        parts = [counter["size"]] if counter.get("size") else []
        parts += [f"{counter['n']} steps", f"{elapsed:.1f}s"]
        detail = " · " + " · ".join(parts)

        print(file=sys.__stderr__)
        print(rule(), file=sys.__stderr__)
        print(
            f"  {_paint('✓', 'ok')}  {_paint('ready', 'text')}{_paint(detail, 'dim')}",
            file=sys.__stderr__,
        )
        print(file=sys.__stderr__, flush=True)
        return manager
    except Exception as error:
        spinner.close()
        width = 62
        label_width = 10
        message = " ".join(str(error).split())
        if len(message) > width - label_width - 4:
            message = message[: width - label_width - 5].rstrip() + "…"
        print(file=sys.__stderr__)
        print(
            f"  {_paint('!', 'warn')}  {_paint('preload skipped', 'text')}",
            file=sys.__stderr__,
        )
        print(f"  {_paint('─' * width, 'dim')}", file=sys.__stderr__)
        print(
            f"  {_paint('error'.ljust(label_width), 'dim')}  {_paint(message, 'text')}",
            file=sys.__stderr__,
        )
        print(
            f"  {_paint('next'.ljust(label_width), 'dim')}  "
            f"{_paint('Settings → Runtime, or relaunch with --checkpoint', 'text')}",
            file=sys.__stderr__,
        )
        print(file=sys.__stderr__, flush=True)
        return None
    finally:
        if spinner is not None:
            spinner.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Dizel terminal client")
    parser.add_argument("--checkpoint", "-c", type=str, default="", help="Path to a checkpoint (.pt/.gguf/.safetensors)")
    parser.add_argument("--device", "-d", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--prompt", "-p", type=str, default="", help="Run one prompt headlessly and exit")
    parser.add_argument("--resume", action="store_true", help="Open the most recent saved chat")
    args = parser.parse_args()

    piped = ""
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()

    if piped or args.prompt:
        full_prompt = args.prompt
        if piped:
            full_prompt = f"{full_prompt}\n\n{piped}" if full_prompt else piped
        full_prompt = full_prompt.strip()
        if not full_prompt:
            return 0
        return _headless(full_prompt, args.checkpoint, args.device)

    # Interactive mode. ChatManager logs with print(); send those to a file so
    # they cannot paint over the screen. Textual renders via sys.__stdout__.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(repo_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    real_stdout = sys.stdout
    log_file = open(os.path.join(log_dir, "tui.log"), "a", buffering=1, encoding="utf-8")
    crash_file = open(
        os.path.join(log_dir, "tui_crash.log"), "a", buffering=1, encoding="utf-8"
    )
    try:
        faulthandler.enable(file=crash_file, all_threads=True)
    except Exception:
        pass
    try:
        sys.stdout = log_file
        _warm_local_runtime(args.checkpoint, repo_root)
        preloaded = _preload_local_model(args.checkpoint, args.device)
        resume_session = ""
        if args.resume:
            try:
                from dizel_core.history_manager import list_sessions

                sessions = list_sessions()
                if sessions:
                    resume_session = sessions[0].get("id", "")
            except Exception:
                pass
        try:
            DizelCMDApp(
                checkpoint=args.checkpoint,
                device=args.device,
                preloaded=preloaded,
                resume_session=resume_session,
            ).run()
        except Exception:
            crash_file.write(traceback.format_exc())
            crash_file.flush()
            raise
    finally:
        sys.stdout = real_stdout
        log_file.close()
        crash_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
