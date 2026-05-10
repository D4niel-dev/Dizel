import argparse
import sys
import os
from inference.cmd_tui.app import DizelCMDApp
from inference.dizel_gui.logic.chat_manager import ChatManager
import threading
import time

def main():
    parser = argparse.ArgumentParser(description="Dizel CMD UI")
    parser.add_argument("--checkpoint", "-c", type=str, default="", help="Path to .pt model checkpoint")
    parser.add_argument("--device", "-d", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--prompt", "-p", type=str, default="", help="Optional prompt for headless scriptable mode")
    args = parser.parse_args()

    piped_input = ""
    if not sys.stdin.isatty():
        piped_input = sys.stdin.read().strip()
        # Re-open stdin to the terminal if we still want to launch the UI? 
        # No, if there is piped input, we assume headless mode.

    if piped_input or args.prompt:
        # Headless Scriptable Mode
        full_prompt = args.prompt
        if piped_input:
            full_prompt += ("\n\n" + piped_input) if full_prompt else piped_input
        full_prompt = full_prompt.strip()
        
        if not full_prompt:
            return

        chat_mgr = ChatManager()
        if args.checkpoint:
            chat_mgr.load_model(args.checkpoint, args.device)
            # Wait a bit for async load to finish if local model. 
            # (Note: robust implementations would wait properly, we sleep briefly here as ChatManager manages its own state)
            time.sleep(2)

        sys.stdout.write("\n")
        sys.stdout.flush()

        done_event = threading.Event()

        def on_token(token: str):
            sys.stdout.write(token)
            sys.stdout.flush()

        def on_done(text: str):
            sys.stdout.write("\n\n")
            sys.stdout.flush()
            done_event.set()

        def on_error(err: str):
            sys.stderr.write(f"\n[Error] {err}\n")
            sys.stderr.flush()
            done_event.set()

        chat_mgr.send_message(
            user_text=full_prompt,
            attachments=[],
            on_token=on_token,
            on_done=on_done,
            on_error=on_error
        )

        done_event.wait()
    else:
        # Interactive TUI Mode
        app = DizelCMDApp(checkpoint=args.checkpoint, device=args.device)
        app.run()

if __name__ == "__main__":
    main()
