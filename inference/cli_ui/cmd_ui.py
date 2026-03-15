"""
inference/cmd_ui.py — Local CLI chat interface for Dizel.

Supports three modes
--------------------
  --mode chat       Interactive multi-turn conversation (default)
  --mode complete   Raw next-token completion from a prompt
  --mode json       Force structured JSON output mode

Usage
-----
    # Interactive chat (SFT checkpoint recommended)
    python inference/cmd_ui.py --checkpoint checkpoints/dizel-sft-best.pt

    # One-shot completion
    python inference/cmd_ui.py --checkpoint checkpoints/dizel-pretrain-best.pt \\
        --mode complete --prompt "The speed of light is"

    # JSON mode
    python inference/cmd_ui.py --checkpoint checkpoints/dizel-sft-best.pt \\
        --mode json --prompt "Tell me about photosynthesis"

    # Greedy decoding (deterministic)
    python inference/cmd_ui.py --checkpoint checkpoints/dizel-sft-best.pt \\
        --temperature 0.0

Keyboard shortcuts in chat mode
---------------------------------
    /quit  or  /exit   — Exit chat
    /new               — Clear conversation history
    /system <text>     — Set a new system prompt
    /info              — Show model info
    /temp  <float>     — Adjust temperature
    /help              — Show this help message
    /clear             — Clear conversation history
    /save <filename>   — Save conversation history
    /load <filename>   — Load conversation history
    /list              — List conversation history
    /delete <filename> — Delete conversation history
    /delete all        — Delete all conversation history
"""


from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.align import Align
from rich.rule import Rule

console = Console()
console.clear()

import argparse
import json
import os
import sys
import time
import textwrap
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import CONFIG
from model.architecture import DizelLM
from model.dizel_info import MODEL_NAME, APP_NAME, VERSION, MODEL_SIZE, CONTEXT_LENGTH, VOCAB_SIZE, AUTHOR, BUILD, DESCRIPTION
from training.dataset import Tokenizer, ROLE_TOKENS, END_TOKEN
import threading
import itertools
from rich.live import Live
from rich.text import Text

def thinking_indicator(stop_event):
    with Live(refresh_per_second=4, transient=True) as live:
        for dots in itertools.cycle([".", "..", "..."]):
            if stop_event.is_set():
                break
            live.update(Text(f"Dizel is thinking{dots}", style="bold cyan"))
            time.sleep(0.4)

# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, device: str) -> tuple:
    """Load model and return (model, model_cfg)."""
    if not os.path.exists(checkpoint_path):
        console.print(f"[red]ERROR: checkpoint not found at '{checkpoint_path}'[/red]")
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_cfg", CONFIG.model)

    model = DizelLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    step     = ckpt.get("step", "?")
    val_loss = ckpt.get("val_loss", float("inf"))
    return model, model_cfg, step, val_loss, checkpoint_path


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Dizel, a structured analytical AI model. "
    "Prioritize clarity, precision, and logical organization. "
    "Use structured formatting when appropriate. "
    "Avoid unnecessary verbosity. "
    "If uncertain, explicitly state limitations."
)


def build_chat_prompt(messages: list, tokenizer: Tokenizer) -> list:
    """
    Format a list of {"role": …, "content": …} dicts into token ids
    and append the assistant role token so the model continues as Dizel.
    """
    ids = []
    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        ids += tokenizer.encode(ROLE_TOKENS[role])
        ids += tokenizer.encode(content)
        ids += tokenizer.encode(END_TOKEN)

    # Prime the model to produce an assistant reply
    ids += tokenizer.encode(ROLE_TOKENS["assistant"])
    return ids


def build_completion_prompt(text: str, tokenizer: Tokenizer) -> list:
    return [tokenizer.bos_id] + tokenizer.encode(text)


# ---------------------------------------------------------------------------
# Generation with streaming output
# ---------------------------------------------------------------------------
@torch.no_grad()
def stream_generate(
    model: DizelLM,
    tokenizer: Tokenizer,
    prompt_ids: list,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    end_tokens: list = None,
    show_prefix=True,
    stream_output=True,
) -> str:
    """
    Generate tokens one at a time, printing each as it is produced
    (streaming output to terminal).
    """
    if end_tokens is None:
        end_tokens = [tokenizer.eos_id]

    # Also stop at <|end|> token to prevent it leaking into output
    end_token_ids = tokenizer.encode(END_TOKEN)
    for etid in end_token_ids:
        if etid not in end_tokens:
            end_tokens.append(etid)

    ctx = model.cfg.context_length
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_ids = []
    console.print("\n[bold cyan]Dizel[/bold cyan]: ", end="")

    for _ in range(max_new_tokens):
        # Truncate to context window
        input_ids = idx[:, -ctx:]

        with torch.amp.autocast(
            device_type=device if device != "cpu" else "cpu",
            enabled=(device == "cuda"),
            dtype=torch.bfloat16,
        ):
            logits, _ = model(input_ids)
        logits = logits[:, -1, :].float()

        # Repetition penalty
        if repetition_penalty != 1.0:
            for tid in set(generated_ids[-64:]):   # only penalise recent tokens
                if logits[0, tid] < 0:
                    logits[0, tid] *= repetition_penalty
                else:
                    logits[0, tid] /= repetition_penalty

        # Greedy if temperature == 0
        if temperature == 0.0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-8)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p < 1.0:
                probs_s, s_idx = torch.sort(
                    F.softmax(logits, dim=-1), descending=True
                )
                cum = probs_s.cumsum(-1)
                probs_s[cum - probs_s > top_p] = 0.0
                probs_s /= probs_s.sum(-1, keepdim=True)
                next_id = torch.multinomial(probs_s, 1)
                next_id = s_idx.gather(-1, next_id)
            else:
                probs   = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)

        tid = next_id.item()
        if tid in end_tokens:
            break

        generated_ids.append(tid)
        idx = torch.cat([idx, next_id], dim=1)

        # Decode and print the new token (handles multi-byte UTF-8 gracefully)
        token_text = tokenizer.sp.id_to_piece(tid)

        # SentencePiece uses ▁ for word-boundary spaces
        token_text = token_text.replace("▁", " ")

        # Filter out special tokens and <unk> from streaming output
        if token_text in ("<unk>", "⁇"):
            continue
        for special in list(ROLE_TOKENS.values()) + [END_TOKEN]:
            token_text = token_text.replace(special, "")
        if not token_text:
            continue

        if stream_output:
            console.print(token_text, end="", highlight=False)

    if stream_output:   
        print()   # newline after generation
        
    full_text = tokenizer.decode(generated_ids)
    return full_text, len(generated_ids)


# ---------------------------------------------------------------------------
# JSON generation helper
# ---------------------------------------------------------------------------
def generate_json(
    model, tokenizer, messages, device, args
) -> dict:
    """
    Append a [json] instruction and try to parse the output as JSON.
    Falls back to returning a raw string if parsing fails.
    """
    # Add json signal to last user message
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] += " [json]"

    prompt_ids = build_chat_prompt(messages, tokenizer)
    text, _ = stream_generate(
        model, tokenizer, prompt_ids, device,
        max_new_tokens=300,
        temperature=0.3,   # lower temperature for structured output
        top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    # Try to extract JSON from the response
    try:
        # Find the first { or [ and try to parse from there
        start = min(
            (text.find("{") if "{" in text else len(text)),
            (text.find("[") if "[" in text else len(text)),
        )
        return json.loads(text[start:])
    except json.JSONDecodeError:
        return {"raw": text, "error": "Could not parse JSON"}


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------
def chat_loop(model, tokenizer, device, args) -> None:
    system_prompt = SYSTEM_PROMPT
    debug_mode = False
    history = []   # list of {"role": ..., "content": ...}

    while True:
        try:
            user_input = console.input("\n[bold green]You[/bold green]: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold cyan]Dizel[/bold cyan]: Goodbye!")
            break

        if not user_input:
            continue

        # ── Commands ─────────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit"):
                console.print("\n[bold cyan]Dizel[/bold cyan]: Goodbye!")
                break
            elif cmd == "/new":
                history.clear()
                console.print("[bold cyan]Dizel[/bold cyan]: Conversation cleared.\n")
            elif cmd.startswith("/system "):
                system_prompt = user_input[8:].strip()
                history.clear()
                console.print("[bold cyan]Dizel[/bold cyan]: System prompt updated. Conversation cleared.\n")
            elif cmd == "/info":
                table = Table(title="Model Info", show_header=True, header_style="bold cyan")
                table.add_column("Property")
                table.add_column("Value")

                table.add_row("Model", str(model))
                table.add_row("Parameters", f"{model.num_parameters()/1e6:.2f} M")
                table.add_row("Device", device)
                table.add_row("Temperature", str(args.temperature))
                table.add_row("Top-K", str(args.top_k))
                table.add_row("Top-P", str(args.top_p))
                table.add_row("Repetition Penalty", str(args.repetition_penalty))

                console.print(table)
                console.print()
            elif cmd.startswith("/temp "):
                try:
                    args.temperature = float(user_input.split()[1])
                    console.print(f"[bold cyan]Dizel[/bold cyan]: Temperature set to {args.temperature}\n")
                except (IndexError, ValueError):
                    console.print("[bold cyan]Dizel[/bold cyan]: Usage: /temp 0.8\n")
            elif cmd.startswith("/save"):
                filename = user_input.split(maxsplit=1)[1] if len(user_input.split()) > 1 else "session.json"
                session_data = {"system_prompt": system_prompt, "history": history}
                
                with open(filename, "w") as f:
                    json.dump(session_data, f, indent=2)
                console.print(f"[cyan]Session saved to {filename}[/cyan]\n")
            elif cmd.startswith("/load"):
                filename = user_input.split(maxsplit=1)[1] if len(user_input.split()) > 1 else "session.json"
                
                if not os.path.exists(filename):
                    console.print("[red]File not found.[/red]\n")
                else:
                    with open(filename, "r") as f:
                        session_data = json.load(f)
                        system_prompt = session_data.get("system_prompt", SYSTEM_PROMPT)
                        history = session_data.get("history", [])
                        console.print(f"[cyan]Session loaded from {filename}[/cyan]\n")
            elif cmd == "/debug":
                debug_mode = not debug_mode
                console.print(f"[cyan]Debug mode {'enabled' if debug_mode else 'disabled'}[/cyan]\n")
            else:
                console.print("[bold cyan]Dizel[/bold cyan]: Unknown command. "
                      "Try /quit /new /system /info /temp /save /load\n")
            continue

        # ── JSON mode ─────────────────────────────────────────────────────
        if args.mode == "json" or "[json]" in user_input:
            messages = (
                [{"role": "system", "content": system_prompt}]
                + history
                + [{"role": "user", "content": user_input}]
            )
            result = generate_json(model, tokenizer, messages, device, args)
            print(f"JSON: {json.dumps(result, indent=2)}\n")
            history += [
                {"role": "user",      "content": user_input},
                {"role": "assistant", "content": json.dumps(result)},
            ]
            continue

        # ── Normal chat ───────────────────────────────────────────────────
        history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": system_prompt}] + history

        # Keep context window manageable: trim old turns
        # (keep system + last N turns)
        max_turns = 6
        if len(messages) > max_turns * 2 + 1:
            messages = [messages[0]] + messages[-(max_turns * 2):]

        prompt_ids = build_chat_prompt(messages, tokenizer)
        if len(prompt_ids) > model.cfg.context_length - args.max_new_tokens:
            # Too long — drop oldest history
            history = history[-4:]
            messages = [{"role": "system", "content": system_prompt}] + history
            prompt_ids = build_chat_prompt(messages, tokenizer)

        t_start = time.time()

        if args.thinking:

            # ── First pass: reasoning ─────────────────────
            reasoning_prompt = messages + [{"role": "assistant", "content": "Provide structured internal reasoning only."}]
            reasoning_ids = build_chat_prompt(reasoning_prompt, tokenizer)
            
            stop_event = threading.Event()
            thread = threading.Thread(target=thinking_indicator, args=(stop_event,))
            thread.start()
            
            reasoning, _ = stream_generate(
                model, tokenizer, reasoning_ids, device,
                max_new_tokens=120,
                temperature=0.6,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                show_prefix=False,
                stream_output=False,
            )

            stop_event.set()
            thread.join()
            
            if debug_mode:
                console.print(
                    Panel(
                        reasoning.strip(),
                        title="Thoughts",
                        border_style="dim",
                        expand=False
                    )
                )
                console.print()

            # ── Final answer ─────────────────────────────────
            console.print("[bold cyan]Dizel[/bold cyan]: ", end="")

            response, token_count = stream_generate(
                model, tokenizer, prompt_ids, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                show_prefix=True,
            )

            t_end = time.time()
        else:
            response, token_count = stream_generate(
                model, tokenizer, prompt_ids, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                show_prefix=True,
            )
            t_end = time.time()

        # Clean up role tokens that leaked into the response
        for tok in list(ROLE_TOKENS.values()) + [END_TOKEN]:
            response = response.replace(tok, "")
        response = response.strip()

        history.append({"role": "assistant", "content": response})
        elapsed = t_end - t_start
        tps = token_count / max(elapsed, 1e-6)
        
        console.print(f"[dim]Response time: {elapsed:.2f}s  |  Tokens: {token_count}  |  {tps:.2f} tok/s[/dim]\n")
        console.print(Rule(style="dim"))
        console.print()

# ---------------------------------------------------------------------------
# Completion mode
# ---------------------------------------------------------------------------
def completion_mode(model, tokenizer, device, args) -> None:
    print(f"Prompt: {args.prompt}\n")
    prompt_ids = build_completion_prompt(args.prompt, tokenizer)
    stream_generate(
        model, tokenizer, prompt_ids, device,
        max_new_tokens     = args.max_new_tokens,
        temperature        = args.temperature,
        top_k              = args.top_k,
        top_p              = args.top_p,
        repetition_penalty = args.repetition_penalty,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dizel local inference")
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path to .pt checkpoint")
    p.add_argument("--mode",        type=str, default="chat",
                   choices=["chat", "complete", "json"])
    p.add_argument("--prompt",      type=str, default="",
                   help="Prompt for complete/json modes")
    p.add_argument("--max_new_tokens", type=int,   default=200)
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--top_k",          type=int,   default=50)
    p.add_argument("--top_p",          type=float, default=0.92)
    p.add_argument("--repetition_penalty", type=float, default=1.15)
    p.add_argument("--device",         type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--thinking", action="store_true", help="Enable thinking mode (shows reasoning before answer)")
    return p.parse_args()


def ui():
    width = console.size.width
    #heigth = console.size.height

    if console.size.width > 120:
        console.print("[bold yellow]Tip: Best wiewed in a smaller terminal window.[/bold yellow]")

    header = Rule(Align.center(
            f"\n[bold cyan]{MODEL_NAME} — {VERSION} — {DESCRIPTION}[/bold cyan]"
    ), style="cyan")

    info = Panel(Align.center(
        f"[dim]Build: {BUILD}[/dim]\n"
        f"[dim]Author: {AUTHOR}[/dim]\n"
        f"[dim]Context Length: {CONTEXT_LENGTH}[/dim]\n"
        f"[dim]Vocabulary Size: {VOCAB_SIZE}[/dim]\n"
        f"[dim]Model Size: {MODEL_SIZE}[/dim]\n"
        f"[dim]Model: {APP_NAME}[/dim]"
    ))

    commands = Panel(Align.center(
            "[bold]Commands[/bold]\n"
            "/quit  /new  /system <text>  /info  /temp /save <file> /load <file> <f>"
    ), border_style="cyan")

    return Group(header, info, commands)

def main() -> None:
    console.clear()
    args = parse_args()
    device = args.device

    model, model_cfg, step, val_loss, checkpoint_path = load_model(args.checkpoint, device)
    tokenizer = Tokenizer()

    with Live(ui(), console=console, refresh_per_second=4) as live:
        time.sleep(0.25)
        live.update(ui())

    if args.mode == "chat":
        chat_loop(model, tokenizer, device, args)
    elif args.mode == "complete":
        if not args.prompt:
            args.prompt = input("Prompt: ").strip()
        completion_mode(model, tokenizer, device, args)
    elif args.mode == "json":
        if not args.prompt:
            args.prompt = input("Prompt: ").strip()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": args.prompt},
        ]
        result = generate_json(model, tokenizer, messages, device, args)
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
