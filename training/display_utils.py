"""
training/display_utils.py — Clean terminal display for Dizel training.

Old-style per-step log lines (no progress bar). Real wall-clock timers.
"""

import sys
import time
import shutil
from typing import Optional, Any

# Ensure non-ASCII glyphs (✓ ⚠ ✖) never crash on redirected stdout (cp1252)
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

_HAS_RICH = False
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _console = Console()
    _HAS_RICH = True
except ImportError:
    _console = None

# ── ANSI fallback colours ──────────────────────────────────────────

class _Style:
    RESET = '\033[0m'
    BOLD  = '\033[1m'
    DIM   = '\033[2m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE  = '\033[34m'
    RED   = '\033[31m'
    CYAN  = '\033[36m'

def _tw() -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 100


# ── Coloured messages ────────────────────────────────────────────

def info(text: str) -> None:
    """Blue info message."""
    if _HAS_RICH:
        _console.print(text, style="blue")
    else:
        print(f"  {_Style.BLUE}{text}{_Style.RESET}")

def ok(text: str) -> None:
    """Green success message."""
    if _HAS_RICH:
        _console.print(f"✓ {text}", style="bold green")
    else:
        print(f"  {_Style.GREEN}[OK]{_Style.RESET} {text}")

def warning(text: str) -> None:
    """Yellow warning."""
    if _HAS_RICH:
        _console.print(f"⚠ {text}", style="bold yellow")
    else:
        print(f"  {_Style.YELLOW}[!]{_Style.RESET} {text}")

def error(text: str) -> None:
    """Red error."""
    if _HAS_RICH:
        _console.print(f"✖ {text}", style="bold red")
    else:
        print(f"  {_Style.RED}[X]{_Style.RESET} {text}")

def dim(text: str) -> None:
    """Dim secondary text."""
    if _HAS_RICH:
        _console.print(text, style="dim white")
    else:
        print(f"  {_Style.DIM}{text}{_Style.RESET}")


# ── Section separators ───────────────────────────────────────────

def rule(title: str = "") -> None:
    """Horizontal rule with optional title."""
    if _HAS_RICH:
        _console.rule(title, style="bold cyan")
        print()
    else:
        w = min(_tw(), 100)
        if title:
            print(f"\n{_Style.BOLD}{_Style.CYAN}{'─' * w}{_Style.RESET}")
            print(f"  {_Style.BOLD}{title}{_Style.RESET}")
            print(f"{_Style.BOLD}{_Style.CYAN}{'─' * w}{_Style.RESET}\n")
        else:
            print(f"{_Style.DIM}{'─' * w}{_Style.RESET}")

def section(title: str) -> None:
    """Smaller section header (▸ Hardware)."""
    if _HAS_RICH:
        _console.print(f"\n[bold cyan]▸ {title}[/bold cyan]")
    else:
        print(f"\n  {_Style.BOLD}{_Style.CYAN}▸ {title}{_Style.RESET}")


# ── Config display ───────────────────────────────────────────────

def config_table(items: dict[str, Any]) -> None:
    """Aligned key-value config display."""
    if _HAS_RICH:
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2, 0, 0))
        table.add_column("key", style="cyan")
        table.add_column("value", style="bold white")
        for k, v in items.items():
            table.add_row(str(k), str(v))
        _console.print(table)
    else:
        pad = max((len(k) + 1 for k in items), default=10)
        for k, v in items.items():
            print(f"  {_Style.CYAN}{k+':':<{pad}}{_Style.RESET} {v}")


# ── Helpers for the step log ─────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    """Format seconds as '2h 13m', '5m 10s', '42s'."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _loss_style(loss: float) -> str:
    """ANSI colour for a loss value (green < 3, yellow < 5, red otherwise)."""
    if loss < 3.0:
        return _Style.GREEN
    if loss < 5.0:
        return _Style.YELLOW
    return _Style.RED


# ── Training step log (old-style, real timers) ───────────────────

class StepMeter:
    """
    Old-style per-step log lines — no progress bar, no rich Live.

    Prints one line per update in the classic Dizel style:

      step     1,234/10,000 | loss 2.3456 | lr 1.23e-04 | 42.7s/50st
             | 1.17 it/s | 12.5k tok/s | elapsed 2h 13m | ETA 1h 47m

    Every timer is a real wall-clock measurement:
      - window   : seconds since the previous update (or run start)
      - it/s     : steps in this window / window seconds
      - tok/s    : tokens in this window / window seconds (needs tokens_per_step)
      - elapsed  : total run time since StepMeter was created
      - ETA      : remaining steps / overall it/s

    Usage:
        meter = StepMeter(max_steps,
                          tokens_per_step=batch_size*grad_accum*ctx_len,
                          start_step=resume_step)
        ...
        if step % 50 == 0 or step == resume_step + 1:
            meter.update(step, loss=train_loss, lr=lr)
        ...
        meter.finalise()
    """

    def __init__(self, max_steps: int,
                 tokens_per_step: Optional[int] = None,
                 start_step: int = 0):
        self.max_steps = max_steps
        self.tokens_per_step = tokens_per_step
        self._step0 = start_step
        self._t_start = time.time()
        self._t_prev = None
        self._step_prev = start_step

    def update(self, step: int,
               loss: Optional[float] = None,
               val_loss: Optional[float] = None,
               lr: Optional[float] = None,
               acc: Optional[float] = None,
               extra: Optional[str] = None) -> None:
        now = time.time()
        window = (now - self._t_start) if self._t_prev is None else (now - self._t_prev)
        steps = max(step - self._step_prev, 1)
        self._t_prev = now
        self._step_prev = step

        it_s = steps / max(window, 1e-9)
        elapsed = now - self._t_start
        overall = (step - self._step0) / max(elapsed, 1e-9)
        remaining = max(self.max_steps - step, 0)
        eta = remaining / overall if overall > 0 else 0.0

        C = _Style.CYAN
        R = _Style.RESET
        D = _Style.DIM
        B = _Style.BOLD
        parts = [f"{D}step{R} {B}{C}{step:>7,}{R}/{self.max_steps:,}"]
        if loss is not None:
            parts.append(f"{D}loss{R} {_loss_style(loss)}{loss:.4f}{R}")
        if val_loss is not None:
            parts.append(f"{D}val{R} {_loss_style(val_loss)}{val_loss:.4f}{R}")
        if lr is not None:
            parts.append(f"{D}lr{R} {lr:.2e}")
        if acc is not None:
            parts.append(f"{D}acc{R} {_Style.GREEN}{acc:.2%}{R}")
        parts.append(f"{D}{window:.1f}s/{steps}st{R}")
        parts.append(f"{D}{it_s:.2f} it/s{R}")
        if self.tokens_per_step:
            tok_s = steps * self.tokens_per_step / max(window, 1e-9)
            tok_str = f"{tok_s/1e3:.1f}k" if tok_s >= 1e3 else f"{tok_s:.0f}"
            parts.append(f"{D}{tok_str} tok/s{R}")
        parts.append(f"{D}elapsed {_fmt_duration(elapsed)}{R}")
        if eta > 0:
            parts.append(f"{D}ETA {_fmt_duration(eta)}{R}")
        if extra:
            parts.append(f"{_Style.YELLOW}{extra}{R}")

        print("  " + " | ".join(parts))

    def finalise(self) -> None:
        """Print a summary line with total time and average speed."""
        elapsed = time.time() - self._t_start
        total = max(self.max_steps - self._step0, 1)
        avg = total / max(elapsed, 1e-9)
        print(f"  {_Style.GREEN}[OK]{_Style.RESET} done in {_fmt_duration(elapsed)} ({avg:.2f} it/s avg)")


# ── Validation marker (old style) ────────────────────────────────

def validation(step: int, val_loss: float, extra: Optional[str] = None) -> None:
    """Old-style validation marker line:  *** step 1000 | val_loss 1.8765 ***"""
    msg = f"*** step {step} | val_loss {val_loss:.4f}"
    if extra:
        msg += f" | {extra}"
    msg += " ***"
    print(f"  {_Style.BOLD}{_Style.YELLOW}{msg}{_Style.RESET}")


# ── Pipeline-level helpers ───────────────────────────────────────

def pipeline_header(title: str) -> None:
    """Big pipeline phase header."""
    if _HAS_RICH:
        from rich.panel import Panel
        from rich.align import Align
        _console.print()
        panel = Panel(
            f"[bold bright_magenta]{title}[/bold bright_magenta]",
            box=box.DOUBLE,
            border_style="bright_magenta",
            expand=False,
        )
        _console.print(Align.center(panel))
        _console.print()
    else:
        w = min(_tw(), 80)
        print(f"\n{'=' * w}")
        print(f"  {title}")
        print(f"{'=' * w}\n")

def pipeline_step(desc: str, status: str = "▶") -> None:
    """Pipeline step indicator."""
    if _HAS_RICH:
        _console.print(f"  [bold cyan][{status}][/bold cyan] {desc}")
    else:
        print(f"  [{status}] {desc}")

def pipeline_ok(desc: str) -> None:
    """Pipeline step complete."""
    if _HAS_RICH:
        _console.print(f"  [bold green]✓[/bold green] {desc}")
    else:
        print(f"  {_Style.GREEN}✓{_Style.RESET} {desc}")

def pipeline_info(desc: str) -> None:
    """Pipeline info."""
    if _HAS_RICH:
        _console.print(f"  [dim cyan]ℹ {desc}[/dim cyan]")
    else:
        print(f"  {desc}")
