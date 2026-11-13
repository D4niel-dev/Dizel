"""Snapshot the TUI with a conversation in it, not just the empty state."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.tui.app import DizelCMDApp
from inference.tui.panels.workspace import WorkspacePanel

REPLY = """Here's a compact version:

```python
def reverse(s: str) -> str:
    return s[::-1]
```

The slice steps backwards through the string, so it reads in reverse."""


async def main() -> None:
    app = DizelCMDApp()
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        ws = app.query_one(WorkspacePanel)
        ws.clear_workspace(show_empty=False)

        ws.add_user_message("Write a python function that reverses a string.")
        await pilot.pause(0.1)

        ws.start_assistant_message()
        message = ws.current_message
        for chunk in REPLY.splitlines(keepends=True):
            message.stream.append(chunk)
            message.update_stream()
            await pilot.pause(0.02)
        message.stream.finish()
        message.finalize(REPLY)
        ws.current_message = None
        await pilot.pause(0.8)

        out = Path(__file__).parent / ".tui-conv.svg"
        out.write_text(app.export_screenshot(), encoding="utf-8")
        print("wrote", out)


asyncio.run(main())
