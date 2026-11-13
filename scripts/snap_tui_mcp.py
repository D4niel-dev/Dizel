"""Headless snapshot of the /mcp modal."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.tui.app import DizelCMDApp
from inference.tui.panels.modals import McpModal

OUT = Path(__file__).parent / ".tui-mcp-snapshot.svg"


async def main() -> None:
    app = DizelCMDApp()
    async with app.run_test(size=(150, 42)) as pilot:
        await pilot.pause()
        app.push_screen(McpModal())
        await pilot.pause()
        OUT.write_text(app.export_screenshot(), encoding="utf-8")
        print("wrote", OUT)


asyncio.run(main())
