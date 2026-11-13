"""Headless snapshot of the TUI so it can be reviewed as an image."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.tui.app import DizelCMDApp

OUT = Path(__file__).parent / ".tui-snapshot.svg"


async def main() -> None:
    app = DizelCMDApp()
    async with app.run_test(size=(150, 42)) as pilot:
        await pilot.pause()
        OUT.write_text(app.export_screenshot(), encoding="utf-8")
        print("wrote", OUT)


asyncio.run(main())
