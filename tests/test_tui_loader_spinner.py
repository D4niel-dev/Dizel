"""The TUI preload spinner animates the current step and finalises it cleanly."""

from __future__ import annotations

import time
import unittest

from inference.tui.main import _StepSpinner


class FakeStream:
    encoding = "utf-8"

    def __init__(self, tty: bool = True) -> None:
        self._tty = tty
        self.parts: list[str] = []

    def isatty(self) -> bool:
        return self._tty

    def write(self, text: str) -> None:
        self.parts.append(text)

    def flush(self) -> None:
        pass

    def text(self) -> str:
        return "".join(self.parts)


class StepSpinnerTests(unittest.TestCase):
    def test_animates_then_finalises_the_step(self):
        stream = FakeStream()
        spinner = _StepSpinner(stream)

        spinner.begin("01", "Loading GGUF model")
        time.sleep(0.28)
        spinner.finish()
        spinner.begin("02", "Reading the weights")
        time.sleep(0.2)
        spinner.close()

        output = stream.text()
        self.assertIn("\r", output)
        self.assertIn("01  ·  Loading GGUF model", output)
        self.assertIn("02", output)
        self.assertIn("⠋", output)
        self.assertIsNone(spinner._thread)

    def test_non_tty_stays_static(self):
        stream = FakeStream(tty=False)
        spinner = _StepSpinner(stream)
        spinner.begin("01", "Loading GGUF model")
        spinner.finish()
        spinner.close()
        self.assertEqual(stream.text(), "")


if __name__ == "__main__":
    unittest.main()
