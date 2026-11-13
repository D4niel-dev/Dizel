from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dizel_core.feedback_store import record_feedback


class FeedbackStoreTests(unittest.TestCase):
    def test_writes_one_jsonl_event(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.dict(os.environ, {"DIZEL_DATA_DIR": temp}):
                path = record_feedback(
                    rating="up",
                    session_id="session-1",
                    message_key="0:abc",
                    reasons=["  Accurate  ", "", "Clear"],
                    note="  useful answer  ",
                    response="Hello",
                    prompt="Hi",
                    model="Dizel 1.5 Distill",
                    provider="local",
                )
            self.assertEqual(path, Path(temp) / "feedback" / "feedback.jsonl")
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry["rating"], "up")
            self.assertEqual(entry["session_id"], "session-1")
            self.assertEqual(entry["reasons"], ["Accurate", "Clear"])
            self.assertEqual(entry["note"], "useful answer")
            self.assertEqual(entry["response"], "Hello")
            self.assertEqual(entry["prompt"], "Hi")

    def test_accepts_removal_and_appends(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.dict(os.environ, {"DIZEL_DATA_DIR": temp}):
                record_feedback(rating="down", session_id="s", message_key="0:x")
                path = record_feedback(rating="removed", session_id="s", message_key="0:x")
            entries = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([entry["rating"] for entry in entries], ["down", "removed"])

    def test_rejects_unknown_rating(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.dict(os.environ, {"DIZEL_DATA_DIR": temp}):
                with self.assertRaises(ValueError):
                    record_feedback(rating="sideways")


if __name__ == "__main__":
    unittest.main()
