"""
One-off: recompute auto-derived chat titles with the new word-boundary rule.

Only sessions whose current title is *exactly* the old auto rule
(`first_user_message[:45].strip()`) are touched, so anything renamed by hand is
left alone. The old title is always recomputable, so this is lossless — and a
full backup is written before anything changes.

    python scripts/retitle_sessions.py          # dry run
    python scripts/retitle_sessions.py --apply  # write
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dizel_core.history_manager import HISTORY_DIR, derive_title  # noqa: E402


def main() -> None:
    apply = "--apply" in sys.argv
    history = Path(HISTORY_DIR)
    files = sorted(history.glob("*.json"))

    changes = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        first_user = next(
            (m.get("content", "") for m in data.get("messages", []) if m.get("role") == "user"),
            "",
        )
        if not first_user:
            continue

        old_rule = first_user[:45].strip()
        if data.get("title", "") != old_rule:
            continue  # renamed by hand, or already migrated

        new_title = derive_title(first_user)
        if new_title and new_title != data.get("title"):
            changes.append((path, data.get("title", ""), new_title))

    print(f"{len(files)} sessions scanned, {len(changes)} auto-titled")
    for _, old, new in changes[:6]:
        print(f"  {old!r}  ->  {new!r}")

    if not apply:
        print("\nDry run. Re-run with --apply to write.")
        return

    backup = history.parent / "history_backup_before_retitle"
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(history, backup)

    for path, _, new_title in changes:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["title"] = new_title
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nRewrote {len(changes)} titles. Backup: {backup}")


if __name__ == "__main__":
    main()
