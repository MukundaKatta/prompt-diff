"""CLI: diff two JSONL files each containing a message list.

Usage:
    python3 -m prompt_diff old.json new.json
    python3 -m prompt_diff old.json new.json --show-same --color
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"prompt-diff: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"prompt-diff: {path}: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print(f"prompt-diff: {path}: expected a JSON array of messages", file=sys.stderr)
        sys.exit(1)
    return data


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="prompt-diff",
        description="Diff two LLM message lists (JSON arrays of role/content dicts).",
    )
    parser.add_argument("old", help="old messages file (JSON array)")
    parser.add_argument("new", help="new messages file (JSON array)")
    parser.add_argument("--show-same", action="store_true", help="also print unchanged messages")
    parser.add_argument("--color", action="store_true", help="colorize +/- lines with ANSI")
    parser.add_argument("--context", type=int, default=3, help="unified diff context lines (default 3)")
    args = parser.parse_args(argv)

    from . import diff_prompts, render_diff

    old = _load(args.old)
    new = _load(args.new)
    diff = diff_prompts(old, new, context=args.context)

    print(render_diff(diff, show_same=args.show_same, color=args.color))

    if diff.any_changed:
        summary = (
            f"changed={diff.changed_count} added={diff.added_count} "
            f"removed={diff.removed_count} same={diff.same_count}"
        )
        print(summary, file=sys.stderr)
        sys.exit(1)  # non-zero exit when there are differences, like `diff`
    else:
        print("(identical)", file=sys.stderr)


if __name__ == "__main__":
    main()
