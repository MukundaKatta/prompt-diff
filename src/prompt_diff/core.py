"""Core diff engine for LLM message lists."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence


class PromptDiffError(Exception):
    """Base exception for prompt-diff failures."""


@dataclass
class MessageDiff:
    """Diff result for a single message slot.

    Attributes:
        index: 0-based position in the message list.
        role: the role from old or new (or "ADDED" / "REMOVED").
        status: "same", "changed", "added", "removed".
        old_content: original content string (None if added).
        new_content: new content string (None if removed).
        unified_lines: unified diff lines (empty if same or no content to diff).
    """

    index: int
    role: str
    status: str  # "same" | "changed" | "added" | "removed"
    old_content: str | None = None
    new_content: str | None = None
    unified_lines: list[str] = field(default_factory=list)


@dataclass
class PromptDiff:
    """Result of diffing two LLM message lists.

    Attributes:
        messages: per-message diff results.
        any_changed: True if at least one message differs.
        added_count: number of added messages.
        removed_count: number of removed messages.
        changed_count: number of changed messages.
        same_count: number of identical messages.
    """

    messages: list[MessageDiff]
    any_changed: bool = False
    added_count: int = 0
    removed_count: int = 0
    changed_count: int = 0
    same_count: int = 0


def _extract_content(msg: dict[str, Any]) -> str:
    """Return a normalized string content from a message dict.

    Handles:
        - {"content": "string"}
        - {"content": [{"type": "text", "text": "..."}]}   (Anthropic content blocks)
        - {"content": [{"type": "text", "text": "..."}]}   (OpenAI messages)
        - Falls back to json.dumps of the content value.
    """
    raw = msg.get("content", "")
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, dict):
                t = block.get("text") or block.get("value") or block.get("content", "")
                parts.append(str(t) if t else json.dumps(block))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return json.dumps(raw)


def _extract_role(msg: dict[str, Any]) -> str:
    return str(msg.get("role", "unknown"))


def _unified_diff(old: str, new: str, role: str, context: int = 3) -> list[str]:
    """Return unified diff lines (without the trailing newline on each)."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    lines = list(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"{role} (old)",
        tofile=f"{role} (new)",
        n=context,
    ))
    return lines


def diff_prompts(
    old: Sequence[dict[str, Any]],
    new: Sequence[dict[str, Any]],
    *,
    context: int = 3,
) -> PromptDiff:
    """Diff two LLM message lists.

    Messages are matched positionally. If the lists have different lengths,
    the shorter one is padded with "removed" or "added" entries.

    Args:
        old: the baseline message list.
        new: the updated message list.
        context: number of unchanged context lines around each change in the
            unified diff (default 3, matching standard diff tools).

    Returns:
        PromptDiff with per-message results and summary counts.
    """
    results: list[MessageDiff] = []
    added = removed = changed = same = 0

    max_len = max(len(old), len(new))

    for i in range(max_len):
        if i >= len(old):
            # New message was added
            msg = new[i]
            role = _extract_role(msg)
            content = _extract_content(msg)
            results.append(MessageDiff(
                index=i,
                role=role,
                status="added",
                old_content=None,
                new_content=content,
                unified_lines=[],
            ))
            added += 1
        elif i >= len(new):
            # Old message was removed
            msg = old[i]
            role = _extract_role(msg)
            content = _extract_content(msg)
            results.append(MessageDiff(
                index=i,
                role=role,
                status="removed",
                old_content=content,
                new_content=None,
                unified_lines=[],
            ))
            removed += 1
        else:
            o_msg = old[i]
            n_msg = new[i]
            o_role = _extract_role(o_msg)
            n_role = _extract_role(n_msg)
            o_content = _extract_content(o_msg)
            n_content = _extract_content(n_msg)

            # Roles changed is treated as a changed message too.
            role = n_role if o_role == n_role else f"{o_role} → {n_role}"

            if o_content == n_content and o_role == n_role:
                results.append(MessageDiff(
                    index=i,
                    role=role,
                    status="same",
                    old_content=o_content,
                    new_content=n_content,
                    unified_lines=[],
                ))
                same += 1
            else:
                diff_lines = _unified_diff(o_content, n_content, role, context=context)
                results.append(MessageDiff(
                    index=i,
                    role=role,
                    status="changed",
                    old_content=o_content,
                    new_content=n_content,
                    unified_lines=diff_lines,
                ))
                changed += 1

    return PromptDiff(
        messages=results,
        any_changed=(added + removed + changed) > 0,
        added_count=added,
        removed_count=removed,
        changed_count=changed,
        same_count=same,
    )


def render_diff(
    diff: PromptDiff,
    *,
    show_same: bool = False,
    color: bool = False,
) -> str:
    """Render a PromptDiff as a human-readable string.

    Args:
        diff: result from diff_prompts().
        show_same: if True, also print unchanged messages.
        color: if True, wrap + lines in ANSI green and - lines in ANSI red.

    Returns:
        A multi-line string suitable for printing.
    """
    lines: list[str] = []
    GREEN = "\033[32m" if color else ""
    RED = "\033[31m" if color else ""
    RESET = "\033[0m" if color else ""

    for md in diff.messages:
        if md.status == "same" and not show_same:
            continue

        header = f"[{md.role}] ({md.status})"
        lines.append(header)

        if md.status == "added":
            for ln in (md.new_content or "").splitlines():
                lines.append(f"  {GREEN}+ {ln}{RESET}")
        elif md.status == "removed":
            for ln in (md.old_content or "").splitlines():
                lines.append(f"  {RED}- {ln}{RESET}")
        elif md.status == "same":
            for ln in (md.old_content or "").splitlines()[:5]:
                lines.append(f"    {ln}")
            rest = len((md.old_content or "").splitlines()) - 5
            if rest > 0:
                lines.append(f"    ... ({rest} more lines)")
        else:  # changed
            for ln in md.unified_lines:
                ln_clean = ln.rstrip("\n")
                if color and ln_clean.startswith("+") and not ln_clean.startswith("+++"):
                    lines.append(f"  {GREEN}{ln_clean}{RESET}")
                elif color and ln_clean.startswith("-") and not ln_clean.startswith("---"):
                    lines.append(f"  {RED}{ln_clean}{RESET}")
                else:
                    lines.append(f"  {ln_clean}")

        lines.append("")

    if not lines:
        lines.append("(no differences)")

    return "\n".join(lines)
