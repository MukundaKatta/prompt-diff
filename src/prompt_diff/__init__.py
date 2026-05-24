"""prompt-diff: diff two LLM message lists to see exactly what changed.

Public API:
    diff_prompts(old, new, *, context=3) -> PromptDiff
    render_diff(diff, *, show_same=False, color=False) -> str
    PromptDiff      — dataclass: messages, any_changed, added/removed/changed/same counts
    MessageDiff     — per-message result: index, role, status, old/new content, unified_lines
    PromptDiffError — base exception
"""

from .core import (
    MessageDiff,
    PromptDiff,
    PromptDiffError,
    diff_prompts,
    render_diff,
)

__all__ = [
    "diff_prompts",
    "render_diff",
    "PromptDiff",
    "MessageDiff",
    "PromptDiffError",
]
__version__ = "0.1.0"
