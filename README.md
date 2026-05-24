# prompt-diff

Diff two LLM message lists (system/user/assistant turns) to see exactly what changed between prompt versions. Zero runtime dependencies.

```bash
pip install prompt-diff
```

## Why

You tweaked the system prompt for your Hermes agent. The model's behavior changed. You want to know exactly what you changed, in context, the way `git diff` shows code changes.

`git diff` works on files but not on structured message lists. prompt-diff fills that gap.

## Python API

```python
from prompt_diff import diff_prompts, render_diff

old = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]
new = [
    {"role": "system", "content": "You are a helpful assistant. Be concise."},
    {"role": "user", "content": "What is the capital of France?"},
]

diff = diff_prompts(old, new)
print(render_diff(diff))
```

Output:

```
[system] (changed)
  --- system (old)
  +++ system (new)
  @@ -1 +1 @@
  -You are a helpful assistant.
  +You are a helpful assistant. Be concise.

```

## CLI

Pass two JSON files, each containing a message array:

```bash
python3 -m prompt_diff old_messages.json new_messages.json
python3 -m prompt_diff old.json new.json --show-same --color
```

Exit code 1 when there are differences (like `diff`), 0 when identical.

## What PromptDiff gives you

```python
diff = diff_prompts(old, new)

diff.any_changed       # bool
diff.changed_count     # int
diff.added_count       # int
diff.removed_count     # int
diff.same_count        # int

for md in diff.messages:
    md.index           # 0-based message position
    md.role            # "system", "user", etc. (or "user → assistant" on role change)
    md.status          # "same" | "changed" | "added" | "removed"
    md.old_content     # str or None
    md.new_content     # str or None
    md.unified_lines   # list[str] (unified diff format, for changed messages)
```

## Handles Anthropic content blocks

Messages with `"content": [{"type": "text", "text": "..."}]` are normalized to strings before diffing, so prompt-diff works with both OpenAI-style string content and Anthropic-style content block arrays.

## Render options

```python
render_diff(diff, show_same=True)   # also print unchanged messages
render_diff(diff, color=True)       # ANSI color for +/- lines
render_diff(diff, show_same=False)  # default: only show what changed
```

## Testing

```bash
PYTHONPATH=src python3 -m pytest tests/ -q
# 22 passed
```

Zero runtime dependencies. Python 3.10+. MIT license.
