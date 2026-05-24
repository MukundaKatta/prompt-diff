"""Tests for prompt_diff."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pytest
from prompt_diff import MessageDiff, PromptDiff, diff_prompts, render_diff

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# ---------------------------------------------------------------------------
# diff_prompts — basic cases
# ---------------------------------------------------------------------------

def test_identical_messages():
    old = [msg("system", "You are a helpful assistant."), msg("user", "Hello")]
    new = [msg("system", "You are a helpful assistant."), msg("user", "Hello")]
    d = diff_prompts(old, new)
    assert d.any_changed is False
    assert d.same_count == 2
    assert d.changed_count == 0


def test_content_changed():
    old = [msg("system", "Be helpful.")]
    new = [msg("system", "Be helpful. Be concise.")]
    d = diff_prompts(old, new)
    assert d.any_changed is True
    assert d.changed_count == 1
    assert d.messages[0].status == "changed"


def test_message_added():
    old = [msg("system", "Be helpful.")]
    new = [msg("system", "Be helpful."), msg("user", "Hello")]
    d = diff_prompts(old, new)
    assert d.added_count == 1
    assert d.messages[1].status == "added"
    assert d.messages[1].new_content == "Hello"


def test_message_removed():
    old = [msg("system", "Be helpful."), msg("user", "Hello")]
    new = [msg("system", "Be helpful.")]
    d = diff_prompts(old, new)
    assert d.removed_count == 1
    assert d.messages[1].status == "removed"
    assert d.messages[1].old_content == "Hello"


def test_role_changed():
    old = [msg("user", "Hello")]
    new = [msg("assistant", "Hello")]
    d = diff_prompts(old, new)
    assert d.changed_count == 1
    assert "→" in d.messages[0].role


def test_empty_both():
    d = diff_prompts([], [])
    assert d.any_changed is False
    assert d.same_count == 0


def test_empty_old_adds():
    new = [msg("system", "You are an agent.")]
    d = diff_prompts([], new)
    assert d.added_count == 1


def test_empty_new_removes():
    old = [msg("system", "You are an agent.")]
    d = diff_prompts(old, [])
    assert d.removed_count == 1


def test_multiple_changes():
    old = [msg("system", "Old system."), msg("user", "Same user."), msg("assistant", "Old answer.")]
    new = [msg("system", "New system."), msg("user", "Same user."), msg("assistant", "New answer.")]
    d = diff_prompts(old, new)
    assert d.changed_count == 2
    assert d.same_count == 1
    assert d.messages[1].status == "same"


# ---------------------------------------------------------------------------
# Anthropic content blocks
# ---------------------------------------------------------------------------

def test_content_blocks_text():
    old = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    new = [{"role": "user", "content": [{"type": "text", "text": "hello world"}]}]
    d = diff_prompts(old, new)
    assert d.changed_count == 1
    assert d.messages[0].old_content == "hello"
    assert d.messages[0].new_content == "hello world"


def test_content_blocks_same():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "same"}]}]
    d = diff_prompts(msgs, msgs)
    assert d.same_count == 1


def test_content_none():
    old = [{"role": "user", "content": None}]
    new = [{"role": "user", "content": None}]
    d = diff_prompts(old, new)
    assert d.same_count == 1


# ---------------------------------------------------------------------------
# unified_lines in changed messages
# ---------------------------------------------------------------------------

def test_changed_has_unified_lines():
    old = [msg("system", "Be helpful.")]
    new = [msg("system", "Be helpful. Be concise.")]
    d = diff_prompts(old, new)
    md = d.messages[0]
    assert md.unified_lines  # non-empty
    full = "".join(md.unified_lines)
    assert "Be concise" in full


def test_same_has_no_unified_lines():
    old = [msg("user", "Hello")]
    d = diff_prompts(old, old)
    assert d.messages[0].unified_lines == []


# ---------------------------------------------------------------------------
# render_diff
# ---------------------------------------------------------------------------

def test_render_diff_changed_contains_diff_text():
    old = [msg("system", "Be helpful.")]
    new = [msg("system", "Be helpful. Be concise.")]
    d = diff_prompts(old, new)
    rendered = render_diff(d)
    assert "system" in rendered
    assert "Be concise" in rendered


def test_render_diff_no_diff():
    old = [msg("user", "Hello")]
    d = diff_prompts(old, old)
    rendered = render_diff(d)
    assert "(no differences)" in rendered


def test_render_diff_show_same():
    old = [msg("user", "Hello")]
    d = diff_prompts(old, old)
    rendered = render_diff(d, show_same=True)
    # Should include the message even though same
    assert "user" in rendered


def test_render_diff_added():
    old = [msg("system", "Base.")]
    new = [msg("system", "Base."), msg("user", "New message")]
    d = diff_prompts(old, new)
    rendered = render_diff(d)
    assert "added" in rendered
    assert "New message" in rendered


def test_render_diff_removed():
    old = [msg("system", "Base."), msg("user", "Old message")]
    new = [msg("system", "Base.")]
    d = diff_prompts(old, new)
    rendered = render_diff(d)
    assert "removed" in rendered
    assert "Old message" in rendered


def test_render_diff_color_adds_ansi():
    old = [msg("system", "Old.")]
    new = [msg("system", "New.")]
    d = diff_prompts(old, new)
    rendered_color = render_diff(d, color=True)
    assert "\033[" in rendered_color


def test_render_diff_no_color_no_ansi():
    old = [msg("system", "Old.")]
    new = [msg("system", "New.")]
    d = diff_prompts(old, new)
    rendered = render_diff(d, color=False)
    assert "\033[" not in rendered


# ---------------------------------------------------------------------------
# context parameter
# ---------------------------------------------------------------------------

def test_context_lines():
    # Long content with a small change buried in the middle
    lines = [f"line {i}" for i in range(20)]
    lines_new = lines[:10] + ["changed line"] + lines[11:]
    old = [msg("system", "\n".join(lines))]
    new = [msg("system", "\n".join(lines_new))]
    d3 = diff_prompts(old, new, context=3)
    d0 = diff_prompts(old, new, context=0)
    # context=3 shows more lines than context=0
    assert len(d3.messages[0].unified_lines) > len(d0.messages[0].unified_lines)
