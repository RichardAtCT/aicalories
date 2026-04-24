"""Regression tests for `_parse_json`.

The naïve `json.loads` implementation died on two real LLM failure modes:
1. Valid JSON with prose appended after the closing brace
   (`Extra data: line ... column ...`).
2. Prose explanation *before* the JSON block.

`_parse_json` must tolerate both — they are the root cause of the Stage 3
"silent empty MealEstimate" bug that made the agent tell users their
(perfectly clear) descriptions were unparseable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calorie_estimator.estimator import _parse_json


def test_plain_json():
    assert _parse_json('{"matches": [1, 2]}') == {"matches": [1, 2]}


def test_markdown_fenced_json():
    wrapped = '```json\n{"ok": true}\n```'
    assert _parse_json(wrapped) == {"ok": True}


def test_trailing_prose_is_ignored():
    """Real failure mode: LLM appended an explanation after the JSON."""
    text = '{"matches": [{"item_id": 1}]}\n\nNotes: I picked the grilled variant.'
    assert _parse_json(text) == {"matches": [{"item_id": 1}]}


def test_prose_prefix_is_stripped():
    """LLM sometimes opens with prose before the JSON."""
    text = 'Here is the best match I could find:\n\n{"matches": []}'
    assert _parse_json(text) == {"matches": []}


def test_both_prose_prefix_and_suffix():
    text = 'Sure — here it is: {"a": 1, "b": 2}\n\nLet me know if that works.'
    assert _parse_json(text) == {"a": 1, "b": 2}


def test_invalid_json_still_raises():
    with pytest.raises(json.JSONDecodeError):
        _parse_json("this is not JSON at all")
