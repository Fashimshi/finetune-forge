# tests/test_llm_client.py

from finetune_forge.utils.llm_client import _strip_json_fence


def test_strip_plain_json():
    assert _strip_json_fence('{"a": 1}') == '{"a": 1}'


def test_strip_fenced_json():
    assert _strip_json_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_bare_fence():
    assert _strip_json_fence('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_incomplete_fence_does_not_eat_content():
    # A lone opening fence with a body but no close should still recover the body.
    assert _strip_json_fence('```json\n{"a": 1}') == '{"a": 1}'


def test_strip_surrounding_whitespace():
    assert _strip_json_fence('   {"a": 1}   ') == '{"a": 1}'
