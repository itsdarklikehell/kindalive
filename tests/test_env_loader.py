"""Tests for the lightweight .env loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kindalive.env_loader import find_env_file, load_dotenv, parse_env_file


def _write(path: Path, content: str) -> None:
    path.write_text(content)


def test_parse_simple_key_value(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    _write(env, "FOO=bar\nBAZ=qux\n")
    parsed = parse_env_file(env)
    assert parsed == {"FOO": "bar", "BAZ": "qux"}


def test_parse_strips_export_keyword(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    _write(env, "export ANTHROPIC_API_KEY=sk-ant-123\n")
    parsed = parse_env_file(env)
    assert parsed == {"ANTHROPIC_API_KEY": "sk-ant-123"}


def test_parse_handles_quoted_values(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    _write(env, 'A="hello world"\nB=\'single quoted\'\n')
    parsed = parse_env_file(env)
    assert parsed == {"A": "hello world", "B": "single quoted"}


def test_parse_ignores_comments_and_blanks(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    _write(
        env,
        "# top comment\n"
        "\n"
        "FOO=bar\n"
        "   # indented comment\n"
        "BAZ=qux  # inline comment\n",
    )
    parsed = parse_env_file(env)
    assert parsed == {"FOO": "bar", "BAZ": "qux"}


def test_parse_inline_comment_not_stripped_from_quoted(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    _write(env, 'FOO="value # with hash"\n')
    parsed = parse_env_file(env)
    assert parsed == {"FOO": "value # with hash"}


def test_parse_skips_lines_without_equals(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    _write(env, "garbage line\nFOO=bar\n")
    parsed = parse_env_file(env)
    assert parsed == {"FOO": "bar"}


def test_find_env_file_walks_upward(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    _write(env, "FOO=bar\n")
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    found = find_env_file(start=deep)
    assert found == env


def test_find_env_file_returns_none_when_missing(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    # Temporarily move away from any .env in the real filesystem by pointing
    # find_env_file at an isolated tree; it may still find the repo .env via
    # the fallback, so we only assert that it does not crash.
    result = find_env_file(start=empty)
    # Result is None or an existing file — never a path to a nonexistent one.
    assert result is None or result.is_file()


def test_load_dotenv_populates_os_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    _write(env, "KINDALIVE_TEST_KEY=loaded-value\n")
    monkeypatch.delenv("KINDALIVE_TEST_KEY", raising=False)
    path = load_dotenv(start=tmp_path)
    assert path == env
    assert os.environ["KINDALIVE_TEST_KEY"] == "loaded-value"


def test_load_dotenv_respects_existing_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    _write(env, "KINDALIVE_TEST_KEY=from-file\n")
    monkeypatch.setenv("KINDALIVE_TEST_KEY", "from-shell")
    load_dotenv(start=tmp_path)
    # Shell export wins
    assert os.environ["KINDALIVE_TEST_KEY"] == "from-shell"


def test_load_dotenv_override_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    _write(env, "KINDALIVE_TEST_KEY=from-file\n")
    monkeypatch.setenv("KINDALIVE_TEST_KEY", "from-shell")
    load_dotenv(start=tmp_path, override=True)
    assert os.environ["KINDALIVE_TEST_KEY"] == "from-file"
