"""kanban.worker_env_allowlist: explicit env var allowlist for dispatch (VEVL-D-P114).

The dispatch mechanism builds the worker agent's environment from an explicit
allowlist declared in ``config.yaml`` (``kanban.worker_env_allowlist``), not
from the dispatcher's inherited environment. Workers get exactly what they
need — nothing more, nothing less. Secret leakage becomes structurally
impossible: if a var isn't on the list, it doesn't get through.

When the allowlist is unset, the dispatch falls back to the current
inherit-scrub-inject behavior and logs a deprecation warning.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def allowlist_config(kanban_home):
    """Write a config.yaml with the worker_env_allowlist key set."""
    (kanban_home / "config.yaml").write_text(
        "kanban:\n"
        "  worker_env_allowlist:\n"
        "    - PATH\n"
        "    - HOME\n"
        "    - USER\n"
        "    - LANG\n"
        "    - LC_*\n"
        "    - HERMES_HOME\n"
        "    - TERMINAL_TIMEOUT\n",
        encoding="utf-8",
    )
    return kanban_home


@pytest.fixture
def no_allowlist_config(kanban_home):
    """Write a config.yaml WITHOUT the worker_env_allowlist key (fallback path)."""
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  dispatch_interval_seconds: 60\n",
        encoding="utf-8",
    )
    return kanban_home


def _make_claimed_task(conn, assignee="default"):
    """Create and claim a task, returning the Task object."""
    tid = kb.create_task(conn, title="test task", assignee=assignee)
    kb.claim_task(conn, tid)
    return kb.get_task(conn, tid)


def _capture_spawn_env(monkeypatch, task, workspace, *, board="default"):
    """Call _default_spawn with a mocked Popen that captures the env dict.

    Returns the env dict that would have been passed to the worker subprocess.
    """
    captured: dict = {}

    class _FakeProc:
        pid = 4242

    def _fake_popen(*args, **kwargs):
        captured["env"] = dict(kwargs.get("env", {}))
        return _FakeProc()

    monkeypatch.setattr("hermes_cli.kanban_db_dispatch.subprocess.Popen", _fake_popen)
    # Prevent the worker from actually running: return a dummy argv.
    monkeypatch.setattr(
        "hermes_cli.kanban_db_dispatch._worker_argv",
        lambda *_a, **_kw: ["/bin/true"],
    )
    monkeypatch.setattr(
        "hermes_cli.kanban_db_dispatch._restart_safe_worker_argv",
        lambda _t, cmd: cmd,
    )
    monkeypatch.setattr(
        "hermes_cli.kanban_db_dispatch._open_worker_log",
        lambda *_a, **_kw: open(os.devnull, "w"),
    )
    # Skip model preflight: the test isn't checking model resolution.
    monkeypatch.setattr(
        "hermes_cli.kanban_worker_preflight.refuse_unresolved_model",
        lambda *_a, **_kw: None,
    )

    pid = kbd._default_spawn(task, str(workspace), board=board)
    assert pid is not None
    return captured["env"]


# ---------------------------------------------------------------------------
# Unit tests: _allowlist_env_filter
# ---------------------------------------------------------------------------

class TestAllowlistEnvFilter:
    """Unit tests for the _allowlist_env_filter function."""

    def test_exact_match_kept(self):
        env = {"PATH": "/bin", "HOME": "/root", "SECRET_KEY": "leak"}
        allowlist = frozenset({"PATH", "HOME"})
        result = kbd._allowlist_env_filter(env, allowlist)
        assert "PATH" in result
        assert "HOME" in result
        assert "SECRET_KEY" not in result

    def test_prefix_wildcard(self):
        env = {"LC_CTYPE": "en_US.UTF-8", "LC_ALL": "C", "LC_MESSAGES": "en",
               "OTHER": "x"}
        allowlist = frozenset({"LC_*"})
        result = kbd._allowlist_env_filter(env, allowlist)
        assert "LC_CTYPE" in result
        assert "LC_ALL" in result
        assert "LC_MESSAGES" in result
        assert "OTHER" not in result

    def test_hermes_kanban_always_kept(self):
        """HERMES_KANBAN_* vars are structurally required and always kept."""
        env = {"HERMES_KANBAN_TASK": "t1", "HERMES_KANBAN_DB": "/db.sqlite",
               "OPENAI_API_KEY": "sk-leak"}
        allowlist = frozenset({"PATH"})
        result = kbd._allowlist_env_filter(env, allowlist)
        assert "HERMES_KANBAN_TASK" in result
        assert "HERMES_KANBAN_DB" in result
        assert "OPENAI_API_KEY" not in result

    def test_case_insensitive(self):
        """Windows env blocks are case-insensitive; the allowlist matches the same way."""
        env = {"path": "/bin", "PATH": "/usr/bin", "Home": "/root"}
        allowlist = frozenset({"PATH", "HOME"})
        result = kbd._allowlist_env_filter(env, allowlist)
        assert "path" in result
        assert "PATH" in result
        assert "Home" in result

    def test_empty_allowlist_keeps_only_kanban(self):
        """An empty allowlist still keeps HERMES_KANBAN_* and deployment names."""
        env = {"HERMES_KANBAN_TASK": "t1", "PATH": "/bin", "SECRET": "x"}
        allowlist = frozenset()
        result = kbd._allowlist_env_filter(env, allowlist)
        assert "HERMES_KANBAN_TASK" in result
        assert "PATH" not in result
        assert "SECRET" not in result

    def test_secret_not_on_allowlist_excluded(self):
        """REQ-3: any var not on the allowlist MUST NOT appear in the worker env."""
        env = {"OPENAI_API_KEY": "sk-leak", "SOME_CUSTOM_VAR": "custom",
               "PATH": "/bin"}
        allowlist = frozenset({"PATH"})
        result = kbd._allowlist_env_filter(env, allowlist)
        assert "OPENAI_API_KEY" not in result
        assert "SOME_CUSTOM_VAR" not in result
        assert "PATH" in result


# ---------------------------------------------------------------------------
# Config reader: configured_worker_env_allowlist
# ---------------------------------------------------------------------------

class TestConfiguredWorkerEnvAllowlist:

    def test_reads_allowlist_from_config(self, allowlist_config, monkeypatch):
        # Force a config reload for the test's isolated home
        from hermes_cli.config import _load_config_impl
        monkeypatch.setattr("hermes_cli.config._load_config_impl", _load_config_impl)
        result = kbd.configured_worker_env_allowlist()
        assert result is not None
        assert "PATH" in result
        assert "HOME" in result
        assert "LC_*" in result
        assert "TERMINAL_TIMEOUT" in result

    def test_returns_none_when_unset(self, no_allowlist_config):
        result = kbd.configured_worker_env_allowlist()
        assert result is None

    def test_returns_none_when_not_a_list(self, kanban_home):
        (kanban_home / "config.yaml").write_text(
            "kanban:\n  worker_env_allowlist: not-a-list\n", encoding="utf-8",
        )
        result = kbd.configured_worker_env_allowlist()
        assert result is None


# ---------------------------------------------------------------------------
# Integration: _default_spawn with allowlist
# ---------------------------------------------------------------------------

@pytest.mark.platforms("linux")
class TestDefaultSpawnWithAllowlist:

    def test_worker_env_contains_only_allowlisted_vars(
        self, allowlist_config, tmp_path, monkeypatch, all_assignees_spawnable,
    ):
        """REQ-1: worker receives exactly allowlisted vars + HERMES_KANBAN_*."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
        monkeypatch.setenv("SOME_CUSTOM_VAR", "should-not-leak")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        monkeypatch.setenv("LANG", "en_US.UTF-8")
        monkeypatch.setenv("LC_CTYPE", "en_US.UTF-8")

        with kbc.connect() as conn:
            task = _make_claimed_task(conn)
            ws = tmp_path / "ws"
            ws.mkdir()
            env = _capture_spawn_env(monkeypatch, task, ws)

        # Secrets that are NOT on the allowlist must not leak
        assert "OPENAI_API_KEY" not in env, "secret leaked through allowlist"
        assert "SOME_CUSTOM_VAR" not in env, "non-allowlisted var leaked"
        # Allowlisted vars should be present
        assert "PATH" in env
        assert "LANG" in env
        # LC_* prefix wildcard
        assert "LC_CTYPE" in env
        # Structurally-required HERMES_KANBAN_* always present
        assert env.get("HERMES_KANBAN_TASK") == task.id
        assert "HERMES_KANBAN_DB" in env

    def test_secret_not_on_allowlist_does_not_leak(
        self, allowlist_config, tmp_path, monkeypatch, all_assignees_spawnable,
    ):
        """REQ-3: OPENAI_API_KEY not on allowlist => not in worker env."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-leak-test")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-leak-test")
        monkeypatch.setenv("PATH", "/bin")

        with kbc.connect() as conn:
            task = _make_claimed_task(conn)
            ws = tmp_path / "ws"
            ws.mkdir()
            env = _capture_spawn_env(monkeypatch, task, ws)

        assert "OPENAI_API_KEY" not in env
        assert "OPENROUTER_API_KEY" not in env

    def test_fallback_when_allowlist_unset(
        self, no_allowlist_config, tmp_path, monkeypatch, all_assignees_spawnable,
        caplog,
    ):
        """REQ-2 fallback: when allowlist is unset, current behavior + deprecation warning."""
        monkeypatch.setenv("PATH", "/bin")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-present-in-fallback")

        with kbc.connect() as conn:
            task = _make_claimed_task(conn)
            ws = tmp_path / "ws"
            ws.mkdir()
            with caplog.at_level("WARNING", logger="hermes_cli.kanban_db_dispatch"):
                env = _capture_spawn_env(monkeypatch, task, ws)

        # Fallback path: the scrub handles secrets, but non-secret vars are inherited
        assert "PATH" in env
        # Deprecation warning was logged
        assert any("worker_env_allowlist" in r.getMessage() and "falling back" in r.getMessage()
                   for r in caplog.records)
        # Worker still gets structural vars
        assert env.get("HERMES_KANBAN_TASK") == task.id

    def test_worker_env_has_required_kanban_vars(
        self, allowlist_config, tmp_path, monkeypatch, all_assignees_spawnable,
    ):
        """The worker client reads HERMES_KANBAN_* from os.environ — verify they're present."""
        with kbc.connect() as conn:
            task = _make_claimed_task(conn)
            ws = tmp_path / "ws"
            ws.mkdir()
            env = _capture_spawn_env(monkeypatch, task, ws)

        # Vars the worker client reads (kanban_worker_client.py)
        assert env.get("HERMES_KANBAN_TASK") == task.id
        assert "HERMES_KANBAN_BOARD" in env
        assert "HERMES_KANBAN_DB" in env
        # HERMES_PROFILE is set by _default_spawn (not from allowlist)
        assert "HERMES_PROFILE" in env