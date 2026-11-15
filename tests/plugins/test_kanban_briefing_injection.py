"""Tests for C51 Agent Briefing injection hook in ``build_worker_context``.

Verifies that a pre-compiled briefing from the ``agent_briefings`` table is
injected as a ``## Agent Briefing`` section in the worker context when a
task's title/body matches a capability in the briefing table.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _insert_briefing(conn, profile="default", task_type="capability:C7",
                     briefing="# Briefing: capability:C7\n\nTest briefing content."):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_briefings (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            profile                 TEXT NOT NULL,
            task_type               TEXT NOT NULL,
            briefing                TEXT,
            compiled_at             REAL,
            token_count             INTEGER,
            page_refs               TEXT,
            source_traces           INTEGER,
            confidence_scores       TEXT,
            full_read_token_count   INTEGER,
            reduction_ratio         REAL,
            updated_at              TEXT
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_briefings_profile_task
        ON agent_briefings(profile, task_type)
    """)
    conn.execute("""
        INSERT INTO agent_briefings
            (profile, task_type, briefing, compiled_at, token_count,
             page_refs, source_traces, confidence_scores,
             full_read_token_count, reduction_ratio, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (profile, task_type, briefing, time.time(), 500,
          "[]", 10, "{}", 5000, 0.1, "2026-01-01 00:00:00"))
    conn.commit()


def test_briefing_injected_for_matching_capability(kanban_home):
    """A task with C7 in its title gets the capability:C7 briefing injected."""
    conn = kbc.connect()
    try:
        _insert_briefing(conn, task_type="capability:C7",
                         briefing="# Briefing: capability:C7\n\nRelevant C7 context here.")
        task_id = kb.create_task(conn, title="C7: Test task for briefing injection")
        ctx = kb.build_worker_context(conn, task_id)
        assert "## Agent Briefing" in ctx
        assert "capability:C7" in ctx
        assert "Relevant C7 context here" in ctx
    finally:
        conn.close()


def test_briefing_not_injected_when_no_match(kanban_home):
    """A task with no capability reference gets no briefing section."""
    conn = kbc.connect()
    try:
        _insert_briefing(conn, task_type="capability:C99",
                         briefing="# Briefing: capability:C99\n\nShould not appear.")
        task_id = kb.create_task(conn, title="Generic task with no capability ref")
        ctx = kb.build_worker_context(conn, task_id)
        assert "## Agent Briefing" not in ctx
    finally:
        conn.close()


def test_briefing_not_injected_when_table_missing(kanban_home):
    """No agent_briefings table → no briefing section, no crash."""
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="C7: task with no briefing table")
        ctx = kb.build_worker_context(conn, task_id)
        assert "## Agent Briefing" not in ctx
    finally:
        conn.close()


def test_briefing_injected_for_profile_fallback(kanban_home):
    """Briefing for 'default' profile is found even when task assignee differs."""
    conn = kbc.connect()
    try:
        _insert_briefing(conn, task_type="capability:C51",
                         briefing="# Briefing: capability:C51\n\nC51 specific context.")
        task_id = kb.create_task(conn, title="C51: Agent Briefings task")
        kb.assign_task(conn, task_id, "vevlivery")
        ctx = kb.build_worker_context(conn, task_id)
        assert "## Agent Briefing" in ctx
        assert "capability:C51" in ctx
    finally:
        conn.close()


def test_briefing_truncation(kanban_home):
    """Large briefings are capped to keep the prompt bounded."""
    conn = kbc.connect()
    try:
        huge_briefing = "# Briefing: C7\n\n" + "x" * 100_000
        _insert_briefing(conn, task_type="capability:C7", briefing=huge_briefing)
        task_id = kb.create_task(conn, title="C7: task needing truncation")
        ctx = kb.build_worker_context(conn, task_id)
        assert "## Agent Briefing" in ctx
        # The 32K cap should keep it bounded
        briefing_start = ctx.index("## Agent Briefing")
        next_section = ctx.find("\n## ", briefing_start + 20)
        if next_section > 0:
            briefing_len = next_section - briefing_start
        else:
            briefing_len = len(ctx) - briefing_start
        assert briefing_len < 40_000, f"briefing section too large: {briefing_len}"
    finally:
        conn.close()