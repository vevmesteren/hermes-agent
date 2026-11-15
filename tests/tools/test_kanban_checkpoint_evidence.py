"""Tests for the kanban_checkpoint_evidence tool.

Verifies that the tool:
- Registers evidence in the checkpoint_evidence table
- Sets the corresponding CP flag on the task
- Rejects invalid checkpoint names
- Rejects missing evidence files
- Supports upsert (re-registering evidence for the same checkpoint)
"""
import json
import os
import tempfile
import time
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_evidence_file(tmp_path):
    """Create a temporary evidence file."""
    evidence = tmp_path / "cp0-routing-decision.json"
    evidence.write_text(json.dumps({
        "signal_scores": {"regulatory": 0, "reversibility": 1},
        "routing_tier": "standard",
    }))
    return str(evidence)


@pytest.fixture
def mock_kanban_env(tmp_path, monkeypatch):
    """Set up a temporary kanban.db with the checkpoint schema."""
    import sqlite3
    from hermes_constants import get_hermes_home

    db_path = tmp_path / "kanban.db"
    conn = sqlite3.connect(str(db_path))

    # Create the tasks table with CP columns
    conn.executescript("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT,
            assignee TEXT,
            status TEXT DEFAULT 'ready',
            card_type TEXT,
            cp0_intake_completed INTEGER NOT NULL DEFAULT 0,
            cp0_5_decomposition_completed INTEGER NOT NULL DEFAULT 0,
            cp1_readiness_completed INTEGER NOT NULL DEFAULT 0,
            cp2_review_completed INTEGER NOT NULL DEFAULT 0,
            cp3_merge_completed INTEGER NOT NULL DEFAULT 0,
            cp4_closure_completed INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE checkpoint_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            checkpoint TEXT NOT NULL,
            evidence_path TEXT NOT NULL,
            evidence_sha256 TEXT NOT NULL,
            written_at INTEGER NOT NULL,
            writer_model_family TEXT,
            UNIQUE(task_id, checkpoint)
        );
        INSERT INTO tasks (id, title, assignee, status) VALUES ('test-cp-task', 'Test CP Task', 'vevlivery', 'running');
    """)
    conn.commit()
    conn.close()

    # Point HERMES_HOME to the temp dir and set the kanban task env
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "test-cp-task")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "test-run-1")

    # Create a symlink from the default kanban.db location to our test db
    hermes_home = tmp_path
    kanban_link = hermes_home / "kanban.db"
    if not kanban_link.exists():
        kanban_link.symlink_to(db_path)

    yield tmp_path


def test_checkpoint_evidence_registers_and_sets_flag(mock_kanban_env, temp_evidence_file):
    """Test that evidence is registered and the CP flag is set."""
    from tools.kanban_tools import _handle_checkpoint_evidence

    result = _handle_checkpoint_evidence({
        "task_id": "test-cp-task",
        "checkpoint": "cp0",
        "evidence_path": temp_evidence_file,
    })
    parsed = json.loads(result)
    assert parsed["ok"] is True
    assert parsed["checkpoint"] == "cp0"
    assert parsed["flag_column"] == "cp0_intake_completed"
    assert len(parsed["evidence_sha256"]) == 64  # SHA-256 hex

    # Verify the DB state
    import sqlite3
    conn = sqlite3.connect(str(mock_kanban_env / "kanban.db"))
    evidence_rows = conn.execute(
        "SELECT checkpoint, evidence_sha256 FROM checkpoint_evidence WHERE task_id = ?",
        ("test-cp-task",)
    ).fetchall()
    assert len(evidence_rows) == 1
    assert evidence_rows[0][0] == "cp0"

    flag = conn.execute(
        "SELECT cp0_intake_completed FROM tasks WHERE id = ?",
        ("test-cp-task",)
    ).fetchone()
    assert flag[0] == 1
    conn.close()


def test_checkpoint_evidence_rejects_invalid_checkpoint(mock_kanban_env, temp_evidence_file):
    """Test that invalid checkpoint names are rejected."""
    from tools.kanban_tools import _handle_checkpoint_evidence

    result = _handle_checkpoint_evidence({
        "task_id": "test-cp-task",
        "checkpoint": "cp99",
        "evidence_path": temp_evidence_file,
    })
    # The _kanban_handler decorator catches the _check exception and returns tool_error
    assert "checkpoint must be one of" in result or "error" in result.lower()


def test_checkpoint_evidence_rejects_missing_file(mock_kanban_env):
    """Test that missing evidence files are rejected."""
    from tools.kanban_tools import _handle_checkpoint_evidence

    result = _handle_checkpoint_evidence({
        "task_id": "test-cp-task",
        "checkpoint": "cp0",
        "evidence_path": "/nonexistent/path/to/evidence.json",
    })
    assert "evidence file not found" in result or "error" in result.lower()


def test_checkpoint_evidence_upsert(mock_kanban_env, temp_evidence_file, tmp_path):
    """Test that re-registering the same checkpoint updates the evidence."""
    from tools.kanban_tools import _handle_checkpoint_evidence

    # First registration
    _handle_checkpoint_evidence({
        "task_id": "test-cp-task",
        "checkpoint": "cp0",
        "evidence_path": temp_evidence_file,
    })

    # Second registration with a different file
    new_evidence = tmp_path / "cp0-v2.json"
    new_evidence.write_text(json.dumps({"routing_tier": "light"}))
    result2 = _handle_checkpoint_evidence({
        "task_id": "test-cp-task",
        "checkpoint": "cp0",
        "evidence_path": str(new_evidence),
    })
    parsed2 = json.loads(result2)
    assert parsed2["ok"] is True

    # Verify only one row exists (upsert, not duplicate)
    import sqlite3
    conn = sqlite3.connect(str(mock_kanban_env / "kanban.db"))
    rows = conn.execute(
        "SELECT COUNT(*) FROM checkpoint_evidence WHERE task_id = ? AND checkpoint = ?",
        ("test-cp-task", "cp0")
    ).fetchone()
    assert rows[0] == 1
    conn.close()


def test_checkpoint_evidence_all_checkpoints(mock_kanban_env, tmp_path):
    """Test that all 6 checkpoint identifiers work."""
    from tools.kanban_tools import _handle_checkpoint_evidence

    checkpoints = ["cp0", "cp0.5", "cp1", "cp2", "cp3", "cp4"]
    for cp in checkpoints:
        evidence = tmp_path / f"{cp}-evidence.json"
        evidence.write_text(json.dumps({"checkpoint": cp}))
        result = _handle_checkpoint_evidence({
            "task_id": "test-cp-task",
            "checkpoint": cp,
            "evidence_path": str(evidence),
        })
        parsed = json.loads(result)
        assert parsed["ok"] is True
        assert parsed["checkpoint"] == cp

    # Verify all 6 evidence rows exist
    import sqlite3
    conn = sqlite3.connect(str(mock_kanban_env / "kanban.db"))
    rows = conn.execute(
        "SELECT checkpoint FROM checkpoint_evidence WHERE task_id = ? ORDER BY checkpoint",
        ("test-cp-task",)
    ).fetchall()
    assert len(rows) == 6

    # Verify all flags are set
    flags = conn.execute(
        "SELECT cp0_intake_completed, cp0_5_decomposition_completed, "
        "cp1_readiness_completed, cp2_review_completed, "
        "cp3_merge_completed, cp4_closure_completed "
        "FROM tasks WHERE id = ?",
        ("test-cp-task",)
    ).fetchone()
    assert all(f == 1 for f in flags)
    conn.close()


def test_checkpoint_evidence_with_model_family(mock_kanban_env, temp_evidence_file):
    """Test that model_family is recorded for the CP-2 independence check."""
    from tools.kanban_tools import _handle_checkpoint_evidence

    result = _handle_checkpoint_evidence({
        "task_id": "test-cp-task",
        "checkpoint": "cp2",
        "evidence_path": temp_evidence_file,
        "model_family": "deepseek",
    })
    parsed = json.loads(result)
    assert parsed["ok"] is True
    assert parsed["model_family"] == "deepseek"

    import sqlite3
    conn = sqlite3.connect(str(mock_kanban_env / "kanban.db"))
    row = conn.execute(
        "SELECT writer_model_family FROM checkpoint_evidence WHERE task_id = ? AND checkpoint = ?",
        ("test-cp-task", "cp2")
    ).fetchone()
    assert row[0] == "deepseek"
    conn.close()
