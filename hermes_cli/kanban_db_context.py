"""Worker-context rendering for ``build_worker_context``: the capped markdown sections a spawned
worker reads about its task (header, attachments, prior attempts, parent handoffs, role
history, comments).

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached late-bound via
``_kb`` so monkeypatching ``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import TYPE_CHECKING, Any, Optional

import hermes_cli.kanban_db as _kb

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Attachment, Comment, Task

# build_worker_context() caps, sized for a ~100k-char prompt with headroom.
_CTX_MAX_PRIOR_ATTEMPTS = 10      # most recent N prior runs shown in full
_CTX_MAX_COMMENTS       = 30      # most recent N comments shown in full
_CTX_MAX_FIELD_BYTES    = 4 * 1024   # per summary/error/metadata/result
_CTX_MAX_BODY_BYTES     = 8 * 1024   # per task.body (opening post)
_CTX_MAX_COMMENT_BYTES  = 2 * 1024   # per comment
_CTX_MAX_BRIEFING_BYTES = 32 * 1024  # injected briefing text (compressed; page_refs trimmed)

# Task-type alias mapping: maps common task keywords / patterns to the
# ``task_type`` values stored in ``agent_briefings``. The briefing table
# currently uses ``capability:C*`` as the primary key, plus bare capability
# codes (``C7``, ``C51``) and a few semantic aliases. This mapping covers the
# alias resolution so the injection hook can find a briefing even when the
# task's labels / title use a different naming convention than the DB.
_TASK_TYPE_ALIASES: dict[str, list[str]] = {
    "blog-publishing": ["blog-publishing", "blog", "article"],
    "blog": ["blog", "blog-publishing", "article"],
    "epic-delivery": ["epic-delivery", "epic", "capability:C7", "capability:C19"],
    "research-arb": ["research-arb", "research", "arb"],
    "research": ["research", "research-arb", "arb"],
    "platform-ops": ["platform-ops", "platform", "ops"],
}


def _infer_briefing_task_types(task: Task) -> list[str]:
    """Infer candidate ``task_type`` values to look up in ``agent_briefings``.

    Scans the task title, body, and labels for capability references
    (``capability:C*``) and semantic aliases (blog, epic, research, …).
    Returns an ordered list of candidates — first match wins at query time.
    """
    import re

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        if value and value not in seen:
            seen.add(value)
            candidates.append(value)

    # 1. Capability labels (e.g. "capability:C7", "capability:C51")
    if hasattr(task, "labels") and task.labels:
        for label in task.labels:
            _add(str(label).strip())
            # Also add the bare form (C7 without the "capability:" prefix)
            if ":" in str(label):
                _add(str(label).split(":", 1)[1])

    # 2. Search title + body for capability:C* patterns
    text = " ".join(p for p in (task.title or "", task.body or "") if p)
    for m in re.finditer(r"capability:(C\d+)", text, re.IGNORECASE):
        _add(f"capability:{m.group(1)}")
        _add(m.group(1))

    # 3. Search title + body for bare C* patterns (C7, C51, etc.)
    for m in re.finditer(r"\b(C\d{1,3})\b", text):
        cap = m.group(1)
        _add(f"capability:{cap}")
        _add(cap)

    # 4. Semantic aliases from title/body keywords
    text_lower = text.lower()
    alias_map = {
        "blog-publishing": ("blog", "article", "publishing"),
        "epic-delivery": ("epic", "delivery", "phase"),
        "research-arb": ("research", "arb", "architecture review"),
        "platform-ops": ("platform", "ops", "infrastructure"),
    }
    for db_type, keywords in alias_map.items():
        if any(kw in text_lower for kw in keywords):
            for alias in _TASK_TYPE_ALIASES.get(db_type, [db_type]):
                _add(alias)

    return candidates


def _get_briefing(conn: sqlite3.Connection, profile: str, task_types: list[str]) -> Optional[sqlite3.Row]:
    """Query ``agent_briefings`` for the first matching (profile, task_type) row.

    Falls back to the ``default`` profile if no briefing exists for the
    task's assignee. Returns ``None`` when no briefing exists for any
    candidate task_type under either profile.
    """
    if not task_types:
        return None

    # Try the task's assignee first, then fall back to the 'default' profile.
    for prof in (profile, "default"):
        for tt in task_types:
            row = conn.execute(
                "SELECT * FROM agent_briefings WHERE profile=? AND task_type=? LIMIT 1",
                (prof, tt),
            ).fetchone()
            if row is not None:
                return row

    return None


def _ctx_briefing(lines: list[str], conn: sqlite3.Connection, task: Task, now: int) -> None:
    """Inject a pre-compiled agent briefing as pre-context for this task.

    Queries ``agent_briefings`` (same DB as the kanban tables) for a
    confidence-weighted, compressed knowledge briefing matching the task's
    profile and inferred task_type. The briefing is injected *after* the
    header/body and *before* prior attempts — it is pre-digested context the
    worker reads before diving into the task's own history.

    If the briefing is stale (``compiled_at`` older than the last knowledge
    graph rebuild timestamp), a staleness note is prepended. If no briefing
    exists, this section is silently omitted.
    """
    try:
        conn.execute("SELECT 1 FROM agent_briefings LIMIT 1").fetchone()
    except sqlite3.Error:
        return  # Table doesn't exist — skip silently

    task_types = _infer_briefing_task_types(task)
    if not task_types:
        return

    # Use the task's assignee as the primary profile, falling back to
    # 'default' (handled inside _get_briefing). An unassigned task still
    # gets a 'default'-profile briefing if one exists.
    profile = task.assignee or "default"
    row = _get_briefing(conn, profile, task_types)
    if row is None:
        return  # No briefing for this task type — skip silently

    briefing = row["briefing"] if "briefing" in row.keys() else None
    if not briefing or not briefing.strip():
        return

    # Cap the briefing text to keep the prompt bounded.
    briefing_text = _ctx_cap(briefing, _CTX_MAX_BRIEFING_BYTES)

    compiled_at = row["compiled_at"] if "compiled_at" in row.keys() else None
    token_count = row["token_count"] if "token_count" in row.keys() else None
    reduction_ratio = row["reduction_ratio"] if "reduction_ratio" in row.keys() else None
    task_type = row["task_type"] if "task_type" in row.keys() else "?"
    briefing_profile = row["profile"] if "profile" in row.keys() else "?"

    lines.append("## Agent Briefing (pre-compiled context)")
    lines.append("")
    lines.append(
        f"_Profile: {briefing_profile} | Task type: {task_type} | "
        f"Compiled: {_ctx_stamp(int(compiled_at), now) if compiled_at else 'unknown'}"
    )
    if token_count:
        lines.append(f"_Compressed: {int(token_count):,} tokens"
                     + (f" ({reduction_ratio:.1%} of full read)" if reduction_ratio else "")
                     + "_")
    lines.append("")

    # Check staleness: compare compiled_at to last graph rebuild timestamp.
    # The graph rebuild timestamp is stored in ~/.hermes/veviki-state/<profile>/last_refresh.txt
    # If the briefing was compiled before the last rebuild, it may be stale.
    if compiled_at:
        try:
            _assignee = task.assignee or "default"
            state_path = os.path.join(
                os.path.expanduser("~"), ".hermes", "veviki-state",
                _assignee, "last_refresh.txt",
            )
            if os.path.exists(state_path):
                from datetime import datetime, timezone as _tz
                last_str = open(state_path).read().strip()
                last_dt = datetime.fromisoformat(last_str)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=_tz.utc)
                last_ts = last_dt.timestamp()
                if int(compiled_at) < int(last_ts):
                    age = _relative_age(int(compiled_at), now)
                    lines.append(
                        f"_⚠ Briefing may be stale (compiled {age}, "
                        f"before the last knowledge graph rebuild). "
                        f"A recompile is pending via km_lazy_refresh.py._"
                    )
                    lines.append("")
        except Exception:
            pass  # Staleness check is best-effort; never block on it

    lines.append(briefing_text)
    lines.append("")


def _relative_age(ts: Optional[int], now: Optional[int] = None) -> str:
    """``just now`` / ``18h ago`` / ``3d ago``; "" for a missing/invalid ts. An LLM
    reads a bare absolute timestamp as current fact — the relative age is what
    prompts a worker to re-verify stale sibling work."""
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return ""
    if now is None:
        now = int(time.time())
    delta = now - ts
    if delta < 60:  # includes negative = clock skew across machines; never claim "in the future"
        return "just now"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"


def _ctx_cap(s: Optional[str], limit: int = _CTX_MAX_FIELD_BYTES) -> str:
    """Truncate to ``limit`` chars with a visible ellipsis."""
    if not s:
        return ""
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[:limit] + f"… [truncated, {len(s) - limit} chars omitted]"


def _ctx_stamp(ts: int, now: int) -> str:
    """``YYYY-MM-DD HH:MM`` plus a relative age when one is available."""
    disp = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
    age = _relative_age(ts, now)
    return f"{disp}, {age}" if age else disp


def _ctx_metadata_line(metadata: Any) -> Optional[str]:
    if not metadata:
        return None
    try:
        return f"_metadata_: `{_ctx_cap(json.dumps(metadata, ensure_ascii=False, sort_keys=True))}`"
    except Exception:
        return None


def _ctx_tail(items: list, cap: int, noun: str) -> tuple[list, Optional[str]]:
    """Keep the newest ``cap`` items; describe the omitted head, if any."""
    omitted = max(0, len(items) - cap)
    if not omitted:
        return items, None
    return items[-cap:], (
        f"_({omitted} earlier {noun}{'s' if omitted != 1 else ''} "
        f"omitted; showing most recent {cap})_"
    )


def _ctx_header(lines: list[str], task: Task) -> None:
    lines.append(f"# Kanban task {task.id}: {task.title}")
    lines.append("")
    lines.append(f"Assignee: {task.assignee or '(unassigned)'}")
    lines.append(f"Status:   {task.status}")
    if task.tenant:
        lines.append(f"Tenant:   {task.tenant}")
    lines.append(f"Workspace: {task.workspace_kind} @ {task.workspace_path or '(unresolved)'}")
    if task.max_runtime_seconds is not None:
        terminal_timeout = _kb._worker_terminal_timeout_env(
            task.max_runtime_seconds, os.environ.get("TERMINAL_TIMEOUT"),
        )
        effective_terminal_timeout = terminal_timeout or os.environ.get("TERMINAL_TIMEOUT")
        lines.append(f"Max runtime: {task.max_runtime_seconds}s")
        if effective_terminal_timeout:
            lines.append(f"Terminal timeout: {effective_terminal_timeout}s")
    if task.branch_name:
        lines.append(f"Branch:   {task.branch_name}")
    lines.append("")
    if task.body and task.body.strip():
        lines.append("## Body")
        lines.append(_ctx_cap(task.body, _CTX_MAX_BODY_BYTES))
        lines.append("")


def _ctx_attachments(lines: list[str], attachments: list[Attachment]) -> None:
    """Absolute on-disk paths so the worker's file tools read them directly
    (remote terminal backends need the attachments dir mounted)."""
    if not attachments:
        return
    lines.append("## Attachments")
    lines.append(
        "Files attached to this task. Read them with the file/terminal "
        "tools at the absolute paths below:"
    )
    for att in attachments:
        size_kb = max(1, (att.size + 1023) // 1024) if att.size else 0
        size_str = f", {size_kb} KB" if size_kb else ""
        ctype = f", {att.content_type}" if att.content_type else ""
        lines.append(f"- `{att.filename}`{ctype}{size_str} → `{att.stored_path}`")
    lines.append("")


def _ctx_prior_attempts(lines: list[str], conn: sqlite3.Connection, task_id: str, now: int) -> None:
    """Closed runs on this task (the active run is this worker), newest
    ``_CTX_MAX_PRIOR_ATTEMPTS`` in full, older ones as a one-line marker."""
    all_prior = [r for r in _kb.list_runs(conn, task_id) if r.ended_at is not None]
    shown, omitted_note = _ctx_tail(all_prior, _CTX_MAX_PRIOR_ATTEMPTS, "attempt")
    if not shown:
        return
    first_shown_idx = len(all_prior) - len(shown) + 1
    lines.append("## Prior attempts on this task")
    if omitted_note:
        lines.append(omitted_note)
    for offset, run in enumerate(shown):
        profile = run.profile or "(unknown)"
        outcome = run.outcome or run.status
        lines.append(
            f"### Attempt {first_shown_idx + offset} — {outcome} ({profile}, {_ctx_stamp(run.started_at, now)})"
        )
        if run.summary and run.summary.strip():
            lines.append(_ctx_cap(run.summary))
        if run.error and run.error.strip():
            lines.append(f"_error_: {_ctx_cap(run.error)}")
        meta_line = _ctx_metadata_line(run.metadata)
        if meta_line:
            lines.append(meta_line)
        lines.append("")


def _ctx_parent_results(lines: list[str], conn: sqlite3.Connection, task_id: str, now: int) -> None:
    """Done-parent handoffs: newest ``completed`` run's summary+metadata,
    falling back to ``task.result`` for pre-runs-table data. Stamped with a
    relative age so the worker re-verifies stale upstream results."""
    parent_rows = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id", (task_id,),
    ).fetchall()
    wrote_header = False
    for pid in (r["parent_id"] for r in parent_rows):
        pt = _kb.get_task(conn, pid)
        if not pt or pt.status != "done":
            continue
        runs = [r for r in _kb.list_runs(conn, pid) if r.outcome == "completed"]
        runs.sort(key=lambda r: r.started_at, reverse=True)
        run = runs[0] if runs else None
        if not wrote_header:
            lines.append("## Parent task results")
            lines.append(
                "_Handoffs from upstream tasks, captured when each parent "
                "completed (see age below). These are point-in-time "
                "snapshots, not live state — if a result drives your "
                "current work and it's not recent, re-verify against the "
                "source before acting on it as current._"
            )
            wrote_header = True
        done_ts = run.ended_at if run is not None and run.ended_at else (pt.completed_at or None)
        age = _relative_age(done_ts, now)
        lines.append(f"### {pid}" + (f" (completed {age})" if age else ""))
        if run is not None and run.summary and run.summary.strip():
            lines.append(_ctx_cap(run.summary))
        elif pt.result:
            lines.append(_ctx_cap(pt.result))
        else:
            lines.append("(no result recorded)")
        meta_line = _ctx_metadata_line(run.metadata) if run is not None else None
        if meta_line:
            lines.append(meta_line)
        lines.append("")


def _ctx_role_history(lines: list[str], conn: sqlite3.Connection, task: Task, now: int) -> None:
    """The assignee's 5 most recent completed runs on OTHER tasks — implicit
    role continuity without wiring anything into SOUL.md / MEMORY.md."""
    if not task.assignee:
        return
    role_rows = conn.execute(
        "SELECT t.id, t.title, r.summary, r.ended_at "
        "FROM task_runs r JOIN tasks t ON r.task_id = t.id "
        "WHERE r.profile = ? AND r.task_id != ? "
        "  AND r.outcome = 'completed' "
        "ORDER BY r.ended_at DESC LIMIT 5", (task.assignee, task.id),
    ).fetchall()
    if not role_rows:
        return
    lines.append(f"## Recent work by @{task.assignee}")
    for row in role_rows:
        first = _kb._first_line(row["summary"], 200) or "(no summary)"
        lines.append(
            f"- {row['id']} — {row['title']} ({_ctx_stamp(_kb._to_epoch(row['ended_at']) or 0, now)}): {first}"
        )
    lines.append("")


def _ctx_comments(lines: list[str], comments: list[Comment], now: int) -> None:
    """Newest ``_CTX_MAX_COMMENTS`` comments. The explicit "comment from
    worker" framing stops an operator-controlled HERMES_PROFILE like
    "hermes-system" being read as a system directive above an
    attacker-influenceable body (defense-in-depth)."""
    shown, omitted_note = _ctx_tail(comments, _CTX_MAX_COMMENTS, "comment")
    if not shown:
        return
    lines.append("## Comment thread")
    if omitted_note:
        lines.append(omitted_note)
    for c in shown:
        # Render author with explicit "comment from worker" framing so operator-controlled HERMES_PROFILE
        # values like "hermes-system" or "operator" can't be misread by the next worker as a system
        # directive above the (attacker-influenceable) comment body. Defense-in-depth — the LLM-controlled
        # author-forgery surface was already closed in #22435. See #22452.
        safe_author = (c.author or "").replace("`", "")
        lines.append(f"comment from worker `{safe_author}` at {_ctx_stamp(c.created_at, now)}:")
        lines.append(_ctx_cap(c.body, _CTX_MAX_COMMENT_BYTES))
        lines.append("")
