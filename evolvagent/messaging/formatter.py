"""
Message formatting for messaging integrations.

Converts EventBus events and agent status into human-readable messages
suitable for Telegram (Markdown-compatible, concise).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evolvagent.core.events import Event
    from evolvagent.core.skill import BaseSkill


def format_event(event: Event) -> str:
    """Format an EventBus event into a readable notification message."""
    topic = event.topic
    data = event.data

    if topic == "skill.executed":
        name = data.get("skill_name", "unknown")
        success = data.get("success", False)
        ms = data.get("execution_time_ms", 0)
        icon = "\u2705" if success else "\u274c"
        status = "OK" if success else "FAIL"
        return f"{icon} Skill `{name}` [{status}] ({ms:.0f}ms)"

    if topic == "agent.state_changed":
        old = data.get("old_state", "?")
        new = data.get("new_state", "?")
        return f"\U0001f504 State: {old} \u2192 {new}"

    if topic == "skill.learned":
        name = data.get("skill_name", "unknown")
        return f"\U0001f393 New skill learned: `{name}`"

    if topic == "scheduler.maintenance_completed":
        decayed = data.get("decayed_count", 0)
        archived = data.get("archived_count", 0)
        return f"\U0001f527 Maintenance: decayed={decayed} archived={archived}"

    if topic == "skill.registered":
        name = data.get("skill_name", "unknown")
        return f"\U0001f4e6 Skill registered: `{name}`"

    if topic == "skill.trust_promoted":
        name = data.get("skill_name", "unknown")
        old_trust = data.get("old_trust", "?")
        new_trust = data.get("new_trust", "?")
        return f"\u2b06\ufe0f Skill `{name}` trust: {old_trust} \u2192 {new_trust}"

    if topic == "network.skill_imported":
        name = data.get("skill_name", "unknown")
        source = data.get("source_agent", "?")
        return f"\U0001f310 Imported skill `{name}` from {source}"

    if topic == "skill.taught":
        name = data.get("skill_name", "unknown")
        return f"\U0001f4da Skill taught: `{name}`"

    if topic == "agent.reflection_started":
        return "\U0001f9d8 Reflection started..."

    if topic == "agent.reflection_completed":
        return "\u2705 Reflection completed."

    # Generic fallback
    return f"\U0001f514 [{topic}] {_truncate(str(data), 200)}"


def format_status_report(status: dict[str, Any]) -> str:
    """Format agent.status_dict() into a readable status report."""
    name = status.get("name", "unknown")
    state = status.get("state", "unknown")
    uptime = status.get("uptime_seconds", 0)
    skill_count = status.get("skill_count", 0)
    learned = status.get("learned_skill_count", 0)
    stats = status.get("stats", {})

    uptime_str = _format_duration(uptime)

    lines = [
        f"\U0001f916 *{name}* \u2014 Status Report",
        f"State: `{state}`",
        f"Uptime: {uptime_str}",
        f"Skills: {skill_count} ({learned} learned)",
        "",
        f"Requests: {stats.get('total_requests', 0)}",
        f"Success: {stats.get('successful_tasks', 0)} | "
        f"Failed: {stats.get('failed_tasks', 0)} | "
        f"No skill: {stats.get('no_skill_found', 0)}",
    ]

    net = status.get("network")
    if net and net.get("running"):
        lines.append(
            f"\nNetwork: port={net['port']} "
            f"peers={net.get('connected_peers', 0)}/{net.get('known_peers', 0)}"
        )

    sched = "running" if status.get("scheduler_running") else "stopped"
    lines.append(f"Scheduler: {sched}")

    return "\n".join(lines)


def format_skills_list(skills: list[BaseSkill]) -> str:
    """Format active skills into a readable list."""
    if not skills:
        return "No active skills."

    lines = [f"\U0001f9e0 *Active Skills* ({len(skills)})"]
    for s in skills:
        m = s.metadata
        rate = f"{m.success_rate:.0%}" if m.total_executions > 0 else "-"
        lines.append(
            f"\u2022 `{m.name}` [{m.trust_level.value}] "
            f"utility={m.utility_score:.2f} runs={m.total_executions} rate={rate}"
        )

    return "\n".join(lines)


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    hours = seconds / 3600
    if hours < 24:
        return f"{hours:.1f}h"
    days = hours / 24
    return f"{days:.1f}d"


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
