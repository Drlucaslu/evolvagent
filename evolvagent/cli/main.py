"""
EvolvAgent CLI interface.

Commands:
    evolvagent repl       - Interactive REPL mode (recommended)
    evolvagent ask        - Ask the agent a single question
    evolvagent context    - Generate/update CLAUDE.md with workspace context
    evolvagent status     - Show agent status
    evolvagent skills     - List registered skills
    evolvagent doctor     - Check setup and diagnose issues
    evolvagent init       - Initialize a new agent workspace
    evolvagent version    - Show version info
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_console = Console()
_err_console = Console(stderr=True)


def _run_async(coro):
    """Run an async function from sync CLI context."""
    return asyncio.run(coro)


def cmd_init(args):
    """Initialize a new EvolvAgent workspace."""
    name = args.name
    data_dir = Path(args.dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    config_path = data_dir / "config.toml"
    if not config_path.exists():
        src = Path(__file__).parent.parent.parent / "config.toml"
        if src.exists():
            shutil.copy(src, config_path)
        else:
            config_path.write_text(f'[agent]\nname = "{name}"\ndata_dir = "{data_dir}"\n')

    for sub in ["logs", "skills", "backups"]:
        (data_dir / sub).mkdir(exist_ok=True)

    body = Text()
    body.append("Name:     ", style="bold")
    body.append(f"{name}\n")
    body.append("Location: ", style="bold")
    body.append(f"{data_dir}\n\n")
    body.append("Next steps:\n", style="bold")
    body.append(f"  1. Edit {config_path} to set your LLM model/key\n", style="dim")
    body.append(f"  2. Run: evolvagent status", style="dim")
    _console.print(Panel(body, title="[green]Workspace Initialized[/]", border_style="green"))


def cmd_status(args):
    """Show current agent status."""
    from evolvagent.core import Agent, Settings, get_settings, reset_settings

    reset_settings()
    settings = get_settings()

    async def _status():
        agent = Agent(settings=settings)
        await agent.start()
        _register_builtins(agent, settings)
        info = agent.status_dict()
        await agent.shutdown()
        return info

    try:
        info = _run_async(_status())
    except Exception as e:
        _err_console.print(f"[red]Error:[/] {e}")
        sys.exit(1)

    body = Text()
    body.append("Name:          ", style="bold")
    body.append(f"{info['name']}\n")
    body.append("State:         ", style="bold")
    body.append(f"{info['state']}\n")
    body.append("Skills:        ", style="bold")
    body.append(f"{info['skill_count']}\n")
    body.append("Active Skills: ", style="bold")
    body.append(f"{', '.join(info['active_skills']) or '(none)'}\n")
    stats = info["stats"]
    body.append("Requests:      ", style="bold")
    body.append(f"{stats['total_requests']}\n")
    body.append("Success/Fail:  ", style="bold")
    body.append(f"{stats['successful_tasks']} / {stats['failed_tasks']}\n")
    sched = "running" if info.get("scheduler_running") else "stopped"
    body.append("Scheduler:     ", style="bold")
    body.append(f"{sched}")
    lr = info.get("last_reflection")
    if lr:
        body.append("\n")
        body.append("Last Reflect:  ", style="bold")
        if lr["skipped_reason"]:
            body.append(f"skipped ({lr['skipped_reason']})")
        else:
            body.append(
                f"analyzed={lr['skills_analyzed']} "
                f"updated={lr['skills_updated']} "
                f"principles={lr['principles_extracted']}"
            )
    _console.print(Panel(body, title="[cyan]Agent Status[/]", border_style="cyan"))


def cmd_skills(args):
    """List all registered skills."""
    from evolvagent.core import Agent, Settings, get_settings, reset_settings

    reset_settings()
    settings = get_settings()

    async def _skills():
        agent = Agent(settings=settings)
        await agent.start()
        _register_builtins(agent, settings)
        result = agent.active_skills
        await agent.shutdown()
        return result

    skill_list = _run_async(_skills())

    if not skill_list:
        _console.print(Panel("[dim]No skills registered yet.[/]", title="Skills",
                             border_style="yellow"))
        return

    table = Table(title="Registered Skills", border_style="blue")
    table.add_column("Name", style="bold cyan", min_width=18)
    table.add_column("Trust", style="yellow")
    table.add_column("Utility", justify="right")
    table.add_column("Runs", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("Status")
    for s in skill_list:
        m = s.metadata
        rate = f"{m.success_rate:.0%}" if m.total_executions > 0 else "-"
        trust_style = {"auto": "green", "suggest": "yellow", "observe": "red"}.get(
            m.trust_level.value, "")
        table.add_row(
            m.name,
            Text(m.trust_level.value, style=trust_style),
            f"{m.utility_score:.2f}",
            str(m.total_executions),
            rate,
            m.status.value,
        )
    _console.print()
    _console.print(table)
    _console.print()


def _make_llm(settings):
    """Create an LLMClient from settings."""
    from evolvagent.core.llm import LLMClient
    return LLMClient(
        model=settings.llm.model,
        fallback_model=settings.llm.fallback_model,
        max_tokens=settings.llm.max_tokens,
        temperature=settings.llm.temperature,
        daily_cost_limit_usd=settings.llm.daily_cost_limit_usd,
    )


def _register_builtin(agent, skill):
    """Register a built-in skill, restoring persisted metadata if available."""
    if not agent.get_skill(skill.metadata.name):
        if agent._store:
            saved = agent._store.load(skill.metadata.name)
            if saved:
                skill.metadata = saved
        agent.register_skill(skill)


def _register_builtins(agent, settings):
    """Register all built-in Skills."""
    from evolvagent.skills.file_search import FileSearchSkill
    from evolvagent.skills.general import GeneralAssistantSkill
    from evolvagent.skills.git_ops import GitActionSkill, GitInfoSkill
    from evolvagent.skills.shell import ShellCommandSkill
    from evolvagent.skills.summarize import SummarizeSkill
    from evolvagent.skills.workspace_context import WorkspaceContextSkill

    llm = _make_llm(settings)
    _register_builtin(agent, FileSearchSkill())
    _register_builtin(agent, GitInfoSkill())
    _register_builtin(agent, GitActionSkill(llm))
    _register_builtin(agent, GeneralAssistantSkill(llm))
    _register_builtin(agent, ShellCommandSkill(llm))
    _register_builtin(agent, SummarizeSkill(llm))
    _register_builtin(agent, WorkspaceContextSkill(llm))


async def _cli_confirm(skill_name: str, preview: str) -> bool:
    """Interactive confirmation callback for SUGGEST mode."""
    _console.print()
    _console.print(Panel(preview, title=f"[yellow]{skill_name}[/]", border_style="yellow"))
    answer = input("  Approve? (y/n): ").strip().lower()
    return answer in ("y", "yes")


def cmd_ask(args):
    """Ask the agent a question."""
    from evolvagent.core import Agent, get_settings, reset_settings

    query = " ".join(args.query)

    reset_settings()
    settings = get_settings()

    async def _ask():
        agent = Agent(settings=settings)
        agent.workspace = str(Path.cwd())
        agent.set_llm(_make_llm(settings))
        await agent.start()
        _register_builtins(agent, settings)
        result = await agent.handle_request(query, confirm_callback=_cli_confirm)
        await agent.shutdown()
        return result

    try:
        result = _run_async(_ask())
    except Exception as e:
        _err_console.print(f"[red]Error:[/] {e}")
        if _is_auth_error(e):
            _err_console.print("[yellow]Hint:[/] Run 'evolvagent doctor' to diagnose setup issues.")
        sys.exit(1)

    _console.print()
    _console.print(Markdown(result))
    _console.print()


def cmd_repl(args):
    """Interactive REPL mode — agent stays running with conversation context."""
    from evolvagent.core import Agent, get_settings, reset_settings

    reset_settings()
    settings = get_settings()

    history: list[dict[str, str]] = []

    def _print_skills(agent):
        table = Table(border_style="dim")
        table.add_column("Name", style="cyan", min_width=18)
        table.add_column("Trust", style="yellow")
        table.add_column("Utility", justify="right")
        table.add_column("Runs", justify="right")
        table.add_column("Rate", justify="right")
        for s in agent.active_skills:
            m = s.metadata
            rate = f"{m.success_rate:.0%}" if m.total_executions > 0 else "-"
            table.add_row(m.name, m.trust_level.value, f"{m.utility_score:.2f}",
                          str(m.total_executions), rate)
        _console.print(table)

    def _print_status(agent):
        info = agent.status_dict()
        stats = info["stats"]
        learned = info.get("learned_skill_count", 0)
        text = Text()
        text.append(f"{info['name']}", style="bold")
        text.append(f"  |  Skills: {info['skill_count']}")
        if learned:
            text.append(f" ({learned} learned)", style="cyan")
        text.append(f"  |  Requests: {stats['total_requests']}")
        text.append(f"  |  Success: {stats['successful_tasks']}", style="green")
        text.append(f"  Fail: {stats['failed_tasks']}", style="red")
        sched = "running" if info.get("scheduler_running") else "stopped"
        text.append(f"  |  Scheduler: {sched}")
        _console.print(text)
        lr = info.get("last_reflection")
        if lr:
            if lr["skipped_reason"]:
                _console.print(f"  [dim]Last reflection: skipped ({lr['skipped_reason']})[/]")
            else:
                _console.print(
                    f"  [dim]Last reflection: analyzed={lr['skills_analyzed']} "
                    f"updated={lr['skills_updated']} "
                    f"principles={lr['principles_extracted']}[/]"
                )
        ll = info.get("last_learning")
        if ll and not ll.get("skipped_reason"):
            _console.print(
                f"  [dim]Last learning: patterns={ll['patterns_analyzed']} "
                f"created={ll['skills_created']}[/]"
            )

    def _print_help():
        help_items = [
            ("/status", "show agent status"),
            ("/skills", "list registered skills"),
            ("/reflect", "trigger reflection + learning"),
            ("/teach", "teach agent a new skill"),
            ("/context", "generate/update CLAUDE.md"),
            ("/history", "show conversation history length"),
            ("/clear", "clear conversation history"),
            ("/exit", "quit REPL"),
        ]
        for cmd, desc in help_items:
            _console.print(f"  [bold cyan]{cmd:<12}[/] [dim]{desc}[/]")

    async def _repl():
        agent = Agent(settings=settings)
        agent.workspace = str(Path.cwd())
        agent.set_llm(_make_llm(settings))
        await agent.start()
        _register_builtins(agent, settings)

        _console.print(Panel(
            f"Agent [bold]'{agent.name}'[/] ready with [bold]{agent.skill_count}[/] skills.\n"
            f"Type [bold cyan]/help[/] for commands, [bold cyan]/exit[/] to quit.",
            title="[bold green]EvolvAgent REPL[/]",
            border_style="green",
        ))

        loop = asyncio.get_event_loop()

        while True:
            try:
                user_input = await loop.run_in_executor(
                    None, lambda: _console.input("[bold green]you>[/] ").strip()
                )
            except (EOFError, KeyboardInterrupt):
                _console.print()
                break

            if not user_input:
                continue

            # Slash commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                if cmd in ("/exit", "/quit", "/q"):
                    break
                elif cmd == "/help":
                    _print_help()
                    continue
                elif cmd == "/status":
                    _print_status(agent)
                    continue
                elif cmd == "/skills":
                    _print_skills(agent)
                    continue
                elif cmd == "/reflect":
                    if agent.state.value != "idle":
                        _console.print(
                            f"  [yellow]Cannot reflect: agent is {agent.state.value}[/]"
                        )
                        continue
                    _console.print("  [dim]Starting reflection...[/]")
                    result = await agent.enter_reflection()
                    if result is None:
                        _console.print("  [red]Reflection failed to start.[/]")
                    elif result.skipped_reason:
                        _console.print(f"  [yellow]Skipped:[/] {result.skipped_reason}")
                    else:
                        _console.print(
                            f"  [green]Reflection complete:[/] "
                            f"analyzed={result.skills_analyzed} "
                            f"updated={result.skills_updated} "
                            f"principles={result.principles_extracted}"
                        )
                        if result.errors:
                            for err in result.errors:
                                _console.print(f"  [red]Error:[/] {err}")
                    continue
                elif cmd == "/teach":
                    _console.print("  [bold]Teach a new skill[/]")
                    try:
                        name = await loop.run_in_executor(
                            None, lambda: input("  Name (snake_case): ").strip()
                        )
                        if not name:
                            _console.print("  [dim]Cancelled.[/]")
                            continue
                        desc = await loop.run_in_executor(
                            None, lambda: input("  Description: ").strip()
                        )
                        triggers_raw = await loop.run_in_executor(
                            None, lambda: input("  Trigger phrases (comma-sep): ").strip()
                        )
                        triggers = [t.strip() for t in triggers_raw.split(",") if t.strip()]
                        prompt = await loop.run_in_executor(
                            None, lambda: input("  System prompt: ").strip()
                        )
                        if not prompt or not triggers:
                            _console.print("  [red]Need at least triggers + prompt.[/]")
                            continue

                        skill = await agent.learn_skill(
                            name=name,
                            description=desc,
                            system_prompt=prompt,
                            triggers=triggers,
                        )
                        if skill:
                            _console.print(
                                f"  [green]Skill '{skill.metadata.name}' created![/] "
                                f"(trust: observe)"
                            )
                        else:
                            _console.print("  [red]Failed to create skill.[/]")
                    except (EOFError, KeyboardInterrupt):
                        _console.print("\n  [dim]Cancelled.[/]")
                    continue
                elif cmd == "/history":
                    _console.print(f"  [dim]{len(history) // 2} turns in history[/]")
                    continue
                elif cmd == "/clear":
                    history.clear()
                    _console.print("  [dim]Conversation history cleared.[/]")
                    continue
                elif cmd == "/context":
                    from evolvagent.skills.workspace_context import WorkspaceContextSkill
                    ctx_skill = WorkspaceContextSkill(_make_llm(settings))
                    ctx_result = await ctx_skill.execute({
                        "workspace": str(Path.cwd()),
                        "store": agent._store,
                    })
                    if ctx_result.success:
                        claude_md = Path.cwd() / "CLAUDE.md"
                        lines = ctx_result.output.split("\n")
                        preview_text = "\n".join(lines[:20])
                        if len(lines) > 20:
                            preview_text += f"\n... ({len(lines) - 20} more lines)"
                        _console.print(Panel(
                            Markdown(preview_text),
                            title=f"[cyan]Preview ({len(lines)} lines)[/]",
                            border_style="cyan",
                        ))
                        answer = input("  Write CLAUDE.md? (y/n): ").strip().lower()
                        if answer in ("y", "yes"):
                            claude_md.write_text(ctx_result.output, encoding="utf-8")
                            _console.print(f"  [green]Written to {claude_md}[/]")
                        else:
                            _console.print("  [dim]Cancelled.[/]")
                    else:
                        _err_console.print(f"  [red]Error:[/] {ctx_result.error}")
                    continue
                else:
                    _console.print(f"  [red]Unknown command:[/] {cmd}  (type /help)")
                    continue

            # Normal request — pass conversation history as context
            try:
                result = await agent.handle_request(
                    user_input,
                    context={"history": list(history)},
                    confirm_callback=_cli_confirm,
                )
                _console.print()
                _console.print(Markdown(result))
                _console.print()

                # Append to conversation history
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": result})

                # Keep history bounded (last 20 turns = 40 messages)
                if len(history) > 40:
                    history[:] = history[-40:]

            except Exception as e:
                _err_console.print(f"\n  [red]Error:[/] {e}")
                if _is_auth_error(e):
                    _err_console.print(
                        "  [yellow]Hint:[/] Run 'evolvagent doctor' to diagnose setup issues."
                    )
                _err_console.print()

        # Graceful shutdown
        _console.print("[dim]Shutting down...[/]")
        await agent.shutdown()
        _console.print("[dim]Goodbye![/]")

    _run_async(_repl())


def cmd_context(args):
    """Generate or update CLAUDE.md with workspace context."""
    from evolvagent.core import Agent, get_settings, reset_settings
    from evolvagent.skills.workspace_context import WorkspaceContextSkill

    workspace = Path(args.dir).resolve()
    reset_settings()
    settings = get_settings()

    async def _context():
        agent = Agent(settings=settings)
        agent.workspace = str(workspace)
        await agent.start()
        _register_builtins(agent, settings)

        llm = None if args.no_llm else _make_llm(settings)
        ctx_skill = WorkspaceContextSkill(llm)

        result = await ctx_skill.execute({
            "workspace": str(workspace),
            "store": agent._store,
            "use_llm": not args.no_llm,
        })

        await agent.shutdown()
        return result

    try:
        result = _run_async(_context())
    except Exception as e:
        _err_console.print(f"[red]Error:[/] {e}")
        sys.exit(1)

    if not result.success:
        _err_console.print(f"[red]Error:[/] {result.error}")
        sys.exit(1)

    if args.stdout:
        print(result.output)
        return

    claude_md = workspace / "CLAUDE.md"
    action = "Update" if claude_md.exists() else "Create"

    if not args.quiet:
        lines = result.output.split("\n")
        _console.print(f"\n  {action} [bold]{claude_md}[/]")
        _console.print(f"  [dim]({len(result.output)} bytes, {len(lines)} lines)[/]")
        answer = input(f"\n  Write CLAUDE.md? (y/n): ").strip().lower()
        if answer not in ("y", "yes"):
            _console.print("  [dim]Cancelled.[/]")
            return

    claude_md.write_text(result.output, encoding="utf-8")

    if not args.quiet:
        _console.print(f"  [green]Done! CLAUDE.md written to {claude_md}[/]")


def _is_auth_error(err: Exception) -> bool:
    """Check if an error is related to API key / authentication."""
    msg = str(err).lower()
    return any(kw in msg for kw in ("api key", "authentication", "authenticationerror"))


def cmd_doctor(args):
    """Run health checks on the EvolvAgent setup."""
    import os

    from rich.table import Table as RichTable

    table = RichTable(title="EvolvAgent Doctor", border_style="cyan")
    table.add_column("Check", style="bold", min_width=20)
    table.add_column("Status")
    table.add_column("Details", style="dim")

    # 1. Config file
    local_cfg = Path("config.toml")
    home_cfg = Path("~/.evolvagent/config.toml").expanduser()
    if local_cfg.exists():
        table.add_row("Config file", "[green]found[/]", str(local_cfg.resolve()))
    elif home_cfg.exists():
        table.add_row("Config file", "[green]found[/]", str(home_cfg))
    else:
        table.add_row("Config file", "[red]not found[/]", "Run 'evolvagent init'")

    # 2. Data directory
    data_dir = Path("~/.evolvagent").expanduser()
    if data_dir.is_dir() and os.access(data_dir, os.W_OK):
        table.add_row("Data directory", "[green]ok[/]", str(data_dir))
    elif data_dir.is_dir():
        table.add_row("Data directory", "[yellow]not writable[/]", str(data_dir))
    else:
        table.add_row("Data directory", "[red]missing[/]", "Run 'evolvagent init'")

    # 3. LLM model
    try:
        from evolvagent.core import get_settings, reset_settings
        reset_settings()
        settings = get_settings()
        model_name = settings.llm.model
    except Exception:
        model_name = "(could not load settings)"
    table.add_row("LLM model", "[cyan]info[/]", model_name)

    # 4. API key check based on model name
    model_lower = model_name.lower()
    if model_lower.startswith("ollama/") or model_lower.startswith("ollama_"):
        table.add_row("API key", "[green]not needed[/]", "Local model via Ollama")
    else:
        env_var = None
        if any(model_lower.startswith(p) for p in ("gpt-", "openai/", "o1-", "o3-")):
            env_var = "OPENAI_API_KEY"
        elif any(model_lower.startswith(p) for p in ("anthropic/", "claude-")):
            env_var = "ANTHROPIC_API_KEY"
        elif model_lower.startswith("gemini/") or model_lower.startswith("google/"):
            env_var = "GEMINI_API_KEY"

        if env_var:
            if os.environ.get(env_var):
                table.add_row("API key", "[green]set[/]", f"${env_var}")
            else:
                table.add_row("API key", "[red]missing[/]", f"Set ${env_var}")
        else:
            table.add_row("API key", "[yellow]unknown[/]", f"Check docs for '{model_name}'")

    # 5. litellm installed
    try:
        import litellm
        table.add_row("litellm", "[green]installed[/]", f"v{litellm.__version__}")
    except ImportError:
        table.add_row("litellm", "[red]missing[/]", "Run: pip install litellm")

    _console.print()
    _console.print(table)
    _console.print()


def cmd_version(args):
    """Show version info."""
    from evolvagent import __version__

    body = Text()
    body.append("EvolvAgent", style="bold")
    body.append(f" v{__version__}\n")
    body.append("A self-evolving AI agent that learns and grows\n", style="dim")
    body.append("Phase 3 — Skill Learning", style="dim italic")
    _console.print(Panel(body, border_style="cyan"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evolvagent",
        description="EvolvAgent — A self-evolving AI agent that learns and grows with you.",
    )
    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Initialize a new agent workspace")
    p_init.add_argument("-n", "--name", default="agent-001", help="Agent name")
    p_init.add_argument("-d", "--dir", default="~/.evolvagent", help="Data directory")
    p_init.set_defaults(func=cmd_init)

    # status
    p_status = sub.add_parser("status", help="Show agent status")
    p_status.set_defaults(func=cmd_status)

    # skills
    p_skills = sub.add_parser("skills", help="List registered skills")
    p_skills.set_defaults(func=cmd_skills)

    # ask
    p_ask = sub.add_parser("ask", help="Ask the agent a question")
    p_ask.add_argument("query", nargs="+", help="Your question or request")
    p_ask.set_defaults(func=cmd_ask)

    # repl
    p_repl = sub.add_parser("repl", help="Interactive REPL mode")
    p_repl.set_defaults(func=cmd_repl)

    # context
    p_ctx = sub.add_parser("context", help="Generate/update CLAUDE.md with workspace context")
    p_ctx.add_argument("-d", "--dir", default=".", help="Workspace directory to analyze")
    p_ctx.add_argument("--no-llm", action="store_true", help="Skip LLM enhancement")
    p_ctx.add_argument("--quiet", action="store_true", help="Minimal output (for git hooks)")
    p_ctx.add_argument("--stdout", action="store_true", help="Print to stdout instead of file")
    p_ctx.set_defaults(func=cmd_context)

    # doctor
    p_doctor = sub.add_parser("doctor", help="Check setup and diagnose issues")
    p_doctor.set_defaults(func=cmd_doctor)

    # version
    p_ver = sub.add_parser("version", help="Show version information")
    p_ver.set_defaults(func=cmd_version)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


app = main  # Alias for entry point

if __name__ == "__main__":
    main()
