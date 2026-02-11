# EvolvAgent

A self-evolving AI agent that learns, reflects, and grows through use.

## Features

- **Natural Language Interface** -- ask questions and give instructions in plain English
- **Skill-Based Architecture** -- modular skills for shell commands, git, file search, summarization, and more
- **Progressive Trust** -- skills start in `observe` mode and earn autonomy through successful executions
- **LLM-Driven Reflection** -- agent periodically analyzes skill history and extracts reusable principles
- **Background Scheduler** -- idle-time maintenance with Ebbinghaus decay and automatic skill archival
- **LLM Agnostic** -- works with OpenAI, Anthropic, Ollama, and any provider supported by LiteLLM
- **Persistent Memory** -- agent state, skill metadata, and utility scores survive across sessions
- **Interactive REPL** -- conversational mode with history, slash commands, and live skill inspection

## Quick Start

```bash
# Install
pip install -e .

# Initialize workspace
evolvagent init

# Set your API key (pick one)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Or use a local model -- no API key needed
# Edit ~/.evolvagent/config.toml -> model = "ollama/llama3"

# Check setup
evolvagent doctor

# Start the interactive REPL
evolvagent repl
```

## Configuration

EvolvAgent reads `config.toml` from the current directory or `~/.evolvagent/config.toml`.

Key settings:

```toml
[llm]
model = "anthropic/claude-haiku-4-5-20251001"   # Any LiteLLM model string
fallback_model = ""                              # Optional fallback
max_tokens = 4096
temperature = 0.3
daily_cost_limit_usd = 1.0                       # Daily spend cap

[trust]
default_level = "observe"       # New skills start here
promote_threshold = 10          # Successes before auto-promotion

[scheduler]
cpu_threshold_percent = 70      # Skip maintenance if CPU above this
memory_threshold_percent = 80   # Skip maintenance if memory above this
idle_check_interval = 300       # Seconds between idle checks
min_idle_for_reflection = 600   # Seconds idle before triggering reflection

[evolution]
decay_factor = 0.95             # Ebbinghaus decay: utility *= factor ^ days_idle
archive_threshold = 0.2         # Archive skills below this utility
archive_after_days = 30         # ...after this many days unused
```

Supported model prefixes: `gpt-*`, `openai/*`, `anthropic/*`, `claude-*`, `ollama/*`, `gemini/*`, and [more](https://docs.litellm.ai/docs/providers).

## Commands

| Command | Description |
|---------|-------------|
| `evolvagent repl` | Interactive REPL mode (recommended) |
| `evolvagent ask <question>` | Ask the agent a single question |
| `evolvagent context` | Generate/update CLAUDE.md with workspace context |
| `evolvagent status` | Show agent status and stats |
| `evolvagent skills` | List registered skills with trust levels and scores |
| `evolvagent doctor` | Check setup and diagnose issues |
| `evolvagent init` | Initialize a new agent workspace |
| `evolvagent version` | Show version information |

### REPL Slash Commands

Inside the REPL you can use `/help`, `/status`, `/skills`, `/reflect`, `/context`, `/history`, `/clear`, and `/exit`.

## Built-in Skills

| Skill | Trust Level | Description |
|-------|-------------|-------------|
| GeneralAssistant | auto | General-purpose AI assistant for questions and tasks |
| ShellCommand | suggest | Execute shell commands from natural language (requires confirmation) |
| Summarize | auto | Summarize text, articles, or file contents |
| FileSearch | auto | Search files by name or content |
| GitInfo | auto | Git read-only operations: status, diff, log, branch |
| GitAction | suggest | Git write operations: commit, stash (requires confirmation) |
| WorkspaceContext | auto | Analyze workspace and generate CLAUDE.md |

**Trust levels:** `observe` (dry-run only) -> `suggest` (ask before executing) -> `auto` (execute freely). Skills promote automatically after reaching the configured success threshold.

## Architecture

```
User Request
  -> Agent.handle_request()
    -> Intent classification (LLM)
    -> Skill selection (best match by name + utility score)
    -> Trust enforcement (observe / suggest / auto)
    -> Skill.execute()
    -> Result + utility update + persist

Background (idle time)
  -> AgentScheduler._check_and_act()
    -> Resource check (CPU / memory)
    -> ReflectionEngine.reflect() -- LLM extracts principles from skill history
    -> Ebbinghaus decay -- unused skills lose utility over time
    -> Archival -- stale low-utility skills get archived
```

The agent maintains a registry of skills. Each skill has metadata tracking its utility score, execution count, and trust level. After every execution the utility score is updated and the skill metadata is persisted to disk. Over time, high-performing skills are promoted and low-utility skills decay. During idle time, the reflection engine uses the LLM to distill reusable principles from skill execution history.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check .
```

## License

MIT
