# EvolvAgent

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/Drlucaslu/evolvagent/actions/workflows/ci.yml/badge.svg)

A decentralized self-evolving agent network where agents learn, reflect, and share skills.

- **Natural Language Interface** -- ask questions and give instructions in plain English
- **Skill-Based Architecture** -- modular skills for shell, git, file search, summarization, and more
- **Progressive Trust** -- skills start in `observe` mode and earn autonomy through successful executions
- **LLM-Driven Reflection** -- agent periodically analyzes skill history and extracts reusable principles
- **Skill Learning** -- agent automatically creates new skills from usage patterns, or learn via `/teach`
- **P2P Network** -- agents discover peers, share skills over WebSocket, and build reputation through gossip
- **Telegram Integration** -- real-time event notifications, hourly status reports, and remote `/ask` commands
- **Background Scheduler** -- idle-time maintenance with Ebbinghaus decay and automatic skill archival
- **LLM Agnostic** -- works with OpenAI, Anthropic, Ollama, and any provider supported by LiteLLM
- **Persistent Memory** -- agent state, skill metadata, and utility scores survive across sessions
- **Interactive REPL** -- conversational mode with history, slash commands, and live skill inspection
- **Docker Ready** -- run a multi-agent network with `docker compose up`

## Table of Contents

- [Quick Start](#quick-start)
- [CLI Commands](#cli-commands)
- [REPL Slash Commands](#repl-slash-commands)
- [Built-in Skills](#built-in-skills)
- [Skill Learning](#skill-learning)
- [P2P Network](#p2p-network)
- [Telegram Integration](#telegram-integration)
- [Configuration Reference](#configuration-reference)
- [Architecture](#architecture)
- [Docker Deployment](#docker-deployment)
- [Development](#development)
- [Project Structure](#project-structure)
- [License](#license)

## Quick Start

### 1. Install

```bash
# Core install
pip install -e .

# With all optional features (network + messaging)
pip install -e ".[full]"
```

### 2. Initialize workspace

```bash
evolvagent init
```

This creates `~/.evolvagent/` with a default `config.toml`, logs directory, and SQLite database.

### 3. Set your API key

```bash
# Pick one depending on your model
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Or use a local model -- no API key needed
# Edit ~/.evolvagent/config.toml -> model = "ollama/llama3"
```

### 4. Verify setup

```bash
evolvagent doctor
```

This checks config file, data directory, LLM model, API key, and dependencies.

### 5. Start the agent

```bash
# Interactive REPL (recommended)
evolvagent repl

# Or single question
evolvagent ask "What files changed in the last commit?"
```

## CLI Commands

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

Pass `--verbose` (`-v`) to any command for DEBUG-level logging.

## REPL Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/status` | Show agent status, scheduler, and network info |
| `/skills` | List registered skills with trust and utility |
| `/reflect` | Trigger reflection + skill learning cycle |
| `/teach` | Interactively teach the agent a new skill |
| `/network start\|stop` | Start or stop the P2P network server |
| `/peers` | List known and connected peers |
| `/connect host:port` | Connect to a peer |
| `/browse [query]` | Browse skills from connected peers |
| `/import <peer_id> <skill>` | Import a skill from a peer |
| `/messaging start\|stop` | Start or stop messaging bridge (Telegram) |
| `/context` | Generate/update CLAUDE.md |
| `/history` | Show conversation history length |
| `/clear` | Clear conversation history |
| `/exit` | Quit REPL |

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

**Trust levels:** `observe` (dry-run only) -> `suggest` (ask before executing) -> `auto` (execute freely). Skills promote automatically after reaching the configured success threshold (default: 10 consecutive successes).

## Skill Learning

EvolvAgent can create new skills in two ways:

### Manual teaching

Use `/teach` in the REPL to define a skill interactively:

```
/teach
  Name (snake_case): code_review
  Description: Review code for bugs and style issues
  Trigger phrases (comma-sep): review code, check code, code review
  System prompt: You are a senior code reviewer. Analyze the given code...
```

The new skill starts at `observe` trust and promotes through use.

### Automatic learning

During reflection, the agent analyzes request patterns where no skill matched. If recurring patterns are detected, the LLM synthesizes new skills automatically. This happens in the background during idle time -- no user intervention needed.

## P2P Network

Agents can form a decentralized network to discover and share skills.

### Getting started

```bash
# In REPL, start the network server
/network start

# Connect to another agent
/connect localhost:8766

# Browse available skills from peers
/browse

# Import a specific skill
/import <peer_id> code_review
```

### How it works

- Each agent runs a WebSocket server (default port `8765`) and connects to peers as a client
- Peers exchange skill catalogs and fetch full skill definitions on demand
- Gossip protocol propagates peer lists so agents discover each other transitively
- Imported skills always start at `observe` trust -- network skills must prove themselves locally
- Reputation feedback lets agents rate skills they imported from peers
- Optional bootstrap registry for automatic peer discovery in larger deployments

### Network configuration

```toml
[network]
listen_port = 8765
seed_peers = ["192.168.1.10:8765"]   # Known peers to connect on startup
auto_start = false                    # Set true to start network on agent boot
gossip_interval = 60                  # Seconds between gossip rounds
max_peers = 20
share_learned_skills = true           # Share user-taught skills with peers
share_builtin_skills = false          # Don't share built-in skills
bootstrap_registry = ""              # Optional registry URL for auto-discovery
```

## Telegram Integration

Connect EvolvAgent to Telegram for real-time monitoring and remote control. The agent sends event notifications, periodic status reports, and accepts commands via Telegram chat.

### Setup

1. **Create a bot** -- talk to [@BotFather](https://t.me/BotFather) on Telegram and run `/newbot`. Copy the bot token.

2. **Configure** -- add to your `config.toml`:

```toml
[messaging]
enabled = true

[messaging.telegram]
enabled = true
bot_token = "123456:ABC-DEF..."       # From @BotFather
allowed_chat_ids = []                  # Empty = allow all (dev mode)
```

Or use an environment variable instead of putting the token in config:

```bash
export EVOLVAGENT_TG_BOT_TOKEN="123456:ABC-DEF..."
```

3. **Install dependency**:

```bash
pip install evolvagent[messaging]
# or: pip install python-telegram-bot>=21.0
```

4. **Start the agent** -- the Telegram bot starts automatically:

```bash
evolvagent repl
```

5. **Open your bot** on Telegram and send `/start`.

### Telegram commands

| Command | Description |
|---------|-------------|
| `/status` | Agent status report (state, uptime, skills, stats) |
| `/skills` | List active skills with trust levels and utility |
| `/reflect` | Trigger a reflection + learning cycle |
| `/ask <query>` | Ask the agent a question |
| `/help` | Show available commands |
| Plain text | Treated as `/ask <text>` |

### Event notifications

The bot automatically sends notifications when events occur:

| Event | Example message |
|-------|-----------------|
| Skill executed | `Skill 'code_review' [OK] (320ms)` |
| State changed | `State: idle -> reflecting` |
| New skill learned | `New skill learned: 'data_analyzer'` |
| Skill taught | `Skill taught: 'my_custom_skill'` |
| Maintenance done | `Maintenance: decayed=2 archived=1` |
| Reflection started | `Reflection started...` |
| Reflection completed | `Reflection completed.` |

### Periodic status reports

By default, the agent sends a status summary every hour. Customize in config:

```toml
[messaging]
report_interval_seconds = 3600   # Every hour (0 = disabled)
```

### SUGGEST confirmation via Telegram

When a skill requires user confirmation (trust level = `suggest`), the bot sends an inline keyboard with Approve / Reject buttons. If no response within `confirm_timeout_seconds` (default 120), the action is auto-rejected.

### Security

- Set `allowed_chat_ids` to restrict which Telegram users can interact with the bot
- Empty list = allow all (for development only)
- Commands can be disabled entirely with `enable_commands = false` (notifications only)

### Manual start/stop in REPL

You can also start/stop messaging manually from the REPL:

```
/messaging start    # Start Telegram bot
/messaging stop     # Stop Telegram bot
/messaging          # Show status
```

## Configuration Reference

EvolvAgent reads `config.toml` from the current directory or `~/.evolvagent/config.toml`. Copy the example to get started:

```bash
cp config.toml.example config.toml
```

### Full configuration

```toml
[agent]
name = "agent-001"                     # Agent display name
description = "My personal dev assistant"
data_dir = "~/.evolvagent"             # Persistent data directory

[llm]
model = "anthropic/claude-haiku-4-5-20251001"  # Any LiteLLM model string
fallback_model = ""                    # Optional fallback model
max_tokens = 4096
temperature = 0.3
daily_cost_limit_usd = 1.0            # Daily spend cap

[trust]
default_level = "observe"              # New skills start here
promote_threshold = 10                 # Successes before auto-promotion

[scheduler]
cpu_threshold_percent = 70             # Skip maintenance if CPU above this
memory_threshold_percent = 80          # Skip maintenance if memory above this
idle_check_interval = 300              # Seconds between idle checks
min_idle_for_reflection = 600          # Seconds idle before triggering reflection

[evolution]
initial_utility = 0.5                  # Starting utility score for new skills
learning_rate = 0.1                    # How much each execution affects utility
decay_factor = 0.95                    # Ebbinghaus decay: utility *= factor ^ days_idle
archive_threshold = 0.2               # Archive skills below this utility
archive_after_days = 30               # ...after this many days unused

[knowledge]
db_name = "knowledge.db"              # SQLite database filename
vector_collection = "skills"
embedding_model = "default"

[logging]
level = "INFO"                         # DEBUG, INFO, WARNING, ERROR, CRITICAL
log_file = "evolvagent.log"
max_bytes = 5_000_000                  # Max log file size before rotation
backup_count = 3                       # Number of rotated log files to keep

[network]
listen_port = 8765                     # WebSocket server port
seed_peers = []                        # e.g. ["192.168.1.10:8765"]
auto_start = false                     # Start network on agent boot
gossip_interval = 60                   # Seconds between gossip rounds
max_peers = 20
share_learned_skills = true            # Share user-taught skills
share_builtin_skills = false           # Don't share built-in skills
bootstrap_registry = ""               # Registry URL for auto-discovery
bootstrap_heartbeat_interval = 60

[messaging]
enabled = false                        # Master switch for messaging
forward_events = [                     # Events sent to messaging adapters
    "skill.executed",
    "agent.state_changed",
    "scheduler.maintenance_completed",
    "skill.learned",
    "skill.taught",
    "agent.reflection_started",
    "agent.reflection_completed",
]
report_interval_seconds = 3600         # Periodic report interval (0 = disabled)
enable_commands = true                 # Allow remote commands via messaging
confirm_timeout_seconds = 120          # SUGGEST confirmation timeout

[messaging.telegram]
enabled = false                        # Enable Telegram adapter
bot_token = ""                         # From @BotFather, or use EVOLVAGENT_TG_BOT_TOKEN env var
allowed_chat_ids = []                  # Empty = allow all (dev mode only)
```

Supported model prefixes: `gpt-*`, `openai/*`, `anthropic/*`, `claude-*`, `ollama/*`, `gemini/*`, and [more](https://docs.litellm.ai/docs/providers).

## Architecture

```
User Request (CLI / Telegram)
  -> Agent.handle_request()
    -> Intent classification (LLM)
    -> Skill selection (best match by name + utility score)
    -> Trust enforcement (observe / suggest / auto)
    -> Skill.execute()
    -> Result + utility update + persist
    -> Record pattern for SkillLearner

Background (idle time)
  -> AgentScheduler._check_and_act()
    -> Resource check (CPU / memory)
    -> ReflectionEngine.reflect() -- LLM extracts principles from skill history
    -> SkillLearner.analyze_and_learn() -- create skills from recurring patterns
    -> Ebbinghaus decay -- unused skills lose utility over time
    -> Archival -- stale low-utility skills get archived

P2P Network
  -> NetworkServer (WebSocket)
    -> Peer discovery via seed_peers + gossip protocol
    -> Skill catalog exchange (browse / fetch / import)
    -> Reputation feedback between peers

Messaging Bridge
  -> EventBus events -> Formatter -> NotifierAdapter(s) -> Telegram / ...
  -> User messages  <- Command router <- NotifierAdapter(s) <- Telegram / ...
  -> ReportScheduler -> periodic status broadcast
```

### Core modules

| Module | Purpose |
|--------|---------|
| `evolvagent/core/agent.py` | Central orchestrator, state machine, skill management |
| `evolvagent/core/events.py` | Pub/sub event bus for inter-module communication |
| `evolvagent/core/skill.py` | Skill metadata, trust levels, utility scoring |
| `evolvagent/core/scheduler.py` | Background maintenance (reflection, decay, archival) |
| `evolvagent/core/reflection.py` | LLM-driven principle extraction |
| `evolvagent/core/learner.py` | Automatic skill creation from usage patterns |
| `evolvagent/core/network.py` | P2P WebSocket server and client |
| `evolvagent/core/peer.py` | Peer management and gossip protocol |
| `evolvagent/core/llm.py` | LLM abstraction via LiteLLM |
| `evolvagent/core/storage.py` | SQLite persistence layer |
| `evolvagent/core/config.py` | TOML configuration with dataclass validation |
| `evolvagent/messaging/base.py` | MessagingBridge + NotifierAdapter ABC |
| `evolvagent/messaging/telegram.py` | Telegram adapter (python-telegram-bot) |
| `evolvagent/messaging/formatter.py` | Event and status message formatting |
| `evolvagent/messaging/report.py` | Periodic status report scheduler |
| `evolvagent/cli/main.py` | CLI entry point and REPL |
| `evolvagent/skills/` | Built-in skill implementations |

### Design principles

- **EventBus-driven** -- all modules communicate through events, making it easy to add new integrations without modifying core logic
- **ImportError-safe** -- optional dependencies (websockets, python-telegram-bot) are imported lazily; missing packages produce a warning but don't break core functionality
- **Trust-first** -- every skill starts at `observe` and must earn autonomy; network-imported skills are always reset to `observe` regardless of their trust level on the source agent
- **Extensible adapters** -- the `NotifierAdapter` ABC makes it straightforward to add WhatsApp, Slack, or other platforms alongside Telegram

## Docker Deployment

### Single agent

```bash
docker build -t evolvagent .
docker run -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" -p 8765:8765 evolvagent
```

### Multi-agent network with Telegram

```bash
# Set environment variables
export ANTHROPIC_API_KEY="sk-ant-..."
export EVOLVAGENT_TG_BOT_TOKEN="123456:ABC-DEF..."   # Optional

# Start both agents + registry
docker compose up -d

# agent-alpha on port 8765, agent-beta on port 8766, registry on port 9000
```

The `docker-compose.yml` creates:
- **registry** -- peer discovery service on port 9000
- **agent-alpha** -- first agent on port 8765
- **agent-beta** -- second agent on port 8766

Each agent has a persistent volume and shares a bridge network (`agentnet`) for inter-agent communication. Both agents receive the Telegram bot token from the environment.

To enable Telegram in Docker, edit `config.toml.example` (mounted as `config.toml` in the container) and set `[messaging] enabled = true` and `[messaging.telegram] enabled = true`.

## Development

### Setup

```bash
# Install all dependencies (dev + network + messaging)
pip install -e ".[dev,network,messaging]"
```

### Running tests

```bash
# Full test suite (389 tests)
pytest tests/ -v

# Just messaging tests (46 tests)
pytest tests/test_messaging.py -v

# Lint
ruff check .
```

### Optional dependency groups

| Group | Dependencies | Purpose |
|-------|-------------|---------|
| `network` | `websockets>=12.0` | P2P network |
| `messaging` | `python-telegram-bot>=21.0` | Telegram integration |
| `full` | All of the above + pydantic | Everything |
| `dev` | pytest, pytest-asyncio, ruff | Development and testing |

## Project Structure

```
evolvagent/
  core/               # Agent engine, events, config, skills, LLM, storage
  messaging/           # Telegram integration (NotifierAdapter, bridge, formatter)
  skills/              # Built-in skill implementations
  cli/                 # CLI entry point and interactive REPL
  registry/            # Optional peer discovery registry server
tests/                 # 389 test cases
config.toml.example    # Full configuration reference
Dockerfile             # Container image
docker-compose.yml     # Multi-agent deployment
pyproject.toml         # Package metadata and dependencies
```

## License

MIT
