# EvolvAgent

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/Drlucaslu/evolvagent/actions/workflows/ci.yml/badge.svg)

A decentralized self-evolving agent network where agents learn, reflect, and share skills.

- **Natural Language Interface** — ask questions and give instructions in plain English
- **Skill-Based Architecture** — modular skills for shell, git, file search, summarization, and more
- **Progressive Trust** — skills start in `observe` mode and earn autonomy through successful executions
- **LLM-Driven Reflection** — agent periodically analyzes skill history and extracts reusable principles
- **Skill Learning** — agent automatically creates new skills from usage patterns, or learn via `/teach`
- **P2P Network** — agents discover peers, share skills over WebSocket, and build reputation through gossip
- **Background Scheduler** — idle-time maintenance with Ebbinghaus decay and automatic skill archival
- **LLM Agnostic** — works with OpenAI, Anthropic, Ollama, and any provider supported by LiteLLM
- **Persistent Memory** — agent state, skill metadata, and utility scores survive across sessions
- **Interactive REPL** — conversational mode with history, slash commands, and live skill inspection
- **Docker Ready** — run a multi-agent network with `docker compose up`

## Quick Start

```bash
# Install
pip install -e .

# Initialize workspace
evolvagent init

# Set your API key (pick one)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Or use a local model — no API key needed
# Edit ~/.evolvagent/config.toml -> model = "ollama/llama3"

# Check setup
evolvagent doctor

# Start the interactive REPL
evolvagent repl
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

**Trust levels:** `observe` (dry-run only) → `suggest` (ask before executing) → `auto` (execute freely). Skills promote automatically after reaching the configured success threshold.

## Skill Learning

EvolvAgent can create new skills in two ways:

**Manual teaching** — use `/teach` in the REPL to define a skill interactively:

```
/teach
  Name (snake_case): code_review
  Description: Review code for bugs and style issues
  Trigger phrases (comma-sep): review code, check code, code review
  System prompt: You are a senior code reviewer. Analyze the given code...
```

**Automatic learning** — during reflection, the agent analyzes request patterns where no skill matched and uses the LLM to synthesize new skills from recurring patterns. Learned skills start at `observe` trust and promote through use like any other skill.

## P2P Network

Agents can form a decentralized network to discover and share skills:

```bash
# Start network in REPL
/network start

# Connect to another agent
/connect localhost:8766

# Browse and import skills
/browse
/import <peer_id> code_review
```

**How it works:**
- Each agent runs a WebSocket server (default port `8765`) and connects to peers as a client
- Peers exchange skill catalogs and fetch full skill definitions on demand
- Gossip protocol propagates peer lists so agents discover each other transitively
- Imported skills always start at `observe` trust — network skills must prove themselves locally
- Reputation feedback lets agents rate skills they imported from peers

## Configuration

EvolvAgent reads `config.toml` from the current directory or `~/.evolvagent/config.toml`. The repo ships `config.toml.example` — copy and customize it:

```bash
cp config.toml.example config.toml
```

```toml
[agent]
name = "agent-001"
description = "My personal development assistant"
data_dir = "~/.evolvagent"

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

[logging]
level = "INFO"
log_file = "evolvagent.log"
max_bytes = 5_000_000
backup_count = 3

[network]
listen_port = 8765
seed_peers = []                 # e.g. ["192.168.1.10:8765"]
auto_start = false
gossip_interval = 60            # Seconds between gossip rounds
max_peers = 20
share_learned_skills = true
share_builtin_skills = false
```

Supported model prefixes: `gpt-*`, `openai/*`, `anthropic/*`, `claude-*`, `ollama/*`, `gemini/*`, and [more](https://docs.litellm.ai/docs/providers).

## Architecture

```
User Request
  → Agent.handle_request()
    → Intent classification (LLM)
    → Skill selection (best match by name + utility score)
    → Trust enforcement (observe / suggest / auto)
    → Skill.execute()
    → Result + utility update + persist
    → Record pattern for SkillLearner

Background (idle time)
  → AgentScheduler._check_and_act()
    → Resource check (CPU / memory)
    → ReflectionEngine.reflect() — LLM extracts principles from skill history
    → SkillLearner.analyze_and_learn() — create skills from recurring patterns
    → Ebbinghaus decay — unused skills lose utility over time
    → Archival — stale low-utility skills get archived

P2P Network
  → NetworkServer (WebSocket)
    → Peer discovery via seed_peers + gossip protocol
    → Skill catalog exchange (browse / fetch / import)
    → Reputation feedback between peers
```

The agent maintains a registry of skills. Each skill has metadata tracking its utility score, execution count, and trust level. After every execution the utility score is updated and the skill metadata is persisted to disk. Over time, high-performing skills are promoted and low-utility skills decay. During idle time, the reflection engine distills reusable principles and the learner creates new skills from usage patterns.

## Docker Deployment

Build and run a single agent:

```bash
docker build -t evolvagent .
docker run -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" -p 8765:8765 evolvagent
```

Run a two-agent network with `docker compose`:

```bash
# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Start both agents
docker compose up -d

# agent-alpha on port 8765, agent-beta on port 8766
```

The `docker-compose.yml` creates two agents on a shared bridge network (`agentnet`) with persistent volumes for each agent's data.

## Development

```bash
# Install dev + network dependencies
pip install -e ".[dev,network]"

# Run tests
pytest tests/ -v

# Lint
ruff check .
```

## License

MIT
