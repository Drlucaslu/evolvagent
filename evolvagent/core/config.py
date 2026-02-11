"""
Configuration management for EvolvAgent.

Loads settings from TOML config file with dataclass validation.
Pure standard library — no external dependencies.
"""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Sub-models (nested config sections)
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    name: str = "agent-001"
    description: str = "My personal development assistant"
    data_dir: str = "~/.evolvagent"

    @property
    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    fallback_model: str = "ollama/llama3"
    max_tokens: int = 4096
    temperature: float = 0.3
    daily_cost_limit_usd: float = 1.0


@dataclass
class SchedulerConfig:
    cpu_threshold_percent: int = 70
    memory_threshold_percent: int = 80
    idle_check_interval: int = 300
    min_idle_for_reflection: int = 600


@dataclass
class KnowledgeConfig:
    db_name: str = "knowledge.db"
    vector_collection: str = "skills"
    embedding_model: str = "default"


@dataclass
class EvolutionConfig:
    initial_utility: float = 0.5
    learning_rate: float = 0.1
    decay_factor: float = 0.95
    archive_threshold: float = 0.2
    archive_after_days: int = 30


@dataclass
class TrustConfig:
    default_level: str = "observe"  # observe | suggest | auto
    promote_threshold: int = 10


@dataclass
class NetworkConfig:
    marketplace_url: str = ""
    listen_port: int = 8765


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    """Root configuration, assembled from TOML file."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATHS = [
    Path("config.toml"),
    Path("~/.evolvagent/config.toml").expanduser(),
]


def _make_section(cls, data: dict) -> object:
    """Create a dataclass instance from a dict, ignoring unknown keys."""
    import dataclasses
    valid_keys = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from TOML file, falling back to defaults."""
    raw: dict = {}

    if config_path and config_path.exists():
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)
    else:
        for p in DEFAULT_CONFIG_PATHS:
            if p.exists():
                with open(p, "rb") as f:
                    raw = tomllib.load(f)
                break

    section_map = {
        "agent": AgentConfig,
        "llm": LLMConfig,
        "scheduler": SchedulerConfig,
        "knowledge": KnowledgeConfig,
        "evolution": EvolutionConfig,
        "trust": TrustConfig,
        "network": NetworkConfig,
    }

    kwargs = {}
    for key, cls in section_map.items():
        if key in raw:
            kwargs[key] = _make_section(cls, raw[key])

    return Settings(**kwargs)


# Module-level singleton (lazy-initialized)
_settings: Settings | None = None


def get_settings(config_path: Path | None = None) -> Settings:
    """Get or create the global settings singleton."""
    global _settings
    if _settings is None:
        _settings = load_settings(config_path)
    return _settings


def reset_settings() -> None:
    """Reset the global settings singleton (for testing)."""
    global _settings
    _settings = None
