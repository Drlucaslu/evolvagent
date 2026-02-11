"""
Skill data structures and base classes.

A Skill is the fundamental unit of capability in EvolvAgent. This module
defines the Skill data model (incorporating utility scores from MemRL,
Ebbinghaus decay from SAGE, and failure classification from our paper review)
and the base class for Skill implementations.

Skill lifecycle: Capture → Execute → Evaluate → Evolve → (Archive | Propagate)
"""

from __future__ import annotations

import hashlib
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TrustLevel(str, Enum):
    """Progressive trust: observe → suggest → auto."""
    OBSERVE = "observe"    # Agent reports what it would do
    SUGGEST = "suggest"    # Agent suggests, user confirms
    AUTO = "auto"          # Agent executes autonomously

    def can_auto_execute(self) -> bool:
        return self == TrustLevel.AUTO

    def next_level(self) -> TrustLevel:
        order = [TrustLevel.OBSERVE, TrustLevel.SUGGEST, TrustLevel.AUTO]
        idx = order.index(self)
        return order[min(idx + 1, len(order) - 1)]


class SkillStatus(str, Enum):
    """Skill lifecycle status."""
    ACTIVE = "active"        # Available for use
    LEARNING = "learning"    # Being refined through use
    ARCHIVED = "archived"    # Decayed below threshold, stored but not active
    DISABLED = "disabled"    # Manually disabled by user


class FailureCategory(str, Enum):
    """
    Classification of failure experiences (from MemRL paper analysis).

    - NEAR_SUCCESS: Method was correct but details wrong. HIGH value.
    - REUSABLE: Provides "don't do this" warnings. MEDIUM value.
    - UNINFORMATIVE: Completely wrong direction. LOW value, eligible for decay.
    """
    NEAR_SUCCESS = "near_success"
    REUSABLE = "reusable"
    UNINFORMATIVE = "uninformative"


class SkillOrigin(str, Enum):
    """Where this Skill came from."""
    BUILTIN = "builtin"        # Shipped with EvolvAgent
    LEARNED = "learned"        # Extracted from user interactions
    NETWORK = "network"        # Acquired from another Agent
    USER_DEFINED = "user_defined"  # Explicitly created by user


# ---------------------------------------------------------------------------
# Failure Lesson (from MemRL paper insight)
# ---------------------------------------------------------------------------

@dataclass
class FailureLesson:
    """A classified failure experience attached to a Skill."""
    category: FailureCategory
    description: str
    root_cause: str
    task_context: str
    timestamp: float = field(default_factory=time.time)

    def is_valuable(self) -> bool:
        return self.category != FailureCategory.UNINFORMATIVE


# ---------------------------------------------------------------------------
# Skill metadata (the data that gets stored and transmitted)
# ---------------------------------------------------------------------------

@dataclass
class SkillMetadata:
    """
    Complete Skill data model.

    Incorporates:
    - utility_score (from MemRL): local effectiveness measure, updated per execution
    - network_reputation (from original design): network consensus score
    - last_used_at (for Ebbinghaus decay): tracks freshness
    - failure_lessons (from MemRL insight): classified failure experiences
    - trust_level (from progressive trust design): user-controlled permission level
    """

    # --- Identity ---
    skill_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    category: str = "general"
    description: str = ""
    version: str = "0.1.0"

    # --- Discovery ---
    tags: list[str] = field(default_factory=list)
    trigger_conditions: list[str] = field(default_factory=list)

    # --- Trust & permissions ---
    trust_level: TrustLevel = TrustLevel.OBSERVE
    interaction_mode: str = "passive"  # passive | active | background

    # --- Effectiveness (MemRL-inspired) ---
    utility_score: float = 0.5      # Local effectiveness [0.0, 1.0]
    network_reputation: float = 0.0  # Network consensus [0.0, 1.0]
    success_count: int = 0
    failure_count: int = 0
    total_executions: int = 0

    # --- Temporal (for Ebbinghaus decay) ---
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    last_reflected_at: float = 0.0

    # --- Learning ---
    failure_lessons: list[FailureLesson] = field(default_factory=list)
    distilled_principles: list[str] = field(default_factory=list)

    # --- Provenance ---
    origin: SkillOrigin = SkillOrigin.BUILTIN
    source_agent: str = ""  # Agent ID if acquired from network
    status: SkillStatus = SkillStatus.ACTIVE

    # --- Embedding (populated by vector store) ---
    embedding: list[float] | None = None

    # --- Computed properties ---

    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.success_count / self.total_executions

    @property
    def days_since_last_use(self) -> float:
        return (time.time() - self.last_used_at) / 86400

    @property
    def is_stale(self) -> bool:
        """Whether this Skill is a candidate for archival."""
        return self.days_since_last_use > 30 and self.utility_score < 0.2

    @property
    def valuable_failures(self) -> list[FailureLesson]:
        return [f for f in self.failure_lessons if f.is_valuable()]

    def content_hash(self) -> str:
        """Hash for deduplication."""
        content = f"{self.name}:{self.category}:{self.description}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # --- Mutation methods ---

    def record_execution(self, success: bool, reward: float = 0.0, learning_rate: float = 0.1):
        """
        Update utility after execution (MemRL-inspired sliding average).

        Args:
            success: Whether the task succeeded
            reward: Composite reward signal [0.0, 1.0]
            learning_rate: How fast utility adapts (from config.evolution.learning_rate)
        """
        self.total_executions += 1
        self.last_used_at = time.time()

        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        # Simple sliding average update (inspired by MemRL, without full MDP)
        if reward == 0.0:
            reward = 1.0 if success else 0.0
        self.utility_score += learning_rate * (reward - self.utility_score)
        self.utility_score = max(0.0, min(1.0, self.utility_score))

    def apply_decay(self, decay_factor: float = 0.95):
        """
        Apply Ebbinghaus-inspired decay based on time since last use.

        Call this periodically (e.g., daily). Skills that are used frequently
        resist decay; unused skills gradually fade.

        Formula: utility *= decay_factor ^ days_since_last_use
        (Applied incrementally, not cumulatively)
        """
        days = self.days_since_last_use
        if days > 1.0:  # Only decay if unused for > 1 day
            decay = decay_factor ** days
            self.utility_score *= decay
            self.utility_score = max(0.0, self.utility_score)

    def add_failure_lesson(self, lesson: FailureLesson):
        """Add a classified failure lesson, keeping max 10."""
        self.failure_lessons.append(lesson)
        # Keep only valuable ones + most recent uninformative
        valuable = [f for f in self.failure_lessons if f.is_valuable()]
        others = [f for f in self.failure_lessons if not f.is_valuable()]
        self.failure_lessons = valuable + others[-3:]  # Keep last 3 uninformative

    def promote_trust(self) -> bool:
        """Try to promote trust level. Returns True if promoted."""
        if self.trust_level == TrustLevel.AUTO:
            return False
        self.trust_level = self.trust_level.next_level()
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "version": self.version,
            "tags": self.tags,
            "trigger_conditions": self.trigger_conditions,
            "trust_level": self.trust_level.value,
            "interaction_mode": self.interaction_mode,
            "utility_score": self.utility_score,
            "network_reputation": self.network_reputation,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_executions": self.total_executions,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "last_reflected_at": self.last_reflected_at,
            "failure_lessons": [
                {
                    "category": f.category.value,
                    "description": f.description,
                    "root_cause": f.root_cause,
                    "task_context": f.task_context,
                    "timestamp": f.timestamp,
                }
                for f in self.failure_lessons
            ],
            "distilled_principles": self.distilled_principles,
            "origin": self.origin.value,
            "source_agent": self.source_agent,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillMetadata:
        """Deserialize from storage."""
        lessons = [
            FailureLesson(
                category=FailureCategory(f["category"]),
                description=f["description"],
                root_cause=f["root_cause"],
                task_context=f["task_context"],
                timestamp=f.get("timestamp", 0),
            )
            for f in data.get("failure_lessons", [])
        ]
        return cls(
            skill_id=data["skill_id"],
            name=data["name"],
            category=data.get("category", "general"),
            description=data.get("description", ""),
            version=data.get("version", "0.1.0"),
            tags=data.get("tags", []),
            trigger_conditions=data.get("trigger_conditions", []),
            trust_level=TrustLevel(data.get("trust_level", "observe")),
            interaction_mode=data.get("interaction_mode", "passive"),
            utility_score=data.get("utility_score", 0.5),
            network_reputation=data.get("network_reputation", 0.0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            total_executions=data.get("total_executions", 0),
            created_at=data.get("created_at", time.time()),
            last_used_at=data.get("last_used_at", time.time()),
            last_reflected_at=data.get("last_reflected_at", 0.0),
            failure_lessons=lessons,
            distilled_principles=data.get("distilled_principles", []),
            origin=SkillOrigin(data.get("origin", "builtin")),
            source_agent=data.get("source_agent", ""),
            status=SkillStatus(data.get("status", "active")),
        )


# ---------------------------------------------------------------------------
# Skill base class (for implementations)
# ---------------------------------------------------------------------------

class BaseSkill(ABC):
    """
    Abstract base class for Skill implementations.

    Each Skill has metadata (stored) and execution logic (code).
    Subclass this to create new Skills.
    """

    def __init__(self, metadata: SkillMetadata | None = None):
        self.metadata = metadata or SkillMetadata(name=self.__class__.__name__)

    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> SkillResult:
        """
        Execute the Skill.

        Args:
            context: Task context including user input, parameters, etc.

        Returns:
            SkillResult with success status, output, and optional feedback.
        """
        ...

    async def preview(self, context: dict[str, Any]) -> str:
        """
        Describe what this Skill would do without executing.

        Used for OBSERVE and SUGGEST trust levels. Override for specific previews.
        """
        return f"{self.metadata.name}: {self.metadata.description}"

    def can_handle(self, intent: str, context: dict[str, Any] | None = None) -> float:
        """
        Return confidence [0.0, 1.0] that this Skill can handle the given intent.

        Default implementation checks trigger_conditions. Override for smarter matching.
        """
        intent_lower = intent.lower()
        for trigger in self.metadata.trigger_conditions:
            if trigger.lower() in intent_lower:
                return 0.8
        return 0.0

    def __repr__(self) -> str:
        m = self.metadata
        return (
            f"<Skill {m.name} [{m.status.value}] "
            f"utility={m.utility_score:.2f} trust={m.trust_level.value}>"
        )


# ---------------------------------------------------------------------------
# Skill execution result
# ---------------------------------------------------------------------------

@dataclass
class SkillResult:
    """Result of a Skill execution."""
    success: bool
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    execution_time_ms: float = 0.0
    # Reward signal for utility update (0.0 = use success bool, otherwise explicit)
    reward: float = 0.0

    @property
    def effective_reward(self) -> float:
        if self.reward > 0:
            return self.reward
        return 1.0 if self.success else 0.0
