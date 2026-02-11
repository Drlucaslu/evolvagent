"""
LLM interface wrapper using LiteLLM.

Provides a unified interface for calling any LLM (OpenAI, Claude, Ollama, etc.)
with retry logic, cost tracking, daily budget limits, and automatic fallback.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""
    content: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    from_fallback: bool = False


@dataclass
class CostTracker:
    """Track daily LLM spending."""
    daily_limit_usd: float = 1.0
    _costs: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, cost)

    @property
    def today_cost(self) -> float:
        """Total cost for today."""
        now = time.time()
        day_start = now - (now % 86400)
        return sum(cost for ts, cost in self._costs if ts >= day_start)

    @property
    def is_over_budget(self) -> bool:
        return self.today_cost >= self.daily_limit_usd

    def record(self, cost: float) -> None:
        self._costs.append((time.time(), cost))
        # Prune old entries (keep last 7 days)
        cutoff = time.time() - 7 * 86400
        self._costs = [(ts, c) for ts, c in self._costs if ts > cutoff]


class LLMClient:
    """
    Unified LLM client with retry, cost tracking, and fallback.

    Uses LiteLLM under the hood, so any model string LiteLLM supports works:
    - "gpt-4o-mini", "gpt-4o"
    - "claude-sonnet-4-20250514"
    - "ollama/llama3", "ollama/mistral"
    - etc.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        fallback_model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
        daily_cost_limit_usd: float = 1.0,
        max_retries: int = 2,
    ):
        self.model = model
        self.fallback_model = fallback_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.cost_tracker = CostTracker(daily_limit_usd=daily_cost_limit_usd)

        # Stats
        self.total_calls: int = 0
        self.total_cost: float = 0.0
        self.total_errors: int = 0

    async def complete(
        self,
        prompt: str,
        system: str = "",
        history: list[dict[str, str]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Send a completion request to the LLM.

        Args:
            prompt: User message / prompt
            system: System prompt (optional)
            history: Prior conversation turns [{"role": "user"/"assistant", "content": "..."}]
            model: Override model for this call
            max_tokens: Override max tokens
            temperature: Override temperature

        Returns:
            LLMResponse with content and metadata

        Raises:
            LLMBudgetExceeded: If daily cost limit is reached
            LLMError: If all retries fail
        """
        if self.cost_tracker.is_over_budget:
            raise LLMBudgetExceeded(
                f"Daily LLM budget exceeded: ${self.cost_tracker.today_cost:.4f} "
                f"/ ${self.cost_tracker.daily_limit_usd:.2f}"
            )

        target_model = model or self.model
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        # Try primary model, then fallback
        last_error: Exception | None = None
        for attempt_model, is_fallback in [
            (target_model, False),
            (self.fallback_model, True),
        ]:
            if not attempt_model:
                continue
            if is_fallback:
                logger.warning("Falling back to model: %s", attempt_model)

            for attempt in range(self.max_retries + 1):
                try:
                    response = await self._call_litellm(
                        model=attempt_model,
                        messages=messages,
                        max_tokens=max_tokens or self.max_tokens,
                        temperature=temperature if temperature is not None else self.temperature,
                        **kwargs,
                    )
                    response.from_fallback = is_fallback
                    return response

                except Exception as e:
                    last_error = e
                    self.total_errors += 1
                    if attempt < self.max_retries:
                        wait = 2 ** attempt  # Exponential backoff
                        logger.warning(
                            "LLM call failed (attempt %d/%d), retrying in %ds: %s",
                            attempt + 1, self.max_retries + 1, wait, e,
                        )
                        import asyncio
                        await asyncio.sleep(wait)
                    else:
                        logger.error("LLM call failed after %d attempts: %s", attempt + 1, e)

        raise LLMError(f"All LLM attempts failed. Last error: {last_error}")

    async def _call_litellm(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> LLMResponse:
        """Make the actual LiteLLM API call."""
        try:
            import litellm
        except ImportError:
            raise LLMError(
                "litellm is not installed. Run: pip install litellm"
            )

        start = time.time()

        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        except litellm.AuthenticationError as e:
            raise LLMError(
                f"API key not configured for model '{model}'. "
                f"Set the appropriate environment variable "
                f"(e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY) "
                f"or switch to a local model like 'ollama/llama3'.\n"
                f"Run 'evolvagent doctor' to check your setup."
            ) from e
        except (litellm.APIConnectionError, ConnectionError, OSError) as e:
            raise LLMError(
                f"Could not connect to LLM provider for model '{model}'. "
                f"Check your network connection and that the API endpoint is reachable.\n"
                f"Run 'evolvagent doctor' to check your setup."
            ) from e

        latency_ms = (time.time() - start) * 1000

        # Extract response data
        content = response.choices[0].message.content or ""
        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        # Calculate cost using LiteLLM's cost tracking
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

        # Record
        self.total_calls += 1
        self.total_cost += cost
        self.cost_tracker.record(cost)

        logger.debug(
            "LLM call: model=%s tokens=%d+%d cost=$%.6f latency=%.0fms",
            model, tokens_in, tokens_out, cost, latency_ms,
        )

        return LLMResponse(
            content=content,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

    def stats_dict(self) -> dict[str, Any]:
        """Return usage statistics."""
        return {
            "model": self.model,
            "fallback_model": self.fallback_model,
            "total_calls": self.total_calls,
            "total_cost_usd": round(self.total_cost, 6),
            "today_cost_usd": round(self.cost_tracker.today_cost, 6),
            "daily_limit_usd": self.cost_tracker.daily_limit_usd,
            "total_errors": self.total_errors,
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """General LLM error."""
    pass


class LLMBudgetExceeded(LLMError):
    """Daily cost limit reached."""
    pass
