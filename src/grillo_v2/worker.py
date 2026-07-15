from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .runtime import GrilloV2Runtime, WorkerTickResult


TickCallback = Callable[[WorkerTickResult], Awaitable[None] | None]
ErrorCallback = Callable[[Exception], Awaitable[None] | None]
SleepFunction = Callable[[float], Awaitable[Any]]


@dataclass(slots=True)
class GrilloV2WorkerConfig:
    interval_seconds: float = 60.0
    initial_delay_seconds: float = 0.0
    scope_key: str | None = None
    scope_limit: int = 10
    batch_size: int = 12
    max_batches: int = 3
    max_consecutive_errors: int = 3


class GrilloV2Worker:
    def __init__(
        self,
        *,
        runtime: GrilloV2Runtime,
        config: GrilloV2WorkerConfig | None = None,
        on_tick: TickCallback | None = None,
        on_error: ErrorCallback | None = None,
        sleep: SleepFunction = asyncio.sleep,
    ):
        self.runtime = runtime
        self.config = config or GrilloV2WorkerConfig()
        self.on_tick = on_tick
        self.on_error = on_error
        self.sleep = sleep
        self.ticks = 0
        self.consecutive_errors = 0
        self.last_result: WorkerTickResult | None = None
        self.last_error: Exception | None = None
        self.running = False

    async def run_once(self) -> WorkerTickResult:
        result = await self.runtime.worker_tick(
            scope_key=self.config.scope_key,
            scope_limit=self.config.scope_limit,
            batch_size=self.config.batch_size,
            max_batches=self.config.max_batches,
        )
        self.ticks += 1
        self.consecutive_errors = 0
        self.last_error = None
        self.last_result = result
        await _maybe_await(self.on_tick, result)
        return result

    async def run_forever(self, *, max_ticks: int | None = None) -> None:
        self.running = True
        try:
            if self.config.initial_delay_seconds > 0:
                await self.sleep(self.config.initial_delay_seconds)
            while max_ticks is None or self.ticks < max_ticks:
                try:
                    await self.run_once()
                except Exception as exc:
                    self.consecutive_errors += 1
                    self.last_error = exc
                    await _maybe_await(self.on_error, exc)
                    if self.consecutive_errors >= max(1, int(self.config.max_consecutive_errors)):
                        raise
                if max_ticks is not None and self.ticks >= max_ticks:
                    break
                await self.sleep(max(0.1, float(self.config.interval_seconds)))
        finally:
            self.running = False


async def _maybe_await(callback, value) -> None:
    if callback is None:
        return
    result = callback(value)
    if result is not None and hasattr(result, "__await__"):
        await result
