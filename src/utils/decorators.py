"""
Decorators Utilities

Reusable decorators for the JPMorgan European Equity Dashboard:

- @timeit          : Measure & log execution time
- @retry           : Robust retry with backoff (for API calls, DB ops)
- @log_exceptions  : Log exceptions with context (optionally re-raise)
- @cached          : Simple in-memory cache with TTL
- @ensure_coroutine: Wrap sync functions as async for integration

These are designed for production use in analytics, data pipelines,
and API layers.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, Dict, Hashable, Optional, Type, TypeVar, Union, Tuple

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# timeit
# ---------------------------------------------------------------------------

def timeit(name: Optional[str] = None, level: int = logging.INFO) -> Callable[[F], F]:
    """
    Decorator to measure and log function execution time.

    Args:
        name: Optional label. If None, uses function.__qualname__.
        level: Logging level used for message.

    Example:
        @timeit("load_market_data")
        def load_market_data(...):
            ...
    """

    def decorator(func: F) -> F:
        label = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                logger.log(level, "TIMEIT %s took %.4f seconds", label, duration)

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------

def retry(
    exceptions: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = Exception,
    *,
    tries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: Optional[float] = None,
    logger_: Optional[logging.Logger] = None,
) -> Callable[[F], F]:
    """
    Decorator providing retry with exponential backoff.

    Args:
        exceptions: Exception or tuple of exceptions to catch.
        tries: Max number of attempts (including the first).
        delay: Initial delay between retries, in seconds.
        backoff: Multiplier for delay (exponential backoff).
        max_delay: Optional max delay cap.
        logger_: Optional custom logger (defaults to module logger).

    Example:
        @retry((ConnectionError, TimeoutError), tries=5, delay=1, backoff=1.5)
        def fetch_data(...):
            ...
    """
    exceptions_tuple: Tuple[Type[BaseException], ...]
    if isinstance(exceptions, type) and issubclass(exceptions, BaseException):
        exceptions_tuple = (exceptions,)
    else:
        exceptions_tuple = exceptions  # type: ignore[assignment]

    log = logger_ or logger

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _tries = max(1, tries)
            _delay = max(0.0, delay)

            for attempt in range(1, _tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions_tuple as exc:
                    if attempt >= _tries:
                        log.error(
                            "RETRY: %s failed on attempt %d/%d. No more retries.",
                            func.__qualname__,
                            attempt,
                            _tries,
                            exc_info=True,
                        )
                        raise
                    else:
                        sleep_for = _delay
                        if max_delay is not None:
                            sleep_for = min(sleep_for, max_delay)
                        log.warning(
                            "RETRY: %s failed on attempt %d/%d. Retrying in %.2f s. Error: %s",
                            func.__qualname__,
                            attempt,
                            _tries,
                            sleep_for,
                            exc,
                        )
                        time.sleep(sleep_for)
                        _delay *= backoff

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# log_exceptions
# ---------------------------------------------------------------------------

def log_exceptions(
    *,
    swallow: bool = False,
    level: int = logging.ERROR,
    message: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator that logs exceptions raised by the wrapped function.

    Args:
        swallow: If True, do not re-raise the exception and return None.
        level: Logging level for the error.
        message: Optional custom message prefix.

    Example:
        @log_exceptions(swallow=False)
        def risky_op(...):
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                msg_prefix = message or f"Exception in {func.__qualname__}"
                logger.log(
                    level,
                    "%s: %s",
                    msg_prefix,
                    exc,
                    exc_info=True,
                )
                if not swallow:
                    raise
                return None

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# cached (TTL cache)
# ---------------------------------------------------------------------------

class _TTLCache:
    """Simple thread-safe TTL cache."""

    def __init__(self, ttl_seconds: float) -> None:
        self.ttl = ttl_seconds
        self._store: Dict[Hashable, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Any:
        with self._lock:
            if key not in self._store:
                raise KeyError
            ts, value = self._store[key]
            if time.time() - ts > self.ttl:
                del self._store[key]
                raise KeyError
            return value

    def set(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.time(), value)


def _make_cache_key(func: Callable[..., Any], args: Any, kwargs: Any) -> Hashable:
    """
    Build a hashable cache key.

    Note: This is a simple implementation. Avoid passing unhashable
    types (like dicts) as positional args if you want caching to work.
    """
    return (func.__module__, func.__qualname__, args, frozenset(kwargs.items()))


def cached(ttl: float = 300.0) -> Callable[[F], F]:
    """
    Decorator that caches function result in memory for `ttl` seconds.

    Important:
        - Args must be hashable (or you'll get a TypeError).
        - Cache is per-process and not shared across workers.

    Example:
        @cached(ttl=600)
        def load_reference_data(...):
            ...
    """
    cache = _TTLCache(ttl_seconds=ttl)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = _make_cache_key(func, args, kwargs)
            try:
                return cache.get(key)
            except KeyError:
                result = func(*args, **kwargs)
                cache.set(key, result)
                return result

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# ensure_coroutine
# ---------------------------------------------------------------------------

def ensure_coroutine(func: F) -> F:
    """
    Wrap a sync function so it can be awaited as a coroutine.

    If `func` is already async, it is returned unchanged.

    Useful for mixing sync utils into async environments (FastAPI, websockets).

    Example:
        @ensure_coroutine
        def compute_factor_exposures(...):
            ...

        async def handler():
            exposures = await compute_factor_exposures(...)
    """

    if asyncio.iscoroutinefunction(func):
        return func

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

    return async_wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    @timeit("demo_sleep")
    @retry(tries=3, delay=0.1)
    @log_exceptions(swallow=False)
    def demo_sleep(x: int) -> int:
        if x < 0:
            raise ValueError("x must be non-negative")
        time.sleep(0.05)
        return x * 2

    print("demo_sleep(2) ->", demo_sleep(2))

    @cached(ttl=1.0)
    def add(a: int, b: int) -> int:
        print("Computing add(...)")
        return a + b

    print(add(1, 2))
    print(add(1, 2))  # cached

    async def main() -> None:
        @ensure_coroutine
        def sync_double(v: int) -> int:
            return v * 2

        res = await sync_double(21)
        print("async sync_double ->", res)

    asyncio.run(main())
