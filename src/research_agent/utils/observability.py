"""Lightweight observability: request IDs and timing spans."""

import contextvars
import functools
import logging
import time
import uuid

logger = logging.getLogger(__name__)

# Context variable for request correlation
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def new_request_id() -> str:
    """Generate and set a new request ID for the current context."""
    rid = uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


def get_request_id() -> str:
    """Get the current request ID (empty string if none set)."""
    return _request_id.get()


def timed(func=None, *, level=logging.DEBUG):
    """Decorator that logs function execution time.

    Usage:
        @timed
        def my_func(): ...

        @timed(level=logging.INFO)
        async def my_async_func(): ...

    Logs: [req_id] module.function took Xms
    """
    def decorator(fn):
        name = f"{fn.__module__}.{fn.__qualname__}"

        if asyncio_iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                rid = _request_id.get()
                prefix = f"[{rid}] " if rid else ""
                start = time.perf_counter()
                try:
                    return await fn(*args, **kwargs)
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    logger.log(level, f"{prefix}{name} took {elapsed_ms:.0f}ms")
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                rid = _request_id.get()
                prefix = f"[{rid}] " if rid else ""
                start = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    logger.log(level, f"{prefix}{name} took {elapsed_ms:.0f}ms")
            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def asyncio_iscoroutinefunction(fn):
    """Check if fn is an async function (avoids importing asyncio at module level)."""
    import asyncio
    return asyncio.iscoroutinefunction(fn)
