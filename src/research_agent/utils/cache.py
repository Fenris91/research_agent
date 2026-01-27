"""
API Response Cache

Provides caching for API responses to reduce rate limit issues
and improve performance.

Features:
- In-memory TTL cache (no external dependencies)
- Optional disk persistence using shelve
- Thread-safe operations
- Configurable TTL and max size
"""

import asyncio
import hashlib
import json
import logging
import shelve
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached value with expiration time."""
    value: Any
    expires_at: float
    created_at: float

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class TTLCache:
    """
    Thread-safe in-memory cache with TTL (time-to-live) support.

    Example:
        cache = TTLCache(default_ttl=3600)  # 1 hour default

        # Cache a value
        cache.set("key", {"data": "value"})

        # Get cached value (returns None if expired/missing)
        value = cache.get("key")

        # Get or compute
        value = await cache.get_or_set("key", expensive_function)
    """

    def __init__(
        self,
        default_ttl: int = 3600,
        max_size: int = 1000,
        cleanup_interval: int = 300
    ):
        """
        Initialize cache.

        Args:
            default_ttl: Default time-to-live in seconds (default: 1 hour)
            max_size: Maximum number of entries (default: 1000)
            cleanup_interval: Seconds between cleanup runs (default: 5 min)
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval

        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()

        # Stats
        self._hits = 0
        self._misses = 0

    def _maybe_cleanup(self):
        """Run cleanup if enough time has passed."""
        now = time.time()
        if now - self._last_cleanup > self.cleanup_interval:
            self._cleanup()
            self._last_cleanup = now

    def _cleanup(self):
        """Remove expired entries and enforce max size."""
        now = time.time()
        expired_keys = [
            k for k, v in self._cache.items()
            if v.expires_at < now
        ]
        for key in expired_keys:
            del self._cache[key]

        # If still over max size, remove oldest entries
        if len(self._cache) > self.max_size:
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].created_at
            )
            to_remove = len(self._cache) - self.max_size
            for key, _ in sorted_entries[:to_remove]:
                del self._cache[key]

        if expired_keys:
            logger.debug(f"Cache cleanup: removed {len(expired_keys)} expired entries")

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            self._maybe_cleanup()

            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None

            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override in seconds
        """
        with self._lock:
            self._maybe_cleanup()

            ttl = ttl if ttl is not None else self.default_ttl
            now = time.time()

            self._cache[key] = CacheEntry(
                value=value,
                expires_at=now + ttl,
                created_at=now
            )

    def delete(self, key: str) -> bool:
        """
        Delete a key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was found and deleted
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")

    async def get_or_set(
        self,
        key: str,
        factory: Callable,
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get from cache or compute and cache the value.

        Args:
            key: Cache key
            factory: Async function to compute value if not cached
            ttl: Optional TTL override

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        # Compute value
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        self.set(key, value, ttl)
        return value

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.1%}",
                "default_ttl": self.default_ttl
            }


class PersistentCache(TTLCache):
    """
    TTL cache with disk persistence using shelve.

    Data is persisted to disk and survives restarts.
    Good for caching expensive API calls across sessions.

    Example:
        cache = PersistentCache(
            cache_dir="./data/cache",
            name="semantic_scholar"
        )
    """

    def __init__(
        self,
        cache_dir: str = "./data/cache",
        name: str = "api_cache",
        default_ttl: int = 3600,
        max_size: int = 5000,
        sync_interval: int = 60
    ):
        """
        Initialize persistent cache.

        Args:
            cache_dir: Directory for cache files
            name: Cache name (used for filename)
            default_ttl: Default TTL in seconds
            max_size: Maximum entries
            sync_interval: Seconds between disk syncs
        """
        super().__init__(default_ttl, max_size)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / name
        self.sync_interval = sync_interval
        self._last_sync = time.time()

        # Load from disk
        self._load()

    def _load(self):
        """Load cache from disk."""
        try:
            with shelve.open(str(self.cache_path)) as db:
                for key, entry_dict in db.items():
                    entry = CacheEntry(**entry_dict)
                    if not entry.is_expired:
                        self._cache[key] = entry

            logger.info(f"Loaded {len(self._cache)} entries from cache: {self.cache_path}")
        except Exception as e:
            logger.warning(f"Could not load cache from disk: {e}")

    def _save(self):
        """Save cache to disk."""
        try:
            with shelve.open(str(self.cache_path)) as db:
                db.clear()
                for key, entry in self._cache.items():
                    if not entry.is_expired:
                        db[key] = {
                            "value": entry.value,
                            "expires_at": entry.expires_at,
                            "created_at": entry.created_at
                        }

            logger.debug(f"Saved {len(self._cache)} entries to cache")
        except Exception as e:
            logger.warning(f"Could not save cache to disk: {e}")

    def _maybe_sync(self):
        """Sync to disk if enough time has passed."""
        now = time.time()
        if now - self._last_sync > self.sync_interval:
            self._save()
            self._last_sync = now

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value and maybe sync to disk."""
        super().set(key, value, ttl)
        self._maybe_sync()

    def close(self):
        """Save and close the cache."""
        with self._lock:
            self._save()
            logger.info(f"Cache closed: {self.cache_path}")


def make_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    Create a cache key from arguments.

    Args:
        prefix: Key prefix (e.g., "s2_search", "openalex_paper")
        *args: Positional arguments to include
        **kwargs: Keyword arguments to include

    Returns:
        Deterministic cache key string
    """
    # Sort kwargs for deterministic ordering
    sorted_kwargs = sorted(kwargs.items())

    # Build key string
    key_parts = [prefix]
    key_parts.extend(str(a) for a in args)
    key_parts.extend(f"{k}={v}" for k, v in sorted_kwargs)

    key_string = "|".join(key_parts)

    # Hash long keys
    if len(key_string) > 200:
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{key_hash}"

    return key_string


# Global cache instances (lazy initialized)
_memory_cache: Optional[TTLCache] = None
_persistent_cache: Optional[PersistentCache] = None


def get_memory_cache(ttl: int = 3600, max_size: int = 1000) -> TTLCache:
    """Get or create the global memory cache."""
    global _memory_cache
    if _memory_cache is None:
        _memory_cache = TTLCache(default_ttl=ttl, max_size=max_size)
    return _memory_cache


def get_persistent_cache(
    cache_dir: str = "./data/cache",
    ttl: int = 86400  # 24 hours default for disk cache
) -> PersistentCache:
    """Get or create the global persistent cache."""
    global _persistent_cache
    if _persistent_cache is None:
        _persistent_cache = PersistentCache(
            cache_dir=cache_dir,
            name="api_responses",
            default_ttl=ttl
        )
    return _persistent_cache
