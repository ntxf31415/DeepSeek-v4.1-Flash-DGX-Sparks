"""Report real bytes for the KV pool that backs `sglang:kv_cache_memory_usage_gb`.

`UnifiedSWAKVPool.__init__` hard-codes `self.mem_usage = 0.0` (upstream marks it
"cosmetic; UnifiedKVPool logs the real size"), so on this stack the scheduler reads
`token_to_kv_pool_allocator.get_kvcache().mem_usage` as 0 and the metric, the load
inquirer, and the server-info payload all report zero KV bytes while the unified
pool holds tens of GB. Upstream fixed this by adding per-pool tensor accounting
(sgl-project/sglang#37935); this adapter lands the same number without forking the
pool classes.

The pool keeps the authoritative byte count already: `UnifiedKVPool(total_bytes=...)`
sums every sub-pool it owns (full / swa / mamba), and `UnifiedSWAKVPool` holds that
buffer on `self.unified_buffer`. Multiply-free: reuse it.

Scope: every pool the DSV4 stack puts behind `get_kvcache()`, so the metric is
correct whichever family the config selects. Idempotent, and it never raises into a
constructor: on any surprise it logs and leaves `mem_usage` as upstream left it.
"""
import logging

logger = logging.getLogger(__name__)

GB = 1024 ** 3

# The pools `kv_cache_configurator` can hand back through `get_kvcache()`.
_TARGETS = (
    'UnifiedSWAKVPool',
    'UnifiedMambaSWATokenToKVPoolAllocator',
    'UnifiedSWATokenToKVPoolAllocator',
    'UnifiedMambaTokenToKVPoolAllocator',
)


def _bytes_of(pool):
    """Authoritative byte count for a pool, or None when it does not carry one."""
    # UnifiedKVPool: the summed allocation of every sub-pool it owns.
    for holder in (pool, getattr(pool, 'unified_buffer', None)):
        total = getattr(holder, 'total_bytes', None)
        if isinstance(total, int) and total > 0:
            return total
    return None


def _install_one(cls):
    original = cls.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        try:
            total = _bytes_of(self)
            if total is None:
                logger.warning(
                    'DSV41 kv mem_usage: %s has no total_bytes (unified_buffer=%s); '
                    'metric stays 0', cls.__name__,
                    type(getattr(self, 'unified_buffer', None)).__name__)
                return
            self.mem_usage = total / GB
            logger.warning('DSV41 kv mem_usage: %s -> %.3f GB (from total_bytes=%d)',
                           cls.__name__, self.mem_usage, total)
        except Exception as exc:  # bookkeeping must never break a constructor
            logger.warning('DSV41 kv mem_usage: %s failed: %s', cls.__name__, exc)

    cls.__init__ = __init__


def install(module):
    installed = []
    for name in _TARGETS:
        cls = getattr(module, name, None)
        if cls is not None:
            _install_one(cls)
            installed.append(name)
    logger.warning('DSV41 kv mem_usage adapter installed for: %s', ', '.join(installed))
