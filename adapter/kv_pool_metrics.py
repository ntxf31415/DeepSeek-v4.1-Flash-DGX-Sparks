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
import os

logger = logging.getLogger(__name__)

GB = 1024 ** 3

# The pools `kv_cache_configurator` can hand back through `get_kvcache()`.
_TARGETS = (
    'UnifiedSWAKVPool',
    'UnifiedMambaSWATokenToKVPoolAllocator',
    'UnifiedSWATokenToKVPoolAllocator',
    'UnifiedMambaTokenToKVPoolAllocator',
    # What DSV4 actually builds. Verified from the live engine, which reported
    # allocator=SWATokenToKVPoolAllocator, kvcache=DeepSeekV4TokenToKVPool,
    # kvcache_has_total_bytes=False -- none of the Unified* names above are in
    # the DSV4 path at all.
    'DeepSeekV4TokenToKVPool',
)


_BUFFER_ATTRS = ('k_buffer', 'v_buffer', 'kv_buffer',
                 'k_scale_buffer', 'v_scale_buffer')


def _nbytes_of(obj):
    """Bytes held by one tensor, or by a list/tuple of them."""
    if obj is None:
        return 0
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes_of(x) for x in obj)
    nbytes = getattr(obj, 'nbytes', None)
    return nbytes if isinstance(nbytes, int) else 0


def _sum_buffer_bytes(obj):
    total = 0
    for name in _BUFFER_ATTRS:
        total += _nbytes_of(getattr(obj, name, None))
    return total


def _holders_of(pool):
    """(name, bytes) for every attribute of `pool` that holds KV bytes.

    DSV4's pool keeps no buffers of its own -- it fans out over sub-pools whose
    layout depends on the config, and the live engine showed the full pool is
    not the SWA one. Walking the attributes and summing reports the real total
    and the breakdown, so a layout change is visible instead of silently wrong.
    """
    out = []
    for name, value in vars(pool).items():
        if value is None:
            continue
        b = _sum_buffer_bytes(value)
        if b == 0 and hasattr(value, 'get_buf_infos'):
            try:
                b = sum(value.get_buf_infos()[1])
            except Exception:
                b = 0
        if b > 0:
            out.append((name, b))
    own = _sum_buffer_bytes(pool)
    if own > 0:
        out.append(('self', own))
    return out


def _bytes_of(pool):
    """Total KV bytes for the pool, plus a breakdown string for the log."""
    holders = _holders_of(pool)
    if holders:
        total = sum(b for _n, b in holders)
        detail = ' '.join(f'{n}={b / GB:.3f}' for n, b in holders)
        return total, detail
    for holder in (pool, getattr(pool, 'unified_buffer', None)):
        total = getattr(holder, 'total_bytes', None)
        if isinstance(total, int) and total > 0:
            return total, 'total_bytes'
    return None, None


def _install_one(cls):
    original = cls.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        try:
            total, source = _bytes_of(self)
            if total is None:
                logger.warning(
                    'DSV41 kv mem_usage: %s carries no byte count (inner pools: '
                    'unified=%s swa=%s window=%s); metric stays 0', cls.__name__,
                    type(getattr(self, 'unified_kv_pool', None)).__name__,
                    type(getattr(self, 'swa_kv_pool', None)).__name__,
                    type(getattr(self, 'request_window', None)).__name__)
                return
            self.mem_usage = total / GB
            logger.warning('DSV41 kv mem_usage: %s -> %.3f GB (via %s)',
                           cls.__name__, self.mem_usage, source)
        except Exception as exc:  # bookkeeping must never break a constructor
            logger.warning('DSV41 kv mem_usage: %s failed: %s', cls.__name__, exc)

    cls.__init__ = __init__


def _install_get_kvcache(cls):
    """Patch the allocator's `get_kvcache`, which every metric reader goes through.

    The `__init__` hook above only fires if the pool class we patched is the one
    the DSV4 path constructs; the measured engine built a pool whose `mem_usage`
    stayed 0 while that hook never logged, so the class we guessed is not the one
    in play. This hook cannot miss: the metric, the load inquirer and the
    server-info payload all read `allocator.get_kvcache().mem_usage`.

    It returns the *real* pool, unmodified in every other respect, and only fills
    `mem_usage` when it is still falsy -- so identity checks and every other
    attribute keep working.
    """
    original = cls.get_kvcache

    def get_kvcache(self):
        real = original(self)
        try:
            if not getattr(real, 'mem_usage', None):
                total, source = _bytes_of(self)   # the allocator holds the buffer
                if total is None:
                    logger.warning(
                        'DSV41 kv mem_usage: allocator %s -> kvcache %s carries no '
                        'byte count; metric stays 0', cls.__name__, type(real).__name__)
                else:
                    real.mem_usage = total / GB
                    logger.warning(
                        'DSV41 kv mem_usage: %s.get_kvcache() -> %s, %.3f GB (via %s)',
                        cls.__name__, type(real).__name__, real.mem_usage, source)
        except Exception as exc:  # a metric read must never break serving
            logger.warning('DSV41 kv mem_usage: get_kvcache hook failed: %s', exc)
        return real

    cls.get_kvcache = get_kvcache


def install_scheduler_diag(module):
    """Log what the KV-cache metric actually reads, from the one call site.

    `Scheduler.emit_metrics_constants` runs once at startup and is the only
    reader of `allocator.get_kvcache().mem_usage` that feeds the gauge. If the
    hooks above never log, this says why: it prints the concrete allocator and
    pool types and the mem_usage it sees, immediately before the original runs.
    Gated on DSV41_KV_METRIC_DIAG so it stays inert in normal operation.
    """
    if os.environ.get('DSV41_KV_METRIC_DIAG', '0') != '1':
        return
    cls = module.Scheduler
    original = cls.emit_metrics_constants

    def emit_metrics_constants(self):
        try:
            alloc = self.token_to_kv_pool_allocator
            kc = alloc.get_kvcache()
            logger.warning(
                'DSV41 kv diag: allocator=%s kvcache=%s mem_usage=%r',
                type(alloc).__name__, type(kc).__name__,
                getattr(kc, 'mem_usage', '<missing>'))
            # Walk every attribute that looks like a buffer or a sub-pool and
            # report the bytes each one holds, so the real total is visible
            # instead of guessed at.
            found = []
            for name, value in vars(kc).items():
                if value is None:
                    continue
                b = _sum_buffer_bytes(value)
                if b == 0 and hasattr(value, 'get_buf_infos'):
                    try:
                        b = sum(value.get_buf_infos()[1])
                    except Exception:
                        b = 0
                if b > 0:
                    found.append(f'{name}:{type(value).__name__}={b / GB:.3f}GB')
            logger.warning('DSV41 kv diag holders: %s', ' | '.join(found) or '(none)')
        except Exception as exc:
            logger.warning('DSV41 kv diag failed: %s', exc)
        return original(self)

    cls.emit_metrics_constants = emit_metrics_constants
    logger.warning('DSV41 kv diag installed on Scheduler.emit_metrics_constants')


def install(module):
    installed = []
    for name in _TARGETS:
        cls = getattr(module, name, None)
        if cls is None:
            continue
        if name.endswith('Allocator'):
            _install_get_kvcache(cls)
        else:
            _install_one(cls)
        installed.append(name)
    logger.warning('DSV41 kv mem_usage adapter installed for: %s', ', '.join(installed))
