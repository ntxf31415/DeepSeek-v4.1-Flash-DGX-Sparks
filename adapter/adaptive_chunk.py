"""Size each prefill chunk from the prefix length, instead of one fixed chunk.

The stack pays two coupled costs on a long prefill. The indexer's per-chunk
transient is ~14 B x chunk x prefix, so a big chunk blows memory at long prefix
(measured: chunk 2048 at a 900K prefix drives MemAvailable to 0.34 GB). But a
small chunk multiplies the step count, and each step costs a ring collective
that does not shrink with the chunk (measured: chunk 1024 makes a 900K prefill
3x slower). One fixed size cannot serve both ends.

SGLang already has the hook: `Scheduler.get_new_batch_prefill` reads

    chunked_prefill_size = self.chunked_prefill_size
    if self.chunked_req is not None and self.dynamic_chunk_sizer is not None:
        dynamic_size = self.dynamic_chunk_sizer.predict(len(prefix_indices))
        if dynamic_size is not None:
            chunked_prefill_size = dynamic_size

`maybe_init_dynamic_chunk_sizer` only installs a sizer under
`--enable-dynamic-chunking` with `pp_size > 1` (its DynamicChunkSizer targets
pipeline stages), so on a TP-only fleet the hook is dead. This adapter supplies
a sizer for the prefix policy instead: keep the static chunk while the prefix is
short, shrink toward a floor as the prefix grows.

Policy (upstream docs/chunked-prefill-memory.md 2.1): short prefixes keep the
full chunk for speed; past the high-water mark use the floor to bound the
transient. In between, interpolate on a fixed grid so kernel tile choices stay
stable.

Known risk, to validate with a needle: `schedule_batch.py` accounts SWA eviction
against the *static* `chunked_prefill_size` (the overlap path, extend_batch_idx
>= 2). A per-batch chunk that differs from it can skew that accounting, which
would show up as a wrong answer or a crash rather than silently. Hence the
needle is the gate, and DSV41_ADAPTIVE_CHUNK=0 disables the whole thing.

Env:
  DSV41_ADAPTIVE_CHUNK=1        enable (default off)
  DSV41_ADAPTIVE_LOW=98304      prefix at/below which the static chunk is kept
  DSV41_ADAPTIVE_HIGH=199680    prefix at/above which the floor chunk is used
  DSV41_ADAPTIVE_FLOOR=512      chunk used at/above HIGH
  DSV41_ADAPTIVE_GRID=128       round interpolated sizes down to this multiple
"""
import logging
import os

logger = logging.getLogger(__name__)

_ENV_ON = 'DSV41_ADAPTIVE_CHUNK'
_LOW = 'DSV41_ADAPTIVE_LOW'
_HIGH = 'DSV41_ADAPTIVE_HIGH'
_FLOOR = 'DSV41_ADAPTIVE_FLOOR'
_GRID = 'DSV41_ADAPTIVE_GRID'


class PrefixChunkSizer:
    """predict(history_len) -> chunk size, or None to keep the static one."""

    def __init__(self, base_chunk, low, high, floor, grid):
        self.base = base_chunk
        self.low = low
        self.high = high
        self.floor = floor
        self.grid = grid

    def predict(self, history_len):
        if history_len <= self.low:
            return None                      # short prefix: static chunk
        if history_len >= self.high:
            return self.floor
        # Linear between (low -> base) and (high -> floor), floored to the grid.
        span = self.high - self.low
        frac = (history_len - self.low) / span
        size = self.base - frac * (self.base - self.floor)
        size = int(size // self.grid) * self.grid
        return max(size, self.floor)

    def __repr__(self):
        return (f'PrefixChunkSizer(base={self.base} low={self.low} high={self.high} '
                f'floor={self.floor} grid={self.grid})')


def install(module):
    if os.environ.get(_ENV_ON, '0') != '1':
        return
    cls = module.Scheduler
    original = cls.maybe_init_dynamic_chunk_sizer

    def maybe_init_dynamic_chunk_sizer(self):
        original(self)                     # upstream sets it to None for pp_size == 1
        try:
            base = self.chunked_prefill_size
            if not base or base <= 0:
                logger.warning('DSV41 adaptive chunk: no static chunked_prefill_size; skipping')
                return
            sizer = PrefixChunkSizer(
                base_chunk=base,
                low=int(os.environ.get(_LOW, '98304')),
                high=int(os.environ.get(_HIGH, '199680')),
                floor=int(os.environ.get(_FLOOR, '512')),
                grid=int(os.environ.get(_GRID, '128')),
            )
            self.dynamic_chunk_sizer = sizer
            logger.warning('DSV41 adaptive chunk enabled: %s', sizer)
        except Exception as exc:           # never break scheduler init
            logger.warning('DSV41 adaptive chunk failed to install: %s', exc)

    cls.maybe_init_dynamic_chunk_sizer = maybe_init_dynamic_chunk_sizer
    logger.warning('DSV41 adaptive chunk adapter installed')
