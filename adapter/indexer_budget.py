"""Cap the SM12x torch top-k indexer's transient score buffer.

`_TORCH_INDEXER_SCORE_BUDGET_BYTES` in `deepseek_v4_backend` bounds how many
query rows the torch indexer scores per step: the prefill path sizes a step as
`budget // (heads * visible_keys * 2)` and the extend path as
`budget // (lc * 4)`, so the constant is a direct cap on the largest transient
allocation a long prefill makes. Upstream ships 1 GiB, which on a GB10 is
memory taken from the same unified pool the KV cache draws on.

Two independent measurements say that transient is what actually binds at long
context, not the KV pool: an SM120/SM121 indexer split reported 1M context
running at 95% VRAM with the KV pool at only 47%, and a sibling DSV41 SGLang
port budgets the same constant at 256 MiB instead of 1 GiB. This adapter is how
we test that without forking the backend.

Inert unless `DSV41_INDEXER_BUDGET_MIB` is set, so the patch can ship in the
image and the comparison stays a single env var.
"""
import logging
import os

logger = logging.getLogger(__name__)

_ATTR = '_TORCH_INDEXER_SCORE_BUDGET_BYTES'


def install(module):
    want = os.environ.get('DSV41_INDEXER_BUDGET_MIB', '').strip()
    if not want:
        return
    try:
        new = int(want) * (1 << 20)
    except ValueError:
        logger.warning('DSV41 indexer budget: DSV41_INDEXER_BUDGET_MIB=%r is not an '
                       'integer; leaving %s at its upstream value', want, _ATTR)
        return
    old = getattr(module, _ATTR, None)
    if not isinstance(old, int):
        logger.warning('DSV41 indexer budget: %s not found in %s; unchanged',
                       _ATTR, module.__name__)
        return
    if new <= 0:
        logger.warning('DSV41 indexer budget: refusing non-positive budget %d', new)
        return
    setattr(module, _ATTR, new)
    logger.warning('DSV41 indexer budget: %s %d -> %d bytes (%.0f MiB -> %.0f MiB)',
                   _ATTR, old, new, old / 2**20, new / 2**20)
