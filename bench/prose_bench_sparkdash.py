#!/usr/bin/env python3
"""Prose decode, measured the way sparkDash measures it.

Matches `MiaAI-Lab/sparkDash` server/collectors/DecodeBench.js:
  - prompt    = DECODE_PROSE_PROMPT (src/shared/llmPrompts.js), not a local essay
  - request   = stream:true, stream_options.include_usage, temperature 0,
                top_p 1, thinking off, min_tokens == max_tokens, ignore_eos
  - metric    = (completion_tokens - 1) / (t_last - t_first content chunk)
                i.e. post-TTFT decode throughput. TTFT and prefill are excluded
                from the denominator -- that is the whole point of the protocol.
  - warmup    = one short stream first, so DFlash2/Triton JIT is not billed

One stream. The upstream table's "1 stream" row (45.4 tok/s on TP4) is this.

Env: URL, MODEL, API_KEY, MAX_TOKENS (default 256, their README's figure), ROUNDS.
"""
import json
import os
import statistics
import sys
import time
import urllib.request

URL = os.environ.get('URL', 'http://127.0.0.1:8899/v1')
MODEL = os.environ.get('MODEL', 'deepseek-v4.1-flash')
KEY = os.environ.get('API_KEY', '')
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', '256'))
ROUNDS = int(os.environ.get('ROUNDS', '5'))

# DECODE_PROSE_PROMPT, verbatim.
PROMPT = ('Write a detailed step-by-step explanation of how a hash map works, '
          'including collision handling, resizing, and time complexity. Be thorough.')


def stream_once(max_tokens):
    """One streaming request. Returns (decode_tps, completion_tokens, ttft_ms, span_ms)."""
    body = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': PROMPT}],
        'max_tokens': max_tokens,
        'min_tokens': max_tokens,
        'ignore_eos': True,
        'stop': [],
        'temperature': 0,
        'top_p': 1,
        'stream': True,
        'stream_options': {'include_usage': True},
        'chat_template_kwargs': {'thinking': False},
    }
    req = urllib.request.Request(
        URL + '/chat/completions',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json',
                 'Authorization': 'Bearer ' + KEY})

    t0 = time.perf_counter()
    t_first = t_last = None
    completion_tokens = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'):
                continue
            payload = line[5:].strip()
            if payload == '[DONE]':
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            now = time.perf_counter()
            for choice in chunk.get('choices') or []:
                delta = choice.get('delta') or {}
                # Content chunks only: a reasoning/role-only delta must not open
                # the window, matching the upstream content-chunk basis.
                if delta.get('content'):
                    if t_first is None:
                        t_first = now
                    t_last = now
            usage = chunk.get('usage')
            if usage and usage.get('completion_tokens') is not None:
                completion_tokens = usage['completion_tokens']

    if t_first is None or t_last is None or completion_tokens is None:
        raise RuntimeError('no content chunks or no usage in stream')
    span = t_last - t_first
    decode = (completion_tokens - 1) / span if span > 0 else 0.0
    return decode, completion_tokens, (t_first - t0) * 1000, span * 1000


def main():
    print(f'protocol: sparkDash DecodeBench | prompt=DECODE_PROSE_PROMPT '
          f'max_tokens={MAX_TOKENS} min_tokens={MAX_TOKENS} ignore_eos | '
          f'model={MODEL}', flush=True)
    # Warmup: 32 tokens, discarded (upstream WARMUP_MAX_TOKENS).
    try:
        stream_once(32)
        print('warmup: ok (32 tokens, discarded)', flush=True)
    except Exception as exc:
        print(f'warmup failed (continuing): {exc}', flush=True)

    vals = []
    for i in range(ROUNDS):
        decode, ct, ttft, span = stream_once(MAX_TOKENS)
        print(f'round{i+1}: decode={decode:.1f} t/s  completion={ct}  '
              f'ttft={ttft:.0f}ms  span={span:.0f}ms', flush=True)
        vals.append(decode)
    print(f'MEDIAN decode={statistics.median(vals):.1f} t/s  '
          f'min={min(vals):.1f} max={max(vals):.1f}  (n={len(vals)})', flush=True)


if __name__ == '__main__':
    sys.exit(main())
