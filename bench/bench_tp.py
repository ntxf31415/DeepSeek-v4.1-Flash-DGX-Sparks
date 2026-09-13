#!/usr/bin/env python3
"""bench_tp.py — 按「第三方报告形状」测并发：TTFT + 聚合 + 单流。

第三方（MIA+PR3）报告字段：并发数 / TTFT / Streams / Aggregate / 单 Stream。
本站口径差异点（需一致才能对表）：
  - 输出长度（本脚本 --max-tokens，默认 400 与 bench_full_uuid 一致）
  - prompt 类型（--prompt-type code|count|mixed）
  - 释放方式（同时释放，非错峰）
  - 流式（TTFT 需流式才能测首 token）

用法：
  bench_tp.py --base http://127.0.0.1:8899/v1 --model deepseek-v4.1-flash \
              --key K --conc 1,2,4,8 --max-tokens 400 --prompt-type code
"""
import argparse, json, statistics, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--base', default='http://127.0.0.1:8899/v1')
ap.add_argument('--model', default='deepseek-v4.1-flash')
ap.add_argument('--key', default='')
ap.add_argument('--conc', default='1,2,4,8')
ap.add_argument('--max-tokens', type=int, default=400)
ap.add_argument('--prompt-type', default='code', choices=['code', 'count', 'mixed'])
ap.add_argument('--thinking', action='store_true')
args = ap.parse_args()

PROMPTS = {
    'code': 'Write a complete Python module implementing an LRU cache with a doubly '
            'linked list and dictionary. Include get, put, delete, iteration, resize, '
            'clear, invariant validation and detailed docstrings. Return code only.',
    'count': 'Count from 1 to 400, one number per line, no explanations.',
    'mixed': 'Explain in depth how a B-tree maintains balance under insertion and '
             'deletion, with the splitting and merging rules and a worked example.',
}


def one(idx, out):
    body = json.dumps({
        'model': args.model, 'temperature': 0, 'max_tokens': args.max_tokens,
        'stream': True, 'stream_options': {'include_usage': True},
        'chat_template_kwargs': {'thinking': args.thinking},
        'messages': [{'role': 'user', 'content': PROMPTS[args.prompt_type]}],
    }).encode()
    req = urllib.request.Request(args.base + '/chat/completions', data=body, headers={
        'Content-Type': 'application/json',
        **({'Authorization': 'Bearer ' + args.key} if args.key else {})})
    t0 = time.time()
    ttft = None
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith('data:'):
                    continue
                p = line[5:].strip()
                if p == '[DONE]':
                    break
                ev = json.loads(p)
                ch = ev.get('choices') or []
                if ttft is None and ch and (ch[0].get('delta') or {}).get('content'):
                    ttft = time.time() - t0
                if ev.get('usage'):
                    usage = ev['usage']
    except Exception as e:
        out[idx] = {'err': repr(e)}
        return
    out[idx] = {'ttft': ttft, 'wall': time.time() - t0,
                'ct': (usage or {}).get('completion_tokens')}


print(f'=== bench_tp · prompt={args.prompt_type} · max_tokens={args.max_tokens} '
      f'· thinking={args.thinking} ===')
print(f'{"conc":>4} {"TTFT":>9} {"Streams":>9} {"Aggregate":>12} {"per-stream":>11}')
for c in [int(x) for x in args.conc.split(',')]:
    out = [None] * c
    ths = [threading.Thread(target=one, args=(i, out)) for i in range(c)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    ok = [o for o in out if o and not o.get('err')]
    ttfts = [o['ttft'] for o in ok if o.get('ttft')]
    toks = sum(o['ct'] for o in ok if o.get('ct'))
    agg = toks / wall if wall else 0
    per = agg / c if c else 0
    ttft_ms = round(statistics.median(ttfts) * 1000) if ttfts else -1
    print(f'{c:>4} {ttft_ms:>7}ms {len(ok)}/{c:<7} '
          f'{agg:>10.1f} t/s {per:>9.1f} t/s')
