#!/usr/bin/env python3
"""matrix_v1: MiaAI matrix.py 的 /v1 等价实现
- uuid nonce + filler + 固定 instruction 构造深前缀, 实际 token 数以 usage 为准
- /v1/chat/completions stream=True, 每 SSE 块记 (seconds, 累计token数)
- 输出 {size}-c{concurrency}.json, 可直接喂 common_window.py
"""
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from tokenizers import Tokenizer

model = Path(os.environ.get('MODEL_PATH', '/models/DeepSeek-V4.1-Flash'))
state = Path(os.environ.get('STATE_PATH', '/state'))
tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
manifest = Path(os.environ.get('MANIFEST_PATH', str(state / 'launch.json')))
identity = hashlib.sha256(manifest.read_bytes()).hexdigest()
sizes = [int(x) for x in os.environ.get('PREFILL_SIZES',
        '512,2048,8192,32768,65536,131072').split(',')]
concurrencies = [int(x) for x in os.environ.get('CONCURRENCIES', '1,2,4').split(',')]
folder = state / ('matrix-' + time.strftime('%Y%m%dT%H%M%S'))
folder.mkdir(parents=True)
key = (state / 'api-key').read_text().strip()

FILLER = ('Reference notes: the cache stores recently accessed entries. An implementation '
          'should maintain ordering, handle replacement and validate its invariants.\n')
INSTRUCTION = ('\nNow write a complete Python LRU cache module with a doubly linked list and '
               'dictionary, including get, put, delete, iteration, resize, clear, invariant '
               'validation, detailed docstrings and ten usage examples. Return code only.')


def encode(text):
    return tokenizer.encode(text, add_special_tokens=False).ids


def build_text(size):
    nonce = uuid.uuid4().hex
    prefix = nonce + '\nRead these notes.\n'
    unit = len(encode(FILLER))
    room = size - len(encode(prefix)) - len(encode(INSTRUCTION))
    assert room >= 0
    reps = (room + unit - 1) // unit
    text = prefix + FILLER * reps + INSTRUCTION
    return text, nonce


def one(args):
    size, index, wave_start = args
    text, nonce = build_text(size)
    result = dict(index=index, target_tokens=size, nonce=nonce,
                  manifest_sha256=identity, started_s=time.monotonic() - wave_start,
                  events=[], output_budget=1024, output_ids=[], finish_reason=None,
                  prompt_tokens=None)
    body = dict(model=os.environ.get('SERVED_NAME', 'deepseek-v4.1-flash'), temperature=0,
                max_tokens=1024, stream=True, chat_template_kwargs={'thinking': False},
                messages=[{'role': 'user', 'content': text}])
    req = urllib.request.Request('http://127.0.0.1:8899/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json',
                                          'Authorization': 'Bearer ' + key})
    try:
        with urllib.request.urlopen(req, timeout=2400) as resp:
            acc = ''
            for line in resp:
                line = line.decode().strip()
                if not line.startswith('data:'):
                    continue
                payload = line[5:].strip()
                if payload == '[DONE]':
                    break
                ev = json.loads(payload)
                choices = ev.get('choices') or []
                if choices and choices[0].get('delta', {}).get('content'):
                    acc += choices[0]['delta']['content']
                if choices and choices[0].get('finish_reason'):
                    result['finish_reason'] = choices[0]['finish_reason']
                if ev.get('usage'):
                    result['prompt_tokens'] = ev['usage'].get('prompt_tokens')
                result['events'].append(dict(seconds=time.monotonic() - wave_start,
                                             count=len(encode(acc))))
        result['output_ids'] = encode(acc)
    except Exception as e:
        result['error'] = repr(e)
    return result


for size in sizes:
    for c in concurrencies:
        wave_start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(c) as pool:
            records = list(pool.map(one, [(size, i, wave_start) for i in range(c)]))
        out = folder / f'{size}-c{c}.json'
        out.write_text(json.dumps(records, indent=1))
        errs = sum(1 for r in records if r.get('error'))
        print(f'{size}-c{c}: done errs={errs}', flush=True)

print('FOLDER=' + str(folder), flush=True)
