#!/usr/bin/env python3
# prose 单流测速: 散文生成, TTFT 分离, 3 轮中位
import os, json, time, urllib.request, statistics, sys
URL = os.environ.get('URL', 'http://127.0.0.1:8899/v1')
MODEL = os.environ.get('MODEL', 'deepseek-v4.1-flash')
KEY = os.environ.get('API_KEY', '')
THINK = os.environ.get('THINK', '0') == '1'
PROMPT = ('Write a vivid prose essay in English about an autumn evening in a small '
          'harbour town: the fading light on the water, the sounds of the quay, the '
          'smell of salt and woodsmoke, a few boats and people. Write about 500 words, '
          'no lists, no headings.')
def run():
    body = dict(model=MODEL, temperature=0, max_tokens=600, stream=False,
                chat_template_kwargs={'thinking': THINK},
                messages=[{'role': 'user', 'content': PROMPT}])
    req = urllib.request.Request(URL + '/chat/completions',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + KEY})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    wall = time.time() - t0
    m = r['choices'][0]['message']
    u = r['usage']
    ct = u['completion_tokens']
    return wall, ct, m.get('content') or '', m.get('reasoning_content') or None
vals = []
for i in range(3):
    wall, ct, content, rc = run()
    decode = (ct - 1) / wall if wall > 0 else 0
    print(f'round{i+1}: completion={ct} wall={wall:.1f}s decode={decode:.1f} t/s think_len={len(rc) if rc else 0}', flush=True)
    vals.append((decode, ct))
decode_med = statistics.median(v[0] for v in vals)
print(f'MEDIAN decode={decode_med:.1f} t/s (thinking={"ON" if THINK else "OFF"})', flush=True)
