#!/usr/bin/env python3
# dsv41 质量门禁套件: needle 梯度 / Hangul corruption / 终止性 / code-gate
import argparse, json, re, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--base', default='http://127.0.0.1:8899/v1')
ap.add_argument('--model', default='deepseek-v4.1-flash')
ap.add_argument('--key', default='YOUR_API_KEY')
ap.add_argument('--needle', action='store_true')
ap.add_argument('--corruption', action='store_true')
ap.add_argument('--termination', action='store_true')
ap.add_argument('--code', action='store_true')
args = ap.parse_args()


def ask(messages, temperature=0, max_tokens=64, thinking=False):
    body = dict(model=args.model, temperature=temperature, max_tokens=max_tokens,
                stream=False, chat_template_kwargs={'thinking': thinking},
                messages=messages)
    req = urllib.request.Request(args.base + '/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json',
                                          'Authorization': 'Bearer ' + args.key})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=2400))
    wall = time.time() - t0
    c = r['choices'][0]
    u = r['usage']
    return dict(content=c['message']['content'] or '', wall=wall,
                prompt_tokens=u['prompt_tokens'],
                completion_tokens=u['completion_tokens'],
                finish=c['finish_reason'])


FILL = ('Reference log entry: the harbour keeper noted the tide tables, lantern fuel '
        'and mooring lines before the evening bell rang across the quay. ')


def haystack(tokens, needle, depth=0.6):
    # token-精确构造: 单位 token 数一次预计算, 增量累加 (O(n), 避免重编码)
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file('/models/DeepSeek-V4.1-Flash/tokenizer.json')
        unit_tok = len(tok.encode(FILL, add_special_tokens=False).ids)
    except Exception:
        tok, unit_tok = None, len(FILL) // 7
    reps = (tokens + unit_tok - 1) // unit_tok
    p = FILL * reps
    pos = int(len(p) * depth)
    p = p[:pos] + needle + p[pos + len(needle):]
    return p


if args.needle:
    for depth, seed in [(30000, 21), (229000, 22), (470000, 23)]:
        needle = f'The secret code word is PELICAN-{seed}.'
        prompt = haystack(depth, needle) + '\nWhat is the secret code word? Answer with the code word only.'
        r = ask([{'role': 'user', 'content': prompt}], temperature=0, max_tokens=30)
        ok = f'PELICAN-{seed}' in r['content']
        print(f'needle {depth}: prefill={r["prompt_tokens"]} wall={r["wall"]:.1f}s '
              f'({r["prompt_tokens"] / r["wall"]:.0f} t/s) answer={r["content"][:40]!r} '
              f'{"PASS" if ok else "FAIL"}', flush=True)

if args.corruption:
    for i in range(3):
        r = ask([{'role': 'user', 'content':
                  'Write the Korean word for "hello" and then a short greeting sentence in Korean.'}],
                temperature=0, max_tokens=100)
        bad = r['content'].count('�')
        print(f'corruption pass{i+1}: U+FFFD={bad} {"PASS" if bad == 0 else "FAIL"} '
              f'sample={r["content"][:60]!r}', flush=True)

if args.termination:
    for temp in (1.0, 0.6):
        stops = 0
        for i in range(18):
            r = ask([{'role': 'user', 'content': 'Write a short haiku about the sea.'}],
                    temperature=temp, max_tokens=200)
            if r['finish'] == 'stop':
                stops += 1
        print(f'termination temp={temp}: {stops}/18 stop', flush=True)

if args.code:
    tasks = [
        'Write a Python function to compute the greatest common divisor of two integers.',
        'Write a Python class implementing a stack with push, pop and peek.',
        'Write a Python function that checks if a string is a palindrome.',
        'Write a Python function to merge two sorted lists.',
        'Write a Python function to find the longest word in a sentence.',
        'Write a Python class for a circular queue.',
        'Write a Python function to count words in a text file given as a string.',
        'Write a Python function to reverse a linked list given as a list of values.',
        'Write a Python function to compute the Fibonacci number n iteratively.',
        'Write a Python function to find all prime numbers below n.',
        'Write a Python function to convert a Roman numeral to an integer.',
        'Write a Python function to compute the Levenshtein distance of two strings.',
    ]
    passed = 0
    for i, t in enumerate(tasks):
        r = ask([{'role': 'user', 'content': t + ' Return code only.'}],
                temperature=0, max_tokens=400)
        body = r['content']
        ok = bool(re.search(r'def |class |```', body)) and not re.search(r'[　-鿿]{20,}', body)
        passed += ok
        print(f'code-gate {i+1}/12: {"PASS" if ok else "FAIL"} tok={r["completion_tokens"]} '
              f'sample={body[:50]!r}', flush=True)
    print(f'code-gate total: {passed}/12', flush=True)
