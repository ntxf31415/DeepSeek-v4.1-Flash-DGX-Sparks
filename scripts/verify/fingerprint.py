#!/usr/bin/env python3
"""Greedy fingerprint: fixed prompts, temperature 0, one hash per prompt.

Exists because several DSV4 / SM12x changes are known to alter greedy output
rather than just its speed -- e.g. sglang#39235 (the SM120 decode path pads
query heads to 64, and removing the pad changes the tokens produced). An A/B
that shows a throughput win is not a win if it also moved the output, so every
arm of a comparison should run this first and diff the hashes.

Env: URL, MODEL, API_KEY, MAX_TOKENS (default 64).
Usage: API_KEY=... python3 fingerprint.py
"""
import hashlib
import json
import os
import sys
import urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:8899/v1/chat/completions")
MODEL = os.environ.get("MODEL", "deepseek-v4.1-flash")
API_KEY = os.environ.get("API_KEY", "")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))

PROMPTS = [
    ("arith", "What is 19 + 23? Reply only the number."),
    ("arith2", "Compute 144 / 12 * 7 - 3. Reply only the number."),
    ("code", "Write a Python function that returns the nth Fibonacci number "
             "iteratively. Code only."),
    ("json", 'Return a JSON object {"a": 1, "b": [2, 3]} and nothing else.'),
    ("prose", "Explain in three sentences why the sky is blue."),
    ("list", "List the first five prime numbers, comma separated, nothing else."),
    ("reason", "A bat and a ball cost $1.10 together. The bat costs $1 more "
               "than the ball. How much is the ball? Reply with the amount only."),
    ("zh", "用一句话说明什么是哈希表。"),
]


def ask(prompt):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "top_p": 1,
        "chat_template_kwargs": {"thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"], d["usage"]["completion_tokens"]


def main():
    digests = []
    for name, prompt in PROMPTS:
        try:
            text, ntok = ask(prompt)
        except Exception as e:
            print(f"{name:8s} ERROR {e}")
            digests.append(f"{name}:ERROR")
            continue
        h = hashlib.sha256(text.encode()).hexdigest()[:12]
        digests.append(f"{name}:{h}")
        flat = text.replace("\n", " ")[:78]
        print(f"{name:8s} {h} {ntok:4d}tok  {flat!r}", flush=True)
    total = hashlib.sha256("|".join(digests).encode()).hexdigest()[:16]
    print(f"FINGERPRINT {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
