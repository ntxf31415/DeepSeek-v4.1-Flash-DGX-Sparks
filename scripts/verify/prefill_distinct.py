#!/usr/bin/env python3
"""Needle prompt of ~<target> tokens with content salted by <seed> (no radix-cache prefix hits), max_tokens 20."""
import json, urllib.request, time, sys, random
URL = "http://127.0.0.1:8899/v1/chat/completions"
target = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
rnd = random.Random(seed)
words = ["harbour","lantern","meadow","copper","violin","orchard","compass","thistle","ember","granite","willow","saffron","anchor","quartz","ledger","falcon"]
def para(i):
    return f"Entry {seed}-{i}: the {rnd.choice(words)} keeper noted {rnd.randint(1,999)} {rnd.choice(words)} crates by the {rnd.choice(words)} gate before the {rnd.choice(words)} bell rang at dusk. "
secret = f"The secret code word is PELICAN-{seed}."
def ask(p, n=20):
    body = dict(model="deepseek-v4.1-flash", temperature=0, max_tokens=n, stream=False, chat_template_kwargs={"thinking": False}, messages=[{"role": "user", "content": p}])
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer YOUR_API_KEY"}), timeout=3000))
    return r["usage"]["prompt_tokens"], time.time() - t0, r["choices"][0]["message"]["content"].strip()
sample = "".join(para(i) for i in range(50)); n50, _, _ = ask(sample, 1); per = (n50 - 20) / 50.0
reps = int((target - 200) / per)
p = "".join(para(i) for i in range(reps)) + "\n" + secret + "\n" + "".join(para(reps + i) for i in range(20)) + "\nWhat is the secret code word? Answer with the code word only."
n, dt, txt = ask(p)
print(f"prompt_tokens={n} prefill+20tok={dt:.1f}s ({n/dt:.0f} tok/s) answer={txt[:40]!r} expected=PELICAN-{seed}", flush=True)
