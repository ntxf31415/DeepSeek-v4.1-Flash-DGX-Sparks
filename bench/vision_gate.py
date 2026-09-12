#!/usr/bin/env python3
# dsv41 vision 双方向门禁: 红左蓝右 + 蓝左红右, 1024 image token 口径
import argparse, base64, io, json, time, urllib.request, zlib, struct

ap = argparse.ArgumentParser()
ap.add_argument('--base', default='http://127.0.0.1:8899/v1')
ap.add_argument('--model', default='deepseek-v4.1-flash')
ap.add_argument('--key', default='YOUR_API_KEY')
args = ap.parse_args()


def png_chunk(t, d):
    c = t + d
    return struct.pack('>I', len(d)) + c + struct.pack('>I', zlib.crc32(c))


def make_image(left_color, right_color, w=1024, h=1024):
    raw = b''
    for y in range(h):
        raw += b'\x00'
        for x in range(w):
            raw += bytes(left_color if x < w // 2 else right_color)
    return (b'\x89PNG\r\n\x1a\n'
            + png_chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + png_chunk(b'IDAT', zlib.compress(raw))
            + png_chunk(b'IEND', b''))


def ask_image(img, question):
    b64 = base64.b64encode(img).decode()
    body = dict(model=args.model, temperature=0, max_tokens=80, stream=False,
                chat_template_kwargs={'thinking': False},
                messages=[{'role': 'user', 'content': [
                    {'type': 'text', 'text': question},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64}}]}])
    req = urllib.request.Request(args.base + '/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json',
                                          'Authorization': 'Bearer ' + args.key})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    wall = time.time() - t0
    u = r['usage']
    details = (u.get('prompt_tokens_details') or {})
    return r['choices'][0]['message']['content'] or '', wall, details.get('image_tokens')


red_left = make_image((255, 0, 0), (0, 0, 255))
blue_left = make_image((0, 0,255), (255, 0, 0))
for name, img, expect in [('red-left', red_left, ('red', 'left')),
                          ('blue-left', blue_left, ('blue', 'left'))]:
    q = ('State the color on the LEFT half and the color on the RIGHT half. '
         'Answer in the exact format: LEFT=<color> RIGHT=<color>.')
    content, wall, img_tok = ask_image(img, q)
    low = content.lower()
    ok = all(w in low for w in expect)
    print(f'vision {name}: img_tokens={img_tok} wall={wall:.1f}s {"PASS" if ok else "FAIL"} '
          f'answer={content[:60]!r}', flush=True)
