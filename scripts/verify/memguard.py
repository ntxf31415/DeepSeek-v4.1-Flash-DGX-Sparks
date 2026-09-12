#!/usr/bin/env python3
"""Memory guard: sample the head's MemAvailable every second over ssh-less HTTP? No: run ON the head.
Usage (on the head): python3 memguard.py <logfile> [threshold_gb]
Writes 'ts memavail_gb' lines; on breach POSTs /abort_request to the server and writes ABORT line."""
import sys, time, json, urllib.request
log = open(sys.argv[1], 'a', buffering=1)
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
minv = 1e9
while True:
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable'):
                gb = int(line.split()[1]) / 1024 / 1024
                break
    minv = min(minv, gb)
    log.write(f'{time.strftime("%H:%M:%S")} {gb:.2f} min={minv:.2f}\n')
    if gb < thr:
        log.write(f'{time.strftime("%H:%M:%S")} ABORT: MemAvailable {gb:.2f} GB < {thr} GB -> POST /abort_request\n')
        try:
            req = urllib.request.Request('http://127.0.0.1:8899/abort_request', data=json.dumps({'abort_all': True}).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer YOUR_API_KEY'})
            urllib.request.urlopen(req, timeout=5).read()
            log.write('abort_request sent\n')
        except Exception as e:
            log.write(f'abort_request failed: {e}\n')
    time.sleep(1)
