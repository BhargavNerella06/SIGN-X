"""End-to-end WebSocket benchmark against a running SIGN-X server (see docs/performance.md).

Sends real 640-px JPEG frames (from INCLUDE test videos, looped) at a fixed
capture rate, like the browser does, and measures for every frame the
round trip send -> prediction message. Also samples server CPU / RSS.

mode=flood     : current frontend behaviour, send on every tick regardless.
mode=inflight1 : at most one frame in flight; a tick is skipped while waiting.

Usage: python e2e_ws.py <port> <fps> <seconds> <mode> [server_pid]
"""
import asyncio
import glob
import json
import sys
import time

import cv2
import numpy as np
import websockets

try:
    import psutil  # optional: server CPU / memory figures
except ImportError:
    psutil = None

port, fps, seconds, mode = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
pid = int(sys.argv[5]) if len(sys.argv) > 5 else None

videos = sorted(glob.glob("datasets/_include_zips/extracted/Greetings/48. Hello/*.MOV"))[:3]
jpegs = []
for v in videos:
    cap = cv2.VideoCapture(v)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (640, 360), interpolation=cv2.INTER_AREA)
        jpegs.append(cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())


async def main():
    proc = psutil.Process(pid) if (pid and psutil) else None
    if proc:
        proc.cpu_percent(None)
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws/predict", max_size=None) as ws:
        json.loads(await ws.recv())  # connection
        sent_at = []
        rtts, server_lat, skipped = [], [], 0
        in_flight = 0
        done = asyncio.Event()

        async def reader():
            nonlocal in_flight
            while True:
                try:
                    msg = json.loads(await ws.recv())
                except Exception:
                    return
                if msg.get("type") in ("prediction", "error"):
                    rtts.append((time.perf_counter() - sent_at[len(rtts)]) * 1e3)
                    server_lat.append(msg.get("latency_ms", 0.0))
                    in_flight -= 1
                    if done.is_set() and in_flight == 0:
                        return

        rt = asyncio.create_task(reader())
        interval = 1.0 / fps
        t_end = time.perf_counter() + seconds
        nxt = time.perf_counter()
        i = 0
        while time.perf_counter() < t_end:
            if mode == "inflight1" and in_flight > 0:
                skipped += 1
            else:
                sent_at.append(time.perf_counter())
                in_flight += 1
                await ws.send(jpegs[i % len(jpegs)])
                i += 1
            nxt += interval
            await asyncio.sleep(max(0.0, nxt - time.perf_counter()))
        done.set()
        while in_flight > 0 and not rt.done():
            await asyncio.sleep(0.01)
        rt.cancel()
        r = np.array(rtts)
        out = {
            "mode": mode, "target_fps": fps, "frames_sent": len(sent_at), "ticks_skipped": skipped,
            "effective_fps": len(sent_at) / seconds,
            "rtt_ms_median": float(np.median(r)), "rtt_ms_p95": float(np.percentile(r, 95)), "rtt_ms_max": float(r.max()),
            "rtt_ms_last10_median": float(np.median(r[-10:])),
            "server_reported_latency_ms_median": float(np.median(server_lat)),
        }
        if proc:
            out["server_cpu_percent"] = proc.cpu_percent(None)
            out["server_rss_mb"] = proc.memory_info().rss / 2**20
        print(json.dumps(out))


asyncio.run(main())
