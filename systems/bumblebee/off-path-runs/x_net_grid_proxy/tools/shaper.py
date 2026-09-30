#!/usr/bin/env python3
"""Userspace link shaper for two co-located parties (replaces tbf+netem on lo).

Model: each sent chunk held for the one-way delay D, then serialised through one token bucket of
rate R shared by both directions and all connections. Per-direction queue bound R x 0.6 s (tbf
latency 600 ms), minimum 4 MiB; a full queue back-pressures the sender through TCP.

    shaper.py --rate-mbit R --delay-ms D --map LISTEN:TARGET [--map ...] --stats FILE
"""
import argparse, asyncio, socket, time, json, signal

ap = argparse.ArgumentParser()
ap.add_argument("--rate-mbit", type=float, required=True)
ap.add_argument("--delay-ms", type=float, required=True)   # one-way
ap.add_argument("--map", action="append", required=True)
ap.add_argument("--burst-bytes", type=int, default=128 * 1024)
ap.add_argument("--stats", required=True)
a = ap.parse_args()

RATE = a.rate_mbit * 1e6 / 8.0          # bytes/s
DELAY = a.delay_ms / 1000.0
QMAX = max(4 << 20, int(RATE * 0.6))
stats = {"bytes": 0, "chunks": 0, "conns": 0, "rate_Bps": RATE, "delay_s": DELAY, "qmax": QMAX}


class Bucket:
    def __init__(self, rate, burst):
        self.rate, self.burst = rate, burst
        self.tokens, self.t = burst, time.monotonic()
        self.lock = asyncio.Lock()

    async def take(self, n):
        async with self.lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= n or (n > self.burst and self.tokens >= self.burst):
                    self.tokens -= n
                    return
                await asyncio.sleep((min(n, self.burst) - self.tokens) / self.rate)


BUCKET = None


async def pump(reader, writer):
    q = asyncio.Queue()
    queued = [0]
    room = asyncio.Event(); room.set()

    async def rx():
        while True:
            await room.wait()
            d = await reader.read(65536)
            if not d:
                await q.put(None); return
            queued[0] += len(d)
            if queued[0] >= QMAX: room.clear()
            await q.put((time.monotonic() + DELAY, d))

    async def tx():
        while True:
            item = await q.get()
            if item is None:
                try:
                    writer.write_eof()
                except Exception:
                    pass
                return
            due, d = item
            w = due - time.monotonic()
            if w > 0: await asyncio.sleep(w)
            await BUCKET.take(len(d))
            writer.write(d)
            await writer.drain()
            queued[0] -= len(d)
            if queued[0] < QMAX: room.set()
            stats["bytes"] += len(d); stats["chunks"] += 1

    await asyncio.gather(rx(), tx(), return_exceptions=True)


def nodelay(w):
    s = w.get_extra_info("socket")
    if s is not None:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


async def main():
    global BUCKET
    BUCKET = Bucket(RATE, a.burst_bytes)
    servers = []
    for m in a.map:
        lp, tp = (int(x) for x in m.split(":"))

        async def handle(r, w, tp=tp):
            stats["conns"] += 1
            for attempt in range(600):
                try:
                    tr, tw = await asyncio.open_connection("127.0.0.1", tp); break
                except OSError:
                    await asyncio.sleep(0.1)
            else:
                w.close(); return
            nodelay(w); nodelay(tw)
            await asyncio.gather(pump(r, tw), pump(tr, w), return_exceptions=True)
            for x in (w, tw):
                try: x.close()
                except Exception: pass

        servers.append(await asyncio.start_server(handle, "127.0.0.1", lp, reuse_address=True))
    print("SHAPER_READY", flush=True)

    async def dump():
        while True:
            with open(a.stats, "w") as f: json.dump(stats, f)
            await asyncio.sleep(5)
    asyncio.ensure_future(dump())
    stop = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
    await stop.wait()
    with open(a.stats, "w") as f: json.dump(stats, f)

asyncio.run(main())
