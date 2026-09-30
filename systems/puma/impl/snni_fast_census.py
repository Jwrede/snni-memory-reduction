"""snni_fast_census: sucht je neuem RSS-Maximum NUR bytes-artige Objekte ab einer Mindestgroesse
(mit Halter), damit der Walk billig genug ist, um den Peak einzuholen.
"""
import os

# Reagiert auf beide Variablen, damit die Datei den bestehenden Census ohne Runner-Aenderung ersetzt.
if os.environ.get("SNNI_PUMA_FASTCENSUS") == "1" or os.environ.get("SNNI_PUMA_CENSUS") == "1":
    import gc
    import sys
    import threading
    import time

    _OUT = os.environ.get("SNNI_PUMA_CENSUS_DIR") or os.environ.get("RESULTS_DIR") or "/tmp"
    _MIN = int(os.environ.get("SNNI_FASTCENSUS_MIN", str(64 * 1024 * 1024)))

    def _rss():
        try:
            for l in open("/proc/self/status"):
                if l.startswith("VmRSS:"):
                    return int(l.split()[1])
        except OSError:
            pass
        return -1

    def _scan(out, rss):
        found = []
        seen = set()
        # Laufende Stapel zuerst: `fr.f_locals` legt einen Snapshot-dict an, der sonst als Halter
        # gemeldet wird statt des benannten Frame-Lokals (gemessen: ein 200-MB-Lokal kam als dict).
        try:
            me = threading.get_ident()
            for tid, fr in sys._current_frames().items():
                if tid == me:
                    continue
                while fr is not None:
                    try:
                        for k, v in fr.f_locals.items():
                            t = type(v)
                            if (t is bytes or t is bytearray) and len(v) >= _MIN and id(v) not in seen:
                                seen.add(id(v))
                                c = fr.f_code
                                found.append((len(v), "stack %s:%s:%d %s" % (
                                    os.path.basename(c.co_filename), c.co_name,
                                    c.co_firstlineno, k)))
                    except Exception:
                        pass
                    fr = fr.f_back
        except Exception:
            pass
        try:
            frontier = gc.get_objects()
            depth = 0
            while frontier and depth < 8:
                nxt = []
                for obj in frontier:
                    if id(obj) in seen:
                        continue
                    seen.add(id(obj))
                    try:
                        for ref in gc.get_referents(obj):
                            if id(ref) in seen:
                                continue
                            t = type(ref)
                            if t is bytes or t is bytearray:
                                seen.add(id(ref))
                                n = len(ref)
                                if n >= _MIN:
                                    h = type(obj).__name__
                                    if h == "frame":
                                        try:
                                            c = obj.f_code
                                            h = "frame %s:%s:%d" % (
                                                os.path.basename(c.co_filename), c.co_name,
                                                c.co_firstlineno)
                                            for k, v in obj.f_locals.items():
                                                if v is ref:
                                                    h += " " + k
                                                    break
                                        except Exception:
                                            pass
                                    found.append((n, h))
                            else:
                                nxt.append(ref)
                    except Exception:
                        continue
                frontier = nxt
                depth += 1
        except Exception as e:
            print("FAST|walk_error|%r" % (e,), file=out, flush=True)
        tot = sum(n for n, _ in found)
        print("FAST|newmax|rss_kb=%d|n=%d|total=%d" % (rss, len(found), tot), file=out, flush=True)
        for n, h in sorted(found, reverse=True)[:20]:
            print("  %14d B  %s" % (n, h), file=out, flush=True)

    def _loop():
        path = os.path.join(_OUT, "fastcensus_pid%d.txt" % os.getpid())
        out = open(path, "w", buffering=1)
        print("LEVER|snni_fast_census|armed pid=%d min=%d" % (os.getpid(), _MIN),
              file=out, flush=True)
        best = 0
        while True:
            r = _rss()
            if r > best * 1.002 and r > 0:
                best = r
                try:
                    _scan(out, r)
                except Exception as e:
                    print("FAST|scan_error|%r" % (e,), file=out, flush=True)
            time.sleep(0.05)

    threading.Thread(target=_loop, daemon=True).start()
