import socket, time, threading
def server():
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 19999)); s.listen(1)
    c, _ = s.accept()
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        d = c.recv(16)
        if not d: break
        c.sendall(d)
t = threading.Thread(target=server, daemon=True); t.start()
time.sleep(0.5)
c = socket.socket(); c.connect(("127.0.0.1", 29999))
c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
n = 20; ts = []
for _ in range(n):
    a = time.perf_counter(); c.sendall(b"x"*8); c.recv(8); ts.append((time.perf_counter()-a)*1000)
ts.sort()
print("RTT_median_ms=%.3f  RTT_min_ms=%.3f  RTT_max_ms=%.3f" % (ts[n//2], ts[0], ts[-1]))
