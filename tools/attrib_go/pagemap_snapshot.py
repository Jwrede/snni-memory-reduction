#!/usr/bin/env python3
"""Page residency snapshot of every readable anonymous mapping of a stopped process.

    pagemap_snapshot.py <pid> <out.bin>

Caller must SIGSTOP the target. Output, little-endian: per mapping `u64 start, u64 n_pages,
presence bitmap`.
"""
import struct
import sys


def main():
    pid, out = sys.argv[1], sys.argv[2]
    page = 4096
    ranges = []
    with open(f"/proc/{pid}/maps") as fh:
        for ln in fh:
            parts = ln.split()
            addrs, perms = parts[0], parts[1]
            path = parts[5] if len(parts) > 5 else ""
            if "r" not in perms:
                continue
            # Anonymous only (no path, or pseudo [heap]/[stack]): the go heap lives there.
            if path and not path.startswith("["):
                continue
            if path in ("[vvar]", "[vdso]", "[vsyscall]"):
                continue
            lo, hi = (int(x, 16) for x in addrs.split("-"))
            ranges.append((lo, hi))

    st = open(f"/proc/{pid}/stat").read().split()
    if st[2] != "T":
        sys.exit(f"REFUSING: process {pid} is in state {st[2]}, not T (stopped). A residency "
                 f"snapshot of a running process describes no instant.")

    written = 0
    present_pages = 0
    total_pages = 0
    with open(f"/proc/{pid}/pagemap", "rb") as pm, open(out, "wb") as o:
        for lo, hi in ranges:
            n = (hi - lo) // page
            pm.seek(lo // page * 8)
            data = pm.read(n * 8)
            if len(data) != n * 8:
                continue
            bits = bytearray((n + 7) // 8)
            cnt = 0
            for i in range(n):
                if data[i * 8 + 7] & 0x80:  # bit 63: page present in RAM
                    bits[i // 8] |= 1 << (i % 8)
                    cnt += 1
            o.write(struct.pack("<QQ", lo, n))
            o.write(bits)
            written += 1
            present_pages += cnt
            total_pages += n
    print(f"PAGEMAP_SNAPSHOT|mappings={written}|pages={total_pages}|present={present_pages}"
          f"|present_mb={present_pages * page / 1e6:.1f}")


if __name__ == "__main__":
    main()
