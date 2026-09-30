package main

// SNNI Go-side attribution adapter (M15). On SIGUSR1, write the runtime's own heap dump to
// SNNI_HEAPDUMP_PATH. WriteHeapDump stops the world for the duration of the write, so the dump
// describes ONE instant by construction -- the Go analogue of the freeze-then-capture rule.
//
// Deliberately NO runtime.GC() before the dump: garbage not yet collected is resident memory,
// and resident memory is the quantity the campaign measures. A GC here would shrink the heap the
// instant before describing it, i.e. describe a peak the program did not have.
//
// The adapter is observational: it changes no draw, no computation, and no allocation the
// program itself makes. Its cost is the stop-the-world pause of the dump plus the dump file,
// both of which occur only when the signal is actually sent (a diagnostic run), never in a
// clean measurement run.

import (
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"runtime/debug"
	"strconv"
	"syscall"
)

func init() {
	path := os.Getenv("SNNI_HEAPDUMP_PATH")
	if path == "" {
		return
	}
	// The dump's kind-16/17 records map a live object to the call stack that allocated it, but
	// only for allocations the profiler sampled. The rate is the campaign's own recorder floor
	// (SNNI_PM_MIN, 64 KiB): an allocation of b bytes is sampled with p = 1-exp(-b/rate), which
	// is ~1 for the MB-scale limb buffers this system's peak is made of and negligible for the
	// small fry -- the same see-large-objects contract pmrec.so has on the C systems, in
	// probabilistic form. rate=1 (sample everything) was tried on the tiny workload and
	// multiplies the runtime several-fold: a stack capture per allocation is exactly the
	// per-allocation cost the campaign refused when it retired memray. The coverage actually
	// achieved is printed per dump by the parser (sited_bytes), so the floor's effect is a
	// published number rather than an assumption.
	rate := 65536
	if v := os.Getenv("SNNI_PM_MIN"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			rate = n
		}
	}
	runtime.MemProfileRate = rate
	ch := make(chan os.Signal, 4)
	signal.Notify(ch, syscall.SIGUSR1)
	go func() {
		n := 0
		for range ch {
			n++
			p := fmt.Sprintf("%s.%d", path, n)
			f, err := os.Create(p)
			if err != nil {
				fmt.Fprintf(os.Stderr, "SNNI_HEAPDUMP|error=%v\n", err)
				continue
			}
			debug.WriteHeapDump(f.Fd())
			st, _ := f.Stat()
			f.Close()
			fmt.Fprintf(os.Stderr, "SNNI_HEAPDUMP|written=%s|bytes=%d\n", p, st.Size())
		}
	}()
	fmt.Fprintf(os.Stderr, "SNNI_HEAPDUMP|armed|path=%s\n", path)
}
