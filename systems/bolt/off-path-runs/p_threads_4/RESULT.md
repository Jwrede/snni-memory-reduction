# p_threads_4: harness defect during the thread probe

Jobs 45800831 (baseline image, by mistake) and 45802984 (`s1` image), 2026-08-05. The thread lever
was measured later and published as `s2_threads_4`.

| | |
|---|---|
| symptom | 45802984 ran 12 h without finishing (baseline: 6.6 min) |
| log | `party0_markers.txt: error: bind: Address already in use`; `party0_stdout.log: ... > Setup NonLinear` and nothing after |
| cause | port `8000 + SLURM_JOB_ID % 1000`: two own jobs 1000 ids apart on one node; party 0 failed to bind and kept running; `run_bolt.sh` reported "listener ... is up" unconditionally after 600 s |
| fix 1 | stop on `bind: Address already in use`; stop if no listener after the wait |
| fix 1 failure | 45802985 timed out the same way: the guard read `party0_stdout.log`, the message is in `party0_markers.txt` |
| fix 2 | guard reads both files; port checked against `/proc/net/tcp` and stepped up to 40 times before launch |
