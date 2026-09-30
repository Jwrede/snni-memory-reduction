# p_disc: discrimination test for the T1s gate

Jobs 45803007 (`SIGMA_SEQ=64`) and 45803020 (`SIGMA_THREADS=4`), against the four reference runs of
`p_det1..4`, 2026-08-05.

| run | Total Comm | Key size |
|---|---:|---:|
| reference, seq 128, 16 threads (n=4) | 1,062,390,674 B | 18,075,947,008 B |
| structural change: sequence length 64 | 441,539,858 B | 7,978,659,840 B |
| non-structural change: 4 threads | 1,062,390,674 B | 18,075,947,008 B |

- Half the sequence: 0.416 of the volume (attention quadratic, the rest linear).
- Four threads: byte-identical (also this system's answer to BumbleBee `b5`-`b8` for the transcript).
- Instrument defect: the first version built `gate.txt` in the runner from `invariants.txt`, which the
  sbatch writes after the runner exits; two complete runs exited 5. The gate is now built beside its
  input.
