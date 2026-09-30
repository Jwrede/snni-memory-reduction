# SHARK at T=4: thread-count control

Complete line at `OMP_NUM_THREADS=4`, six steps, 3 replicates, same images and levers, leader-only
freeze. `steps.csv` here is regenerated with the current `tools/make_steps.py` (maximum over the
parties). Step order: batch check first, weights second.

```
step                   T=4 peak kB   T=16 peak kB   difference   T=4 cost          T=16 cost
s0_default              58,098,544     58,089,352     -0.02 %    baseline          baseline
batch_check_stream      56,891,436     56,892,384     +0.00 %    free  (-1.4 %)    see steps.csv
layer_weights (both)    56,283,336     56,284,912     +0.00 %    free  (-0.2 %)    see steps.csv
s3_key_stream            1,198,664      1,200,076     +0.12 %    free  (+6.9 %, R 8.0 %)   paid (+5.1 %, R 1.1 %)
s4_ars_charge              510,336        510,060     -0.05 %    free  (-0.8 %)    paid (+14.4 %, R 3.0 %)
s5_up_split                467,720        469,792     +0.44 %    free  (-2.6 %)    free (-0.8 %)
```

- Peaks agree within 0.5%; step costs do not transfer between thread counts.
- The A/B of the streaming mechanism at T=4 (`../off-path-runs/s1b_stream_buffered`, +6.4% at 1.0%
  spread) agrees with the T=16 step cost.
- Used for one statement of the thesis (Section 4.1).
