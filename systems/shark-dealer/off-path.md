# SHARK-DEALER: off-path

## Status: closed at the target

| step | peak (kB) | `W` | cost |
|---|---:|---:|---|
| `d0_default` | 867,360 | 4.76 | baseline |
| `d1_layer_weights` | 258,312 | 1.40 | free (-0.4%); -70.2%, 3.358x |

| | |
|---|---|
| protocol target | 181,248 kB |
| instrument | 4,565 kB |
| excess over target | 72,499 kB = 70.8 MB |
| largest object | `shark::span<unsigned __int128>::span(unsigned long)`, 113.3 MB, 42.8% |

The largest object exceeds the whole gap to the target, so it is target memory. Per-pool stop at
the floor (README.md step 5).

## Object tables

```
d0   679.7 MB  76.5%  std::vector<shark::span<unsigned long>>::operator...   <- d1 attacks this
     113.3 MB  12.8%  shark::span<unsigned __int128>::span
d1   113.3 MB  42.8%  shark::span<unsigned __int128>::span                  <- unchanged
      56.6 MB  21.4%  std::vector<shark::span<unsigned long>>::operator...   <- fell 91.7%
```

## Measured and excluded

`off-path-runs/p_batch_check_only` (jobs 45774969-71, 3 replicates): the batch-check lever carried
in `d1`'s image is inert for the dealer, so `d1`'s -70.2% is the weight mechanism alone.

## Trades

None. Fewer triples, shorter keys or another correlation would change the protocol.

## Relation to the SHARK online line

Separate system and `steps.csv` (as `llama` / `llama-dealer`); a lever's effect in one role is not
evidence for the other.
