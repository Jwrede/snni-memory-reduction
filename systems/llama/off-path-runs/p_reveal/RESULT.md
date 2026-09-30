# p_reveal: the zero output

Job 45768413, 2026-08-04, one layer, instrumented overlay (`build/overlay-probe`, image
`llama-online-probe.sif`). Diagnostic; no peak taken.

`outputA` gives the client its share minus a mask from the key file. The probe prints, per party,
the nonzero count and the first eight raw values before and after `outputA`.

```
PROBE|party=2|before|nonzero=85911/98304|6755399441055745,2251799813685249,2251799813685248,6755399441055745,...
PROBE|party=2|after |nonzero=85911/98304|6755399441055745,2251799813685249,2251799813685248,6755399441055745,...
PROBE|party=3|before|nonzero=85911/98304|6755399441055745,2251799813685249,2251799813685248,6755399441055745,...
PROBE|party=3|after |nonzero=0/98304    |0,0,0,0,0,0,0,0
PROBE|party=1|before|nonzero=49122/98304|1,1,0,1,0,0,1,0
PROBE|party=1|after |nonzero=0/98304    |0,0,0,0,0,0,0,0
```

| observation | meaning |
|---|---|
| parties 2 and 3 hold identical activations | masked evaluation: both know `x + r`, the dealer holds `r`; `outputA` has no branch for party 2 |
| dealer masks are one bit (49,122 of 98,304 nonzero) | `LlamaConfig::bitlength` is 51; a mask should be a uniform 51-bit element |
| online values are `k * 2^51 + b` | below |

| raw value | decomposition |
|---:|---|
| 6755399441055745 | `3 * 2^51 + 1` |
| 2251799813685249 | `1 * 2^51 + 1` |
| 2251799813685248 | `1 * 2^51 + 0` |
| 4503599627370497 | `2 * 2^51 + 1` |

- `outputA` reduces mod `bitlength = 51` before subtracting: the remaining low bit equals the mask,
  so the result is zero everywhere regardless of the computation.
- Established: reconstruction not broken, shares not mismatched.
- Not established: whether the one-bit mask or the result above bit 51 is upstream (e.g. `bitlength`
  reaching the mask generator as 1, or a skipped scale-down letting scale-24 products pass the ring).
  Key-generation source read: job 45768584.
- Consequence: the value observable cannot fail; the gate is structural (`T1s`, `../p_wire/`,
  `../p_disc/`).
