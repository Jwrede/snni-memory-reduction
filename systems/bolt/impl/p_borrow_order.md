# BOLT: borrow order (derivation note, 2026-08-10)

State at the time: line `s0 -> s1_stream_weights -> s2_threads_4`, both paid. `s2`'s skip ("no free
fix remains: the one object above 1% is the same SEAL pool `s1` already attacked") rested on a depth-1
table with one row (`BOLT_BERT+0x1d1c3b`, near `seal::util::MemoryPoolHeadMT::get()`, 96.5%).
Current values: `steps.csv`.

## Depth probes

`p_depth3`: one replicate, `s1` image at `OMP_NUM_THREADS=4` (the `s2` configuration),
`SNNI_PM_DEPTH=3`, `express` (frame-choice probe). Binding party, 14,018.4 MB over 63 sites against a
17.19 GB peak:

| resident | count | site |
|---:|---:|---|
| 6171.5 MB | 38 | `stl_vector.h:558: std::vector<std::vector<seal::Plaintext>>::vector(...)` |
| 3177.7 MB | 111 | `BOLT_BERT+0x18bfba` (not yet symbolised) |
| 2282.5 MB | 20 | `stl_vector.h:558: std::vector<seal::Plaintext>::vector(...)` |
| 761.9 MB | 83 | `pointer.h:1235: seal::util::allocate<unsigned long>` |
| 365.4 MB | 12 | `stl_vector.h:558: std::vector<seal::Ciphertext>::vector(...)` |
| 362.2 MB | 10 | `linear.cpp:563: Linear::bert_cross_packing_single_matrix` |
| 200.8 MB | 24 | `linear.cpp:532: Linear::bert_cross_packing_single_matrix` |
| 89.6 MB | 512 | `bert.cpp:169: Bert::Bert(...)` |

| depth | top rows | level |
|---|---|---|
| 3 | 6171.5 MB `std::vector<std::vector<seal::Plaintext>>::vector` | container |
| 4 | 2609.0 / 2572.8 / 1533.8 MB `linear.cpp:681 / :680 / :677 bert_cross_packing_single_matrix_2` | fill line |
| 5 | 5176.8 MB `linear.cpp:223 params_preprocessing_ct_pt`, 2167.1 MB `linear.cpp:297 preprocess_layer(int)` | program stage |

- Depth 5 chosen (rule 4: one change lifts the whole group; the stage is the unit a deferral or a
  window acts on). Also at depth 5: 987.3 MB under `libgomp.so.1.0.0+0x1dc0e` (OpenMP per-thread).
- Baseline tables at depth 5 and 4: `p_depth5_base` (46009425), `p_depth4_base` (46009427). The
  baseline's `#pragma omp parallel for` adds an outlined function above the stage, so depth 5 still
  lands on the stage.

## Source

`weights_preprocess` (`linear.cpp:228`, baseline):

```
void Linear::weights_preprocess(BertModel &bm){
    #pragma omp parallel for
    for(int i = 0; i < ATTENTION_LAYERS; i++){
        pp_1[i] = params_preprocessing_ct_ct(he_8192, bm.w_q[i], ...);
```

- `params_preprocessing_ct_ct` calls `bert_cross_packing_single_matrix` three times, each returning
  a `vector<vector<Plaintext>>`: the encoded weight plaintexts of all twelve layers at once.
- The layers coexist because they are built in parallel. `s1_stream_weights` builds one layer at a
  time (`preprocess_layer(i)`, "no omp"): same work, the runtime cost is lost outer parallelism.

## Donors

| donor | cost there | mechanism | assessment for BOLT at the time |
|---|---|---|---|
| BumbleBee `b1_dot_encode_chunk` | free, 15.09 -> 8.41 GB | encode a slice, use it, drop it | mechanism class right; BOLT's weights are retained until their layer runs, not transient |
| BumbleBee `b2_result_ct_pack_free` | free, -1.09 GB | free dead result ciphertexts | candidate for the 365.4 MB `vector<Ciphertext>` row |
| BumbleBee `b3_weight_defer` | free, -0.58 GB | build a weight set at first use | same class as `b1` |
| BumbleBee `b4_slice_budget_64m` | paid, -0.24 GB | cap the slice by resident bytes | knob for a layer window |
| BumbleBee `b5`-`b8` | paid | fewer OT instances | shape of `s2` |
| SHARK `s2_layer_weights` | free | per-layer weight residency | this is `s1_stream_weights` (free on SHARK: no layer parallelism) |
| SHARK `s3_key_stream` | | stream key material | candidate for the 761.9 MB `seal::util::allocate<unsigned long>` |
| SHAFT-CPU `s1_plaintext_free` | marginal | release past last use | predicted inert: SEAL's pool retains freed memory, so only "do not materialise yet" works |

## Plan at the time

1. Free candidates first (`b2`-type on the ciphertext vectors; `b3`/`s3_key_stream`-type on
   `seal::util::allocate<unsigned long>`).
2. `p_alloc_style`: SHAFT-CPU `s1_plaintext_free` borrowed literally, expected zero.
3. The paid deferral as a window of `k` layers (`k/12` of the weights, `k`-way parallelism): measured
   in `p_window_curve.md`.
4. Thread count.
