# s3_inter_output_free: lever fires, peak unchanged

Jobs 46162014 (clean, 55:51, rc=0) and 46162015 (device attribution, 56:10, rc=0). Earlier ordering,
on `s2_boot_input_release`.

```
                    s2_boot_input_release   s3_inter_output_free
peak_alloc_kb             108,321,427            108,321,427     BYTE-IDENTICAL
peak_reserved_kb          123,928,576            122,585,088     -1.08%, inside this metric's noise
polled host peak            2,104,588              2,104,272     -0.015%
wall                            3,342                  3,344 s   +0.06%
marker after_gelu            99,634.1               71,986.1 MiB -27,648.0 MiB
gate.txt                                 md5 01469da3... IDENTICAL
```

## Wrong target

Chosen from `s2`'s device table:

```
42,316.3 MB  38.1%  secretkey.cu:275:encrypt_zero_symmetric        library, no admissible fix
33,843.8 MB  30.5%  evaluate.cu:1414:mod_switch_scale_to_next      library, no admissible fix
16,911.4 MB  15.2%  test.cu:70:main <- single_layer_test():868     program-owned, fix is PAID
 8,076.1 MB   7.3%  evaluate.cu:1555:mod_switch_to_next            library, no admissible fix
 6,442.5 MB   5.8%  test_single_layer.cuh:1073 <- gelu_v2          <- taken, and MISREAD
```

- The row is `gelu_v2` building `gelu_output` (allocation inside `gelu_v2`), read as GELU's input; the
  step released `inter_output`.

```
6,442.5 MB / 3072 = 2.097 MB = 2 MiB each   -> chain depth 1  -> gelu_output
27,648   MiB / 3072 =          9 MiB each   -> chain depth 8  -> inter_output
```

- `levels.txt`: `Modulus chain index for gelu: 1`.
- Both runs' derived tables total 110,921.1 MB, every object unchanged; six keys differ by relabelling
  only (call site moved seven lines by the patch; an inlining difference):

```
-1509.9  test_single_layer.cuh:1265:single_layer_test() <- layernorm2(...)
+1509.9  test_single_layer.cuh:1272:single_layer_test() <- layernorm2(...)
 -18.9   layernorm.cuh:1236 <- moai::Encoder::encode(...)
 +18.9   layernorm.cuh:1236 <- PhantomCKKSEncoder::encode<double>(...)      (and the same at :1149)
```

- `inter_output` is freed one line after the marker in the stock code:

```cpp
snni_mem_marker("after_gelu");
vector<PhantomCiphertext>().swap(inter_output);      // stock code frees the array HERE
```

## Peak location

`peak_alloc_kb` 105,783.6 MiB is above every marker:

```
after_intermediate    93,490.1 MiB
after_gelu            99,634.1        highest marker, 6,149.5 below the peak
after_final           72,754.1
after_bootstrap3      88,114.1
```

Between `after_gelu` and `after_final` runs one call:

```cpp
vector<PhantomCiphertext> final_output =
    ct_pt_matrix_mul_wo_pre_w_mask_single(gelu_output, final_weight, b_vec, ...);
```

Entering at 71,986.1, leaving at 72,754.1 MiB. `enc_X` (`gelu_output`) is read by every output
element; `nthreads` is clamped to 1. Measured next in `../p_finalprobe/` (the peak is not there; the
published value is a cumulative reservation).
