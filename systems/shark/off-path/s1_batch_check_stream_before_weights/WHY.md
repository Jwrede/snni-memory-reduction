# s1_batch_check_stream_before_weights: withdrawn order

| | |
|---|---|
| was | step `s1` at the T=4 operating point |
| ranking at T=4 | MAC-tag buffers `std::_Vector_base<unsigned __int128>::_M_allocate` 720.7 MB (u64 twin 360.6 MB) above the weights 679.7 MB |
| ranking at T=16 | weights 679.7 MB above the tag buffer 518.2 MB (`steps/s0_default/peak-objects.md`) |
| published order | `s1_layer_weights` (weights alone), `s2_batch_check_stream` (both levers) |
| these runs | the check alone on the baseline: peak 56,892,384 kB, run time inside the spread |
