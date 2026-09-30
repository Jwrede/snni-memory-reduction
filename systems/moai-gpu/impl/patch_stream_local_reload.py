#!/usr/bin/env python3
"""MOAI-GPU s8: FFN reload on the worker thread's stream instead of the default stream. Free.

    patch_stream_local_reload.py <moai source tree>

Basis: max live 64,306 MiB vs peak reservation 82,176 MiB (17,870 MiB = 21.7% fragmentation). Donor:
retired s24_stream_local_reload (default-stream syncs as fragmentation source, 65.19 -> 60.97 GiB).
GELU loop already uses stream_pool[tid]; the reload alone used the default stream
(cudaStreamSynchronize for all 3,072 cts). Serial reload sites (LN1 residual, LayerNorm2) unchanged.
Free. Expected -10% to -20% (66,000-74,000 MiB). Gate T1s.
"""

import os
import sys

SRC = "src/include/test/test_single_layer.cuh"

OLD = """                // s4_chunk_ffn: the FFN output lives on the host; reload just this one.
                PhantomCiphertext _gin;
                reload_cipher_from_host(_gin, inter_output_host[i*32+j]);
                gelu_output[i*32+j] = gelu_v2(_gin,context,relin_keys,secret_key, stream);
"""

NEW = """                // s4_chunk_ffn: the FFN output lives on the host; reload just this one.
                // s8_stream_local_reload: on the THREAD's stream, not the default one. The helper
                // ends in cudaStreamSynchronize on whichever stream it is given, and on the default
                // stream that is a synchronisation point every worker hits for every one of 3,072
                // ciphertexts, in the middle of the phase where the pool allocates hardest. The
                // retired campaign measured that as its fragmentation source and ~3 GiB of pool.
                PhantomCiphertext _gin;
                reload_cipher_from_host(_gin, inter_output_host[i*32+j], stream);
                gelu_output[i*32+j] = gelu_v2(_gin,context,relin_keys,secret_key, stream);
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    tree = sys.argv[1]
    src = os.path.join(tree, SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + src)
    txt = open(src).read()

    if "s8_stream_local_reload" in txt:
        sys.exit("FAILED: this tree already carries the lever")

    # The stream must be in scope at the site: passing the wrong stream would compile and silently
    # measure something else.
    if "auto &stream = stream_pool[tid];" not in txt:
        sys.exit("FAILED: no per-thread stream (`auto &stream = stream_pool[tid];`) in this tree; "
                 "the GELU loop is not threaded here and this lever has nothing to hand it")

    # The helper must accept a stream (ours does, with a default arg); a tree built before
    # borrowed/wpark_helpers.cuh gained the parameter would compile as a two-arg overload and fail.
    helpers = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "borrowed", "wpark_helpers.cuh")
    if os.path.exists(helpers):
        h = open(helpers).read()
        if "cuda_stream_wrapper &stream" not in h:
            sys.exit("FAILED: borrowed/wpark_helpers.cuh has no stream parameter on "
                     "reload_cipher_from_host; this lever cannot be applied against it")

    if txt.count(OLD) != 1:
        sys.exit(f"FAILED: the GELU reload site matched {txt.count(OLD)} times, expected exactly 1")

    txt = txt.replace(OLD, NEW, 1)

    # The two serial sites must be unchanged: this lever is one line and must not rewrite the
    # LN1-residual or LayerNorm2 reloads.
    if txt.count("reload_cipher_from_host(_res, ") != 2:
        sys.exit("FAILED: the two serial reload sites are not both present and untouched; this "
                 "lever changes exactly one call")

    open(src, "w").write(txt)
    print("patch_stream_local_reload: the FFN reload runs on the worker thread's stream")


if __name__ == "__main__":
    main()
