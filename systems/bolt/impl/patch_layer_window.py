#!/usr/bin/env python3
"""BOLT: make the resident layer WINDOW (BOLT_LAYER_WINDOW) a parameter instead of the constant 1.
Applied to the s1_stream_weights overlay tree; k=1 reproduces s1 exactly.
Usage: python3 patch_layer_window.py <path to SCI/tests/bert_bolt>
"""
import os
import sys

BUILD_ANCHOR = """        // s4 weight streaming: build only THIS layer's HE-encoded weights now (JIT).
        if(party == ALICE){
            lin.preprocess_layer(layer_id);
        }
"""

BUILD_PATCH = """        // p_window: hold a WINDOW of BOLT_LAYER_WINDOW layers resident instead of exactly one.
        // k=1 reproduces s1_stream_weights exactly: one layer built here, freed below.
        if(party == ALICE){
            static int bolt_win = -1;
            static std::vector<char> bolt_resident;
            if(bolt_win < 0){
                const char *bw = getenv("BOLT_LAYER_WINDOW");
                bolt_win = (bw && atoi(bw) > 0) ? atoi(bw) : 1;
                if(bolt_win > ATTENTION_LAYERS) bolt_win = ATTENTION_LAYERS;
                bolt_resident.assign(ATTENTION_LAYERS, 0);
                std::cerr << "LEVER|layer_window|k=" << bolt_win << std::endl;
            }
            int bolt_hi = layer_id + bolt_win;
            if(bolt_hi > ATTENTION_LAYERS) bolt_hi = ATTENTION_LAYERS;
            std::vector<int> bolt_todo;
            for(int bj = layer_id; bj < bolt_hi; ++bj){
                if(!bolt_resident[bj]) bolt_todo.push_back(bj);
            }
            // The layers of a fresh window are independent -- each reads its own slices of
            // bm_stored and writes its own pp_*[j] -- which is why the stock upfront pass could
            // parallelise over all twelve. This restores that parallelism, to the window's width.
            #pragma omp parallel for
            for(int bt = 0; bt < (int)bolt_todo.size(); ++bt){
                lin.preprocess_layer(bolt_todo[bt]);
            }
            for(int bt = 0; bt < (int)bolt_todo.size(); ++bt){
                bolt_resident[bolt_todo[bt]] = 1;
            }
        }
"""

FREE_ANCHOR = """        // s4 weight streaming: release this layer's encoded weights (keep ~1 layer resident).
        if(party == ALICE){
            lin.free_layer(layer_id);
        }
"""

# The free site is deliberately not patched: free_layer(layer_id) already releases the finished layer
# correctly at any window width, and the loop counter only rises so no freed layer is revisited.


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = sys.argv[1]
    src = os.path.join(root, "bert.cpp")
    if not os.path.exists(src):
        sys.exit("FAILED: no bert.cpp under " + root)
    txt = open(src).read()

    if "LEVER|layer_window" in txt:
        sys.exit("FAILED: bert.cpp already carries the window lever")
    if txt.count(BUILD_ANCHOR) != 1:
        sys.exit("FAILED: the s1 build call site is not present exactly once "
                 f"(found {txt.count(BUILD_ANCHOR)}); this patch expects the "
                 "s1_stream_weights overlay, not the pristine tree")
    if txt.count(FREE_ANCHOR) != 1:
        sys.exit("FAILED: the s1 free call site is not present exactly once "
                 f"(found {txt.count(FREE_ANCHOR)})")

    txt = txt.replace(BUILD_ANCHOR, BUILD_PATCH, 1)

    # Ensure <cstdlib> (getenv/atoi) so a missing include does not fail deep in a 40-minute build.
    if "#include <cstdlib>" not in txt:
        first = txt.index("#include")
        txt = txt[:first] + "#include <cstdlib>\n" + txt[first:]

    open(src, "w").write(txt)
    print("patch_layer_window: bert.cpp patched, window read from BOLT_LAYER_WINDOW (default 1)")


if __name__ == "__main__":
    main()
