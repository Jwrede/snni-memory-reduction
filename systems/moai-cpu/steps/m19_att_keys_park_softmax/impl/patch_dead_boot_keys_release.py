#!/usr/bin/env python3
"""MOAI-CPU dead_boot_keys_release: drop the last-loaded bootstrap phase keys when a bootstrap ends.

    patch_dead_boot_keys_release.py <moai source tree>

Phase-split bootstrap loads three key phases into bs.gal_keys (stage_load_boot_phase /
stage_softmax_load_boot_phase), each replacing the previous; the last stays until the next
bootstrap's first load. m8/m9 live-pool dumps: 385 x 37.7 MB = 14.5 GB of softmax phase keys at the
attention peak; 14.5 GB of main phase keys through the FFN. Every consumer loads its own phase first.
Bit-identical, free (next bootstrap reloads phase 0 anyway).
Where: (1) end of stage_softmax_bootstrap_split (softmax.hpp); (2) end of stage_bootstrap_chunked
(test_full_scheme.hpp), after the chunk outputs are reloaded.
"""
import os
import sys

EDITS = [
    ("include/source/non_linear_func/softmax.hpp",
     '  cleanup_softmax_log_rss("softmax_split_stc3_done");\n}',
     '  cleanup_softmax_log_rss("softmax_split_stc3_done");\n'
     '  // dead_boot_keys_release: the last phase loaded is dead once this bootstrap returns.\n'
     '  bs.gal_keys = GaloisKeys();\n'
     '  malloc_trim(0);\n'
     '  { static bool _snni_dk = false; if (!_snni_dk) { _snni_dk = true;\n'
     '      fprintf(stderr, "LEVER|dead_boot_keys_release|softmax\\n"); fflush(stderr); } }\n'
     '}'),
    ("include/test/test_full_scheme.hpp",
     '    cleanup_main_log_rss(stage_name + "_chunks_reloaded");\n}',
     '    cleanup_main_log_rss(stage_name + "_chunks_reloaded");\n'
     '    // dead_boot_keys_release: the last phase loaded is dead until the next bootstrap loads its own.\n'
     '    bs.gal_keys = GaloisKeys();\n'
     '    malloc_trim(0);\n'
     '    { static bool _snni_dk = false; if (!_snni_dk) { _snni_dk = true;\n'
     '        fprintf(stderr, "LEVER|dead_boot_keys_release|main\\n"); fflush(stderr); } }\n'
     '}'),
]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = sys.argv[1]
    for rel, old, new in EDITS:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            sys.exit("FAILED: no " + rel + " under " + root)
        txt = open(path).read()
        if "dead_boot_keys_release" in txt:
            sys.exit("FAILED: " + rel + " already carries the lever")
        n = txt.count(old)
        if n != 1:
            sys.exit(f"FAILED: anchor in {rel} appears {n} times, expected 1")
        txt = txt.replace(old, new, 1)
        open(path, "w").write(txt)
        if "dead_boot_keys_release" not in open(path).read():
            sys.exit("FAILED: applied but marker absent in " + rel)
        print("patch_dead_boot_keys_release: " + rel)


if __name__ == "__main__":
    main()
