#!/usr/bin/env python3
"""Selects the measured workload in the pristine MOAI-GPU source (select_workload.py <source-root>).

Uncomments single_layer_test() and comments out upstream's RMSNorm_test(). Part of the measured
system, same at every step; no measurement. Edits in place, refuses a second application.
"""
import sys

CALL_OLD = [
    '    // cout << "single layer test" << endl;',
    '    // single_layer_test();',
    '    // cout << "single layer test passed!" << endl;',
]
CALL_NEW = [
    '    cout << "single layer test" << endl;',
    '    single_layer_test();',
    '    cout << "single layer test passed!" << endl;',
]

# Upstream leaves this live; it must go, or the peak is the BERT layer plus an unrelated
# 4096-ciphertext test.
RMS_OLD = [
    '    cout << "RMSNorm test" << endl;',
    '    RMSNorm_test();',
    '    cout << "RMSNorm test passed!" << endl;',
]


def fail(msg):
    sys.exit(f"select_workload.py: {msg}")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = sys.argv[1].rstrip('/')
    path = f"{root}/src/test.cu"
    src = open(path).read()

    if '\n    single_layer_test();' in src:
        fail(f"{path} already selects the single layer test, refusing to apply twice")

    # The call only compiles because include.cuh pulls the header in; checked here rather than assumed.
    inc = f"{root}/src/include/include.cuh"
    if '#include "test/test_single_layer.cuh"' not in open(inc).read():
        fail(f"{inc} does not include test/test_single_layer.cuh, so the call will not compile")

    for old, new in zip(CALL_OLD, CALL_NEW):
        if old not in src:
            fail(f"expected commented line not found: {old!r}")
        src = src.replace(old, new, 1)

    for line in RMS_OLD:
        if line not in src:
            fail(f"expected live RMSNorm line not found: {line!r}; "
                 "upstream may have changed which test is selected")
        src = src.replace(line, '    // ' + line.strip(), 1)

    open(path, 'w').write(src)

    # Prove the result: a selection that silently did nothing is the state this script removes.
    check = open(path).read()
    live = [ln.strip() for ln in check.splitlines()
            if ln.strip() and not ln.strip().startswith('//')
            and ln.strip().endswith('_test();')]
    if live != ['single_layer_test();']:
        fail(f"after editing, the live test calls are {live}, expected exactly "
             "['single_layer_test();']")
    print(f"  test.cu: single_layer_test selected, RMSNorm_test disabled")
    print(f"  live test calls now: {live}")


if __name__ == '__main__':
    main()
