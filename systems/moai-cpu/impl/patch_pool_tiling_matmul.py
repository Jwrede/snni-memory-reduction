#!/usr/bin/env python3
"""MOAI-CPU pool_tiling_matmul: port of moai_waterfall/palma_campaign ct_ct diagpacking (scoped
MemoryPoolThresholdMT rotation pool + CT_CT_TILE_SIZE tiling + malloc_trim). diagpacking arithmetic
identical (only pool/tiling wrapper and thread model differ; gate bitwise); colpacking untouched; plain
omp pragma (OMP_NUM_THREADS), not palma's 48 threads. Targets per-step runtime churn.

    patch_pool_tiling_matmul.py <moai source root>
"""
import os, sys

SRC = "include/source/matrix_mul/Ct_ct_matrix_mul.hpp"

TOP_FROM = 'using namespace std;\nusing namespace seal;\n'
TOP_TO   = '#include <cstdlib>\n#include <malloc.h>\n#ifndef CT_CT_TILE_SIZE\n#define CT_CT_TILE_SIZE 16\n#endif\nusing namespace std;\nusing namespace seal;\n'
FROM_A = '  //rotate X\n  \n  vector<Ciphertext> rot_enc_X(row_X);\n\n  #pragma omp parallel for \n  \n  for (int i = 0; i < b; ++i){\n    for (int j = 0 ; j < g ; ++j){\n      int index = i*g+j;\n      if(index >= row_X){\n        break;\n      }\n      else{\n        int rot_ind = (col_X-i*g)*num_batch;\n        if(rot_ind != col_X*num_batch){\n          evaluator.rotate_vector(enc_X[index],rot_ind,RotK,rot_enc_X[index]);\n        }\n        else{\n          rot_enc_X[index] = enc_X[index];\n        }\n      }\n    }\n  }\n'
TO_A   = '  //rotate X -- LEVER pool_tiling_matmul: CT_CT_TILE_SIZE tiling + malloc_trim over the rotation\n  // phase (ported from moai_waterfall/palma_campaign ct_ct diagpacking; math unchanged). The\n  // threshold pool is already the global default on this line (m1_pool_threshold), so palma\'s\n  // scoped pool is dropped -- tiling bounds concurrent rotation buffers, malloc_trim returns arena.\n  { static bool _snni_o=false; if(!_snni_o){_snni_o=true; fprintf(stderr,"LEVER|pool_tiling_matmul|diag\\n"); fflush(stderr);} }\n\n  vector<Ciphertext> rot_enc_X(row_X);\n\n  for (int tile_start = 0; tile_start < b; tile_start += CT_CT_TILE_SIZE) {\n    int tile_end = min(tile_start + CT_CT_TILE_SIZE, b);\n    #pragma omp parallel for\n    for (int i = tile_start; i < tile_end; ++i){\n      for (int j = 0 ; j < g ; ++j){\n        int index = i*g+j;\n        if(index >= row_X){\n          break;\n        }\n        else{\n          int rot_ind = (col_X-i*g)*num_batch;\n          if(rot_ind != col_X*num_batch){\n            evaluator.rotate_vector(enc_X[index],rot_ind,RotK,rot_enc_X[index]);\n          }\n          else{\n            rot_enc_X[index] = enc_X[index];\n          }\n        }\n      }\n    }\n    malloc_trim(0);\n  }\n'
FROM_B = '  //baby step + gaint step (col_w times)\n  #pragma omp parallel for \n\n  for (int i = 0 ; i < col_W ; ++i){\n'
TO_B   = '  //baby step + gaint step (col_w times) -- tiled (LEVER pool_tiling_matmul)\n  for (int tile_start = 0; tile_start < col_W; tile_start += CT_CT_TILE_SIZE) {\n    int tile_end = min(tile_start + CT_CT_TILE_SIZE, col_W);\n    #pragma omp parallel for\n    for (int i = tile_start ; i < tile_end ; ++i){\n'
FROM_C = '    }\n  }\n\n  return output;\n'
TO_C   = '    }\n    }\n    malloc_trim(0);\n  }\n\n  return output;\n'

def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()
    if "pool_tiling_matmul" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    for name, a in (("TOP", TOP_FROM), ("A", FROM_A), ("B", FROM_B), ("C", FROM_C)):
        if txt.count(a) != 1:
            sys.exit(f"FAILED: FROM_{name} matched {txt.count(a)} times, expected 1")
    # C anchor closes the babystep loop; it must sit after B.
    if txt.index(FROM_B) > txt.index(FROM_C):
        sys.exit("FAILED: babystep header is after its closing brace anchor")
    txt = txt.replace(TOP_FROM, TOP_TO, 1).replace(FROM_A, TO_A, 1).replace(FROM_B, TO_B, 1).replace(FROM_C, TO_C, 1)
    open(src, "w").write(txt)
    out = open(src).read()
    for what in ("LEVER|pool_tiling_matmul", "CT_CT_TILE_SIZE", "#include <malloc.h>"):
        if what not in out:
            sys.exit(f"FAILED: applied the patch but {what!r} is absent")
    # tiling present on both phases
    if out.count("tile_start += CT_CT_TILE_SIZE") != 2:
        sys.exit("FAILED: expected exactly 2 tiled loops (rotation + babystep)")
    if out.count("malloc_trim(0)") != 2:
        sys.exit("FAILED: expected exactly 2 malloc_trim calls")
    print("patch_pool_tiling_matmul: diagpacking pool+tiling applied (" + SRC + ")")

if __name__ == "__main__":
    main()
