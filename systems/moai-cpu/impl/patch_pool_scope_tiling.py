#!/usr/bin/env python3
"""MOAI-CPU pool_scope_tiling: moai_new/palma_campaign ct_ct fix: rotation phase on a caching
MemoryPoolHandle::New(true) pool (key-switch buffers reused from its free list; no per-rotation
munmap/mmap) plus CT_CT_TILE_SIZE tiling; global threshold restored before the babystep. diagpacking
arithmetic unchanged (gate bitwise); colpacking untouched.

    patch_pool_scope_tiling.py <moai source root>
"""
import os, sys

SRC = "include/source/matrix_mul/Ct_ct_matrix_mul.hpp"

TOP_FROM = 'using namespace std;\nusing namespace seal;\n'
TOP_TO   = '#include <cstdlib>\n#include <malloc.h>\n#ifndef CT_CT_TILE_SIZE\n#define CT_CT_TILE_SIZE 16\n#endif\nusing namespace std;\nusing namespace seal;\n'
FROM_A = '  //rotate X\n  \n  vector<Ciphertext> rot_enc_X(row_X);\n\n  #pragma omp parallel for \n  \n  for (int i = 0; i < b; ++i){\n    for (int j = 0 ; j < g ; ++j){\n      int index = i*g+j;\n      if(index >= row_X){\n        break;\n      }\n      else{\n        int rot_ind = (col_X-i*g)*num_batch;\n        if(rot_ind != col_X*num_batch){\n          evaluator.rotate_vector(enc_X[index],rot_ind,RotK,rot_enc_X[index]);\n        }\n        else{\n          rot_enc_X[index] = enc_X[index];\n        }\n      }\n    }\n  }\n'
TO_A   = '  //rotate X -- LEVER pool_scope_tiling: scope the rotation onto a caching New(true) pool so the\n  // reused key-switch buffers stay in a free-list (no per-rotation munmap/mmap on the global\n  // threshold) + CT_CT_TILE_SIZE tiling to bound the cached set. Ports moai_new/palma_campaign;\n  // math unchanged; the global threshold is restored before the babystep.\n  { static bool _snni_o=false; if(!_snni_o){_snni_o=true; fprintf(stderr,"LEVER|pool_scope_tiling|diag\\n"); fflush(stderr);} }\n  MemoryPoolHandle _snni_rot_pool = MemoryPoolHandle::New(true);\n  auto _snni_old_rp = MemoryManager::SwitchProfile(make_unique<MMProfFixed>(_snni_rot_pool));\n\n  vector<Ciphertext> rot_enc_X(row_X);\n\n  for (int tile_start = 0; tile_start < b; tile_start += CT_CT_TILE_SIZE) {\n    int tile_end = min(tile_start + CT_CT_TILE_SIZE, b);\n    #pragma omp parallel for\n    for (int i = tile_start; i < tile_end; ++i){\n      for (int j = 0 ; j < g ; ++j){\n        int index = i*g+j;\n        if(index >= row_X){\n          break;\n        }\n        else{\n          int rot_ind = (col_X-i*g)*num_batch;\n          if(rot_ind != col_X*num_batch){\n            evaluator.rotate_vector(enc_X[index],rot_ind,RotK,rot_enc_X[index]);\n          }\n          else{\n            rot_enc_X[index] = enc_X[index];\n          }\n        }\n      }\n    }\n    malloc_trim(0);\n  }\n  MemoryManager::SwitchProfile(move(_snni_old_rp));\n'
FROM_B = '  //baby step + gaint step (col_w times)\n  #pragma omp parallel for \n\n  for (int i = 0 ; i < col_W ; ++i){\n'
TO_B   = '  //baby step + gaint step (col_w times) -- tiled (LEVER pool_scope_tiling)\n  for (int tile_start = 0; tile_start < col_W; tile_start += CT_CT_TILE_SIZE) {\n    int tile_end = min(tile_start + CT_CT_TILE_SIZE, col_W);\n    #pragma omp parallel for\n    for (int i = tile_start ; i < tile_end ; ++i){\n'
FROM_C = '    }\n  }\n\n  return output;\n'
TO_C   = '    }\n    }\n    malloc_trim(0);\n  }\n\n  return output;\n'

def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()
    if "pool_scope_tiling" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    for name, a in (("TOP", TOP_FROM), ("A", FROM_A), ("B", FROM_B), ("C", FROM_C)):
        if txt.count(a) != 1:
            sys.exit(f"FAILED: FROM_{name} matched {txt.count(a)} times, expected 1")
    if txt.index(FROM_B) > txt.index(FROM_C):
        sys.exit("FAILED: babystep header is after its closing brace anchor")
    txt = txt.replace(TOP_FROM, TOP_TO, 1).replace(FROM_A, TO_A, 1).replace(FROM_B, TO_B, 1).replace(FROM_C, TO_C, 1)
    open(src, "w").write(txt)
    out = open(src).read()
    for what in ("LEVER|pool_scope_tiling", "CT_CT_TILE_SIZE", "#include <malloc.h>",
                 "MemoryPoolHandle::New(true)", "SwitchProfile(make_unique<MMProfFixed>",
                 "SwitchProfile(move(_snni_old_rp))"):
        if what not in out:
            sys.exit(f"FAILED: applied the patch but {what!r} is absent")
    # the scope must be restored before the babystep reads rot_enc_X
    if out.index("SwitchProfile(move(_snni_old_rp))") > out.index("baby step + gaint step"):
        sys.exit("FAILED: rotation pool restored after the babystep starts")
    if out.count("tile_start += CT_CT_TILE_SIZE") != 2:
        sys.exit("FAILED: expected exactly 2 tiled loops (rotation + babystep)")
    if out.count("malloc_trim(0)") != 2:
        sys.exit("FAILED: expected exactly 2 malloc_trim calls")
    print("patch_pool_scope_tiling: caching New(true) rotation pool + tiling applied (" + SRC + ")")

if __name__ == "__main__":
    main()
