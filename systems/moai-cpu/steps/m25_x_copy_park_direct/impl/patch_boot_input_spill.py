#!/usr/bin/env python3
"""MOAI-CPU boot_input_spill: bootstrap input chunks after the first written to disk, each read back
right before its chunk runs.

    patch_boot_input_spill.py <moai source tree>

The chunked bootstrap (m7) consumes one chunk at a time (chunk_in[i] = move(input[cs+i])) but all 768
inputs at the layernorm level are resident from the start: 14.1 GB at m20's peak (layer 0
bootstrap2), largest object after the live phase keys. Keeps one chunk's input resident instead of
six. Paid (one write, one read per ciphertext per bootstrap). Same bytes (compr_mode none, as for the
chunk outputs); gate unchanged.
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"

ANCHOR_LOOP = """    int num_chunks = (count + BOOT_CHUNK_SIZE - 1) / BOOT_CHUNK_SIZE;
    string chunk_dir = "./";

    for (int cs = 0; cs < count; cs += BOOT_CHUNK_SIZE) {
"""
SPILL = """    int num_chunks = (count + BOOT_CHUNK_SIZE - 1) / BOOT_CHUNK_SIZE;
    string chunk_dir = "./";

    // boot_input_spill: every chunk's input is resident from the start although the loop below
    // consumes one chunk at a time. Write the chunks after the first to disk now and free them; each
    // is read back right before its own chunk runs. Same bytes (compr_mode none, as the chunk outputs).
    if (num_chunks > 1) {
        int spilled = 0;
        for (int i = BOOT_CHUNK_SIZE; i < count; i++) {
            string in_path = chunk_dir + stage_name + "_in_" + to_string(i) + ".ct";
            ofstream out(in_path, ios::binary);
            if (!out) {
                fprintf(stderr, "LEVER|boot_input_spill|FAILED|cannot open %s\\n", in_path.c_str());
                fflush(stderr);
                exit(9);
            }
            input[i].save(out, compr_mode_type::none);
            input[i] = Ciphertext();
            spilled++;
        }
        jemalloc_purge_all();
        fprintf(stderr, "LEVER|boot_input_spill|stage=%s|spilled=%d|resident=%d\\n",
                stage_name.c_str(), spilled, BOOT_CHUNK_SIZE);
        fflush(stderr);
    }

    for (int cs = 0; cs < count; cs += BOOT_CHUNK_SIZE) {
"""

ANCHOR_MOVE = """        // Move chunk input out of the main array
        vector<Ciphertext> chunk_in(cn);
        for (int i = 0; i < cn; i++) chunk_in[i] = move(input[cs + i]);
        jemalloc_purge_all();
"""
RELOAD = """        // Move chunk input out of the main array; chunks after the first come back from disk
        // (boot_input_spill), one chunk at a time, and their files are deleted as they are read.
        vector<Ciphertext> chunk_in(cn);
        if (cs == 0) {
            for (int i = 0; i < cn; i++) chunk_in[i] = move(input[cs + i]);
        } else {
            for (int i = 0; i < cn; i++) {
                string in_path = chunk_dir + stage_name + "_in_" + to_string(cs + i) + ".ct";
                {
                    ifstream in(in_path, ios::binary);
                    if (!in) {
                        fprintf(stderr, "LEVER|boot_input_spill|FAILED|cannot reopen %s\\n", in_path.c_str());
                        fflush(stderr);
                        exit(9);
                    }
                    chunk_in[i].load(bs.context, in);
                }
                remove(in_path.c_str());
            }
        }
        jemalloc_purge_all();
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()
    if "boot_input_spill" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    for name, a in (("loop", ANCHOR_LOOP), ("move", ANCHOR_MOVE)):
        if txt.count(a) != 1:
            sys.exit(f"FAILED: the {name} anchor appears {txt.count(a)} times, expected 1")
    txt = txt.replace(ANCHOR_LOOP, SPILL, 1).replace(ANCHOR_MOVE, RELOAD, 1)
    open(src, "w").write(txt)
    print("patch_boot_input_spill: input chunks after the first spilled to disk and reloaded per chunk (" + SRC + ")")


if __name__ == "__main__":
    main()
