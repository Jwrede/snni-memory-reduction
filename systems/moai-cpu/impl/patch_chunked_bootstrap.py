#!/usr/bin/env python3
"""Paid lever: chunked bootstrap (port of moai_waterfall/17_stage_streaming to the phase-split line).

Target: Bootstrapper modraise (~40.9 GiB, working set of all 768 CTs) after the phase split. The 768
CTs pass the full phase-split bootstrap (stage_bootstrap_batch_split) in chunks of
BOOT_CHUNK_SIZE (=128); each chunk's output saved to disk and freed before the next; all reloaded at
the end. modraise ~40.9 -> ~7 GiB.
Bit-identical: same code, same order (0..767); gate md5 == m0. Paid: each chunk reloads the 3 boot-phase
keys (stage_load_boot_phase), 6 chunks = 6x the phase-key reads.
Dependencies present in the phase-split tree: jemalloc_purge_all()=malloc_trim(0) (glibc),
cleanup_main_log_rss (no-op), stage_bootstrap_batch_split, bs.context, <cstdio>.

    patch_chunked_bootstrap.py <moai source root>
"""
import os
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
ROOT = sys.argv[1]
TFS = os.path.join(ROOT, "include", "test", "test_full_scheme.hpp")
if not os.path.isfile(TFS):
    sys.exit(f"FAILED: {TFS} not found")

src = open(TFS, encoding="utf-8").read()
if "stage_bootstrap_chunked" in src:
    sys.exit("FAILED: this tree already carries the chunked-bootstrap lever")

CHUNKED_FN = r'''// SNNI s1-scout PAID lever: chunked bootstrap. Wraps the phase-split bootstrap
// (stage_bootstrap_batch_split) in chunks of BOOT_CHUNK_SIZE, saving/freeing chunk outputs to disk
// so only one chunk's modraise working set is resident. Bit-identical (same order), paid (re-loads
// the boot-phase keys per chunk).
#ifndef BOOT_CHUNK_SIZE
#define BOOT_CHUNK_SIZE 128
#endif

static void stage_bootstrap_chunked(Bootstrapper &bs, vector<Ciphertext> &rtn,
                                     vector<Ciphertext> &input, int count,
                                     const string &stage_name) {
    { static bool _snni_o=false; if(!_snni_o){_snni_o=true;
      fprintf(stderr,"LEVER|chunked_bootstrap|size=%d\n", BOOT_CHUNK_SIZE); fflush(stderr);} }
    int num_chunks = (count + BOOT_CHUNK_SIZE - 1) / BOOT_CHUNK_SIZE;
    string chunk_dir = "./";

    for (int cs = 0; cs < count; cs += BOOT_CHUNK_SIZE) {
        int ce = min(cs + BOOT_CHUNK_SIZE, count);
        int cn = ce - cs;
        int chunk_idx = cs / BOOT_CHUNK_SIZE;

        cout << "[CHUNK] " << stage_name << " chunk " << (chunk_idx + 1)
             << "/" << num_chunks << " (" << cn << " CTs)" << endl;
        cleanup_main_log_rss(stage_name + "_chunk_" + to_string(chunk_idx) + "_begin");

        // Move chunk input out of the main array
        vector<Ciphertext> chunk_in(cn);
        for (int i = 0; i < cn; i++) chunk_in[i] = move(input[cs + i]);
        jemalloc_purge_all();

        // Process this chunk through the full phase-split bootstrap
        vector<Ciphertext> chunk_out;
        stage_bootstrap_batch_split(bs, chunk_out, chunk_in, cn, stage_name + "_c" + to_string(chunk_idx));

        // Save chunk output to disk and free
        for (int i = 0; i < cn; i++) {
            ofstream out(chunk_dir + stage_name + "_" + to_string(cs + i) + ".ct", ios::binary);
            chunk_out[i].save(out, compr_mode_type::none);
            chunk_out[i] = Ciphertext();
        }
        vector<Ciphertext>().swap(chunk_out);
        jemalloc_purge_all();
        cleanup_main_log_rss(stage_name + "_chunk_" + to_string(chunk_idx) + "_done");
    }
    vector<Ciphertext>().swap(input);
    jemalloc_purge_all();

    // Reload all outputs from disk and delete each file immediately to free disk
    rtn.resize(count);
    for (int i = 0; i < count; i++) {
        string ct_path = chunk_dir + stage_name + "_" + to_string(i) + ".ct";
        {
            ifstream in(ct_path, ios::binary);
            rtn[i].load(bs.context, in);
        }
        remove(ct_path.c_str());
    }
    cleanup_main_log_rss(stage_name + "_chunks_reloaded");
}

'''

# 1. insert the chunked function at file scope, right before all_layer_test (after batch_split, so
#    stage_bootstrap_batch_split is declared). Anchor is unique.
ANCHOR = "// ===== end SNNI m7 phase_split =====\n\nvoid all_layer_test(){\n"
if src.count(ANCHOR) != 1:
    sys.exit(f"FAILED: insertion anchor not unique/found ({src.count(ANCHOR)})")
src = src.replace(ANCHOR, "// ===== end SNNI m7 phase_split =====\n\n" + CHUNKED_FN + "\nvoid all_layer_test(){\n", 1)

# 2. replace the 4 bootstrap call sites: batch_split -> chunked
CALLS = [
    '    stage_bootstrap_batch_split(bootstrapper, rtn, att_selfoutput, 768, "bootstrap1");',
    '    stage_bootstrap_batch_split(bootstrapper, rtn, layernorm_selfoutput, 768, "bootstrap2");',
    '    stage_bootstrap_batch_split(bootstrapper, rtn2, final_output, 768, "bootstrap3");',
    '    stage_bootstrap_batch_split(bootstrapper, rtn2, layernorm_finaloutput, 768, "bootstrap4");',
]
for call in CALLS:
    if src.count(call) != 1:
        sys.exit(f"FAILED: call site not unique/found: {call.strip()} ({src.count(call)})")
    src = src.replace(call, call.replace("stage_bootstrap_batch_split", "stage_bootstrap_chunked"), 1)

# 3. verify
for m in ("stage_bootstrap_chunked", "LEVER|chunked_bootstrap"):
    if m not in src:
        sys.exit(f"FAILED: applied but marker {m!r} absent")
if src.count("stage_bootstrap_chunked(bootstrapper") != 4:
    sys.exit(f"FAILED: expected 4 chunked calls, found {src.count('stage_bootstrap_chunked(bootstrapper')}")

open(TFS, "w", encoding="utf-8").write(src)
print("patch_chunked_bootstrap: stage_bootstrap_chunked inserted + 4 bootstrap calls chunked (BOOT_CHUNK_SIZE=128)")
