#!/usr/bin/env python3
"""Fix of the phase-split bootstrap port: two-ciphertext FULL path (bootstrap_full_3, used by m0 since
logn==logNh==15) instead of the single-ciphertext REAL path (coefftoslot_3/sflinv_3).

Defect: stage_bootstrap_batch_split (test_full_scheme.hpp) and stage_softmax_bootstrap_split
(softmax.hpp) implemented sflinv_3 (CTS3 coeff_logn=logn+1) + one complex_conjugate + one modred
(coefftoslot_3/slottocoeff_3); bootstrap_3 dispatches to bootstrap_full_3 (coefftoslot_full_3 ->
rtn1,rtn2 -> 2x modred -> slottocoeff_full_3). Second ciphertext (imaginary slots) dropped; gate != m0.
Fix (op for op from Bootstrapper.cpp coefftoslot_full_3/slottocoeff_full_3/sflinv_full_3/sfl_full_3):
  - CTS3 coeff_logn: logn+1 -> logn  (sflinv_full_3 uses logn)
  - after CTS3 (phase 2 loaded): full split -> rtn1 = t1+conj(t1); rtn2 = (-i*t1)+conj(-i*t1)
  - modred BOTH rtn1 and rtn2
  - before STC1: recombine tmpct3 = modrtn1 + i*modrtn2, feed to STC1
STC transforms and phase-key streaming unchanged; phase keys unchanged (conjugate key = step 0 in
every phase).

    patch_ps_full3_fix.py <moai source root>
"""
import os, sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
ROOT = sys.argv[1]
TFS = os.path.join(ROOT, "include", "test", "test_full_scheme.hpp")
SM = os.path.join(ROOT, "include", "source", "non_linear_func", "softmax.hpp")
for p in (TFS, SM):
    if not os.path.isfile(p):
        sys.exit(f"FAILED: {p} not found")

# ---------------------------------------------------------------- test_full_scheme.hpp (batched) ---
tfs = open(TFS, encoding="utf-8").read()
if "SNNI full_3 fix" in tfs:
    sys.exit("FAILED: test_full_scheme already carries the full_3 fix")

# (1) CTS3 coeff_logn logn+1 -> logn
T_OLD1 = """        bs.bsgs_linear_transform(cts3[i], cts2[i], tl3, bs3_inv,
                                 logn + 1, bs.invfftcoeff3[bs.slot_index]);"""
T_NEW1 = """        bs.bsgs_linear_transform(cts3[i], cts2[i], tl3, bs3_inv,
                                 logn, bs.invfftcoeff3[bs.slot_index]);"""

# (2) single conjugate loop -> full split (coefftoslot_full_3 tail): cts3=rtn1, cts3b=rtn2
T_OLD2 = """    #pragma omp parallel for num_threads(cleanup_bootstrap_threads())
    for (int i = 0; i < count; i++) {
        Ciphertext conj;
        bs.evaluator.complex_conjugate(cts3[i], bs.gal_keys, conj);
        bs.evaluator.add_inplace_reduced_error(cts3[i], conj);
    }
    jemalloc_purge_all();
    cleanup_main_log_rss(stage_name + "_stage_cts3_post_conjugate");
    cleanup_main_log_rss(stage_name + "_stage_cts3_done");"""
T_NEW2 = """    // SNNI full_3 fix: coefftoslot_full_3 tail. cts3[i] holds sflinv_full_3 output (tmpct1).
    // rtn1 = tmpct1 + conj(tmpct1) (-> cts3[i]); rtn2 = (-i*tmpct1) + conj(-i*tmpct1) (-> cts3b[i]).
    // Conjugate key (step 0) is in every phase; phase 2 is loaded here.
    Plaintext snni_neg_i_plain;
    {
        vector<complex<double>> snni_nv(bs.Nh, complex<double>(0.0, -1.0));
        bs.encoder.encode(snni_nv, 1.0, snni_neg_i_plain);
        bs.evaluator.mod_switch_to_inplace(snni_neg_i_plain, cts3[0].parms_id());
    }
    vector<Ciphertext> cts3b(count);
    #pragma omp parallel for num_threads(cleanup_bootstrap_threads())
    for (int i = 0; i < count; i++) {
        Ciphertext snni_t2, snni_t3, snni_t4, snni_r1, snni_r2;
        bs.evaluator.multiply_plain(cts3[i], snni_neg_i_plain, snni_t2);
        bs.evaluator.complex_conjugate(snni_t2, bs.gal_keys, snni_t3);
        bs.evaluator.complex_conjugate(cts3[i], bs.gal_keys, snni_t4);
        bs.evaluator.add_reduced_error(cts3[i], snni_t4, snni_r1);
        bs.evaluator.add_reduced_error(snni_t2, snni_t3, snni_r2);
        cts3[i] = snni_r1;
        cts3b[i] = snni_r2;
    }
    jemalloc_purge_all();
    cleanup_main_log_rss(stage_name + "_stage_cts3_post_conjugate");
    cleanup_main_log_rss(stage_name + "_stage_cts3_done");"""

# (3) single modred -> double modred (rtn1, rtn2)
T_OLD3 = """    vector<Ciphertext> modrtn(count);
    cleanup_main_log_rss(stage_name + "_stage_modred_alloc_ready");
    dump_live_breakdown(stage_name + "_post_modred_alloc");
    probe_with_trim(stage_name + "_post_modred_alloc");
    #pragma omp parallel for num_threads(cleanup_bootstrap_threads())
    for (int i = 0; i < count; i++) {
        bs.mod_reducer->modular_reduction(modrtn[i], cts3[i]);
        cts3[i] = Ciphertext();
    }
    vector<Ciphertext>().swap(cts3);
    jemalloc_purge_all();
    cleanup_main_log_rss(stage_name + "_stage_modred_done");"""
T_NEW3 = """    vector<Ciphertext> modrtn(count);
    vector<Ciphertext> modrtnb(count);
    cleanup_main_log_rss(stage_name + "_stage_modred_alloc_ready");
    dump_live_breakdown(stage_name + "_post_modred_alloc");
    probe_with_trim(stage_name + "_post_modred_alloc");
    #pragma omp parallel for num_threads(cleanup_bootstrap_threads())
    for (int i = 0; i < count; i++) {
        bs.mod_reducer->modular_reduction(modrtn[i], cts3[i]);
        cts3[i] = Ciphertext();
        bs.mod_reducer->modular_reduction(modrtnb[i], cts3b[i]);
        cts3b[i] = Ciphertext();
    }
    vector<Ciphertext>().swap(cts3);
    vector<Ciphertext>().swap(cts3b);
    jemalloc_purge_all();
    cleanup_main_log_rss(stage_name + "_stage_modred_done");"""

# (4) STC1 on modrtn -> recombine (slottocoeff_full_3 head) then STC1
T_OLD4 = """    vector<Ciphertext> stc1(count);
    #pragma omp parallel for num_threads(cleanup_bootstrap_threads())
    for (int i = 0; i < count; i++) {
        bs.bsgs_linear_transform(stc1[i], modrtn[i], tl1f, bs1_fwd, logn, bs.fftcoeff1[bs.slot_index]);
        bs.evaluator.rescale_to_next_inplace(stc1[i]);
        modrtn[i] = Ciphertext();
    }
    vector<Ciphertext>().swap(modrtn);
    jemalloc_purge_all();
    cleanup_main_log_rss(stage_name + "_stage_stc1_done");"""
T_NEW4 = """    // SNNI full_3 fix: slottocoeff_full_3 head. tmpct3 = modrtn1 + i*modrtn2, then sfl_full_3 (STC1).
    Plaintext snni_pos_i_plain;
    {
        vector<complex<double>> snni_pv(bs.Nh, complex<double>(0.0, 1.0));
        bs.encoder.encode(snni_pv, 1.0, snni_pos_i_plain);
        bs.evaluator.mod_switch_to_inplace(snni_pos_i_plain, modrtnb[0].parms_id());
    }
    vector<Ciphertext> stc1(count);
    #pragma omp parallel for num_threads(cleanup_bootstrap_threads())
    for (int i = 0; i < count; i++) {
        Ciphertext snni_pt1, snni_pt3;
        bs.evaluator.multiply_plain(modrtnb[i], snni_pos_i_plain, snni_pt1);
        bs.evaluator.add_reduced_error(modrtn[i], snni_pt1, snni_pt3);
        modrtn[i] = Ciphertext();
        modrtnb[i] = Ciphertext();
        bs.bsgs_linear_transform(stc1[i], snni_pt3, tl1f, bs1_fwd, logn, bs.fftcoeff1[bs.slot_index]);
        bs.evaluator.rescale_to_next_inplace(stc1[i]);
    }
    vector<Ciphertext>().swap(modrtn);
    vector<Ciphertext>().swap(modrtnb);
    jemalloc_purge_all();
    cleanup_main_log_rss(stage_name + "_stage_stc1_done");"""

for i, (o, n) in enumerate([(T_OLD1, T_NEW1), (T_OLD2, T_NEW2), (T_OLD3, T_NEW3), (T_OLD4, T_NEW4)], 1):
    if tfs.count(o) != 1:
        sys.exit(f"FAILED: tfs block {i} anchor count = {tfs.count(o)} (expected 1)")
    tfs = tfs.replace(o, n, 1)

# --------------------------------------------------------------------------- softmax.hpp (single) ---
sm = open(SM, encoding="utf-8").read()
if "SNNI full_3 fix" in sm:
    sys.exit("FAILED: softmax already carries the full_3 fix")

S_OLD1 = """  bs.bsgs_linear_transform(cts3, cts2, tl3, bs3_inv, logn + 1, bs.invfftcoeff3[bs.slot_index]);"""
S_NEW1 = """  bs.bsgs_linear_transform(cts3, cts2, tl3, bs3_inv, logn, bs.invfftcoeff3[bs.slot_index]);"""

S_OLD2 = """  cts2 = Ciphertext();
  Ciphertext conj;
  bs.evaluator.complex_conjugate(cts3, bs.gal_keys, conj);
  bs.evaluator.add_inplace_reduced_error(cts3, conj);
  cleanup_softmax_log_rss("softmax_split_cts3_done");

  Ciphertext modrtn;
  bs.mod_reducer->modular_reduction(modrtn, cts3);
  cts3 = Ciphertext();
  cleanup_softmax_log_rss("softmax_split_modred_done");

  Ciphertext stc1;
  bs.bsgs_linear_transform(stc1, modrtn, tl1f, bs1_fwd, logn, bs.fftcoeff1[bs.slot_index]);
  bs.evaluator.rescale_to_next_inplace(stc1);
  modrtn = Ciphertext();
  cleanup_softmax_log_rss("softmax_split_stc1_done");"""
S_NEW2 = """  cts2 = Ciphertext();
  // SNNI full_3 fix: coefftoslot_full_3 tail (single-CT). cts3 = sflinv_full_3 output (tmpct1).
  // rtn1 = tmpct1+conj(tmpct1) (-> cts3); rtn2 = (-i*tmpct1)+conj(-i*tmpct1) (-> cts3b).
  Plaintext snni_neg_i_plain;
  {
    vector<complex<double>> snni_nv(bs.Nh, complex<double>(0.0, -1.0));
    bs.encoder.encode(snni_nv, 1.0, snni_neg_i_plain);
    bs.evaluator.mod_switch_to_inplace(snni_neg_i_plain, cts3.parms_id());
  }
  Ciphertext cts3b, snni_t2, snni_t3, snni_t4, snni_r1, snni_r2;
  bs.evaluator.multiply_plain(cts3, snni_neg_i_plain, snni_t2);
  bs.evaluator.complex_conjugate(snni_t2, bs.gal_keys, snni_t3);
  bs.evaluator.complex_conjugate(cts3, bs.gal_keys, snni_t4);
  bs.evaluator.add_reduced_error(cts3, snni_t4, snni_r1);
  bs.evaluator.add_reduced_error(snni_t2, snni_t3, snni_r2);
  cts3 = snni_r1;
  cts3b = snni_r2;
  cleanup_softmax_log_rss("softmax_split_cts3_done");

  Ciphertext modrtn, modrtnb;
  bs.mod_reducer->modular_reduction(modrtn, cts3);
  cts3 = Ciphertext();
  bs.mod_reducer->modular_reduction(modrtnb, cts3b);
  cts3b = Ciphertext();
  cleanup_softmax_log_rss("softmax_split_modred_done");

  // SNNI full_3 fix: slottocoeff_full_3 head. tmpct3 = modrtn1 + i*modrtn2 -> STC1.
  Plaintext snni_pos_i_plain;
  {
    vector<complex<double>> snni_pv(bs.Nh, complex<double>(0.0, 1.0));
    bs.encoder.encode(snni_pv, 1.0, snni_pos_i_plain);
    bs.evaluator.mod_switch_to_inplace(snni_pos_i_plain, modrtnb.parms_id());
  }
  Ciphertext snni_pt1, snni_pt3;
  bs.evaluator.multiply_plain(modrtnb, snni_pos_i_plain, snni_pt1);
  bs.evaluator.add_reduced_error(modrtn, snni_pt1, snni_pt3);
  modrtn = Ciphertext();
  modrtnb = Ciphertext();
  Ciphertext stc1;
  bs.bsgs_linear_transform(stc1, snni_pt3, tl1f, bs1_fwd, logn, bs.fftcoeff1[bs.slot_index]);
  bs.evaluator.rescale_to_next_inplace(stc1);
  cleanup_softmax_log_rss("softmax_split_stc1_done");"""

for i, (o, n) in enumerate([(S_OLD1, S_NEW1), (S_OLD2, S_NEW2)], 1):
    if sm.count(o) != 1:
        sys.exit(f"FAILED: softmax block {i} anchor count = {sm.count(o)} (expected 1)")
    sm = sm.replace(o, n, 1)

# ---------------------------------------------------------------------------------------- verify ---
if tfs.count("logn + 1") != 0:
    sys.exit(f"FAILED: 'logn + 1' still present in tfs ({tfs.count('logn + 1')})")
if sm.count("logn + 1") != 0:
    sys.exit(f"FAILED: 'logn + 1' still present in softmax ({sm.count('logn + 1')})")
if tfs.count("SNNI full_3 fix") != 2 or sm.count("SNNI full_3 fix") != 2:
    sys.exit("FAILED: fix markers count wrong")
if "add_inplace_reduced_error" in tfs or "add_inplace_reduced_error" in sm:
    sys.exit("FAILED: single-path add_inplace_reduced_error still present")

open(TFS, "w", encoding="utf-8").write(tfs)
open(SM, "w", encoding="utf-8").write(sm)
print("patch_ps_full3_fix: full_3 bootstrap applied to test_full_scheme.hpp + softmax.hpp")
