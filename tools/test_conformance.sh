#!/bin/bash
# Conformance test for make_steps.py: synthetic lines with known verdicts; fails if a check passes silently.
#
# Block 1: wrong-pool line (SHAFT-GPU numbers discarded by the 2026-07-26 audit; s1 attacked VRAM
# while host had the larger W). Expected exit 1, s1 conformant, s2 flagged.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

cat > "$T/MANIFEST.md" <<'EOF'
target_vram_kb: 192512
target_host_kb: 192512
overhead_vram_kb: 0
overhead_host_kb: 0
EOF

mk() {  # step_id vram_MB host_MB declared_pool
    d="$T/steps/$1"; mkdir -p "$d/logs/n1"
    echo "GPUPEAK|party=0|peak_alloc_kb=0|peak_reserved_kb=$(( $2 * 1024 ))" > "$d/logs/n1/all_markers.txt"
    printf '# ts rss vmpeak\n1 %d 0\n' $(( $3 * 1024 )) > "$d/logs/n1/vmrss.log"
    echo "rc=0" > "$d/logs/n1/PEAK.txt"
    echo 50 > "$d/logs/wall_seconds.txt"
    printf '# pool_attacked: %s\nLEVER=%s\n' "$4" "$1" > "$d/levers.env"
}
mk s0_default  7139 2904 ""
mk s1_embed    1977 3428 vram
mk s2_per_layer 817 1760 vram

out=$(python3 tools/make_steps.py "$T" 2>&1); rc=$?
echo "$out"

fail=0
[ "$rc" -eq 0 ] && { echo "FAIL: a non-conformant waterfall built successfully"; fail=1; }
grep -q "NON-CONFORMANT POOL s2_per_layer" <<<"$out" || { echo "FAIL: s2 not flagged"; fail=1; }
grep -q "NON-CONFORMANT POOL s1_embed"  <<<"$out" && { echo "FAIL: s1 flagged, it is conformant"; fail=1; }
# Checked by column NAME, not position, so a schema addition does not break this test.
python3 - "$T/steps.csv" <<'PY' || fail=1
import csv, sys
want = {"s1_embed": ("vram", "vram", "yes"), "s2_per_layer": ("vram", "host", "no")}
rows = {r["step_id"]: r for r in csv.DictReader(open(sys.argv[1]))}
ok = True
for sid, (att, req, conf) in want.items():
    r = rows.get(sid)
    if not r:
        print(f"FAIL: no row {sid}"); ok = False; continue
    got = (r["pool_attacked"], r["pool_required"], r["conformant"])
    if got != (att, req, conf):
        print(f"FAIL: {sid} is {got}, expected {(att, req, conf)}"); ok = False
sys.exit(0 if ok else 1)
PY

[ "$fail" -eq 0 ] && echo "PASS: the check catches the ordering the audit rejected"

# ---------------------------------------------------------------------------
# Block 2: does a step that did not move the peak get rejected, for the right reason? (The guard
# once ran only for a negative delta, so SHARK's s6_down_split at +644 kB vs 0.168% spread was
# published.) Tested both ways: n1_real (genuine drop) conformant, n2_inert and n3_worse rejected as
# PEAK DID NOT MOVE, not as a POOL violation (both set conformant=no, but the reason must be right).
T2=$(mktemp -d)
trap 'rm -rf "$T" "$T2"' EXIT

cat > "$T2/MANIFEST.md" <<'MANEOF'
target_host_kb: 100000
overhead_host_kb: 0
replicates: 3
MANEOF

mk3() {  # step_id host_MB_run1 host_MB_run2 host_MB_run3 declared_pool
    d="$T2/steps/$1"; i=0
    for mb in "$2" "$3" "$4"; do
        i=$((i+1)); mkdir -p "$d/logs/run$i"
        printf '# ts rss vmpeak\n1 %d 0\n' $(( mb * 1024 )) > "$d/logs/run$i/vmrss.log"
        echo "rc=0" > "$d/logs/run$i/PEAK.txt"
        echo 50 > "$d/logs/run$i/wall_seconds.txt"
    done
    printf '# pool_attacked: %s\nLEVER=%s\n' "$5" "$1" > "$d/levers.env"
}
# ~1% spread inside each step, so a 0.2% move is noise and a 30% move is not.
mk3 n0_default 1000 1005 1010 ""
mk3 n1_real     700  703  707 host
mk3 n2_inert    701  704  708 host
mk3 n3_worse    760  763  767 host

out2=$(python3 tools/make_steps.py "$T2" 2>&1); rc2=$?
echo "$out2"

fail2=0
[ "$rc2" -eq 0 ] && { echo "FAIL: a line with an inert step built successfully"; fail2=1; }
grep -q "PEAK DID NOT MOVE n2_inert" <<<"$out2" || { echo "FAIL: n2_inert not flagged as inert"; fail2=1; }
grep -q "PEAK DID NOT MOVE n3_worse" <<<"$out2" || { echo "FAIL: n3_worse not flagged"; fail2=1; }
grep -qE "(PEAK DID NOT MOVE|NON-CONFORMANT POOL) n1_real" <<<"$out2" && { echo "FAIL: n1_real flagged, it is a real reduction"; fail2=1; }
grep -q "NON-CONFORMANT POOL n2_inert" <<<"$out2" && { echo "FAIL: n2_inert reported as a POOL violation, which it is not"; fail2=1; }

python3 - "$T2/steps.csv" <<'CSVEOF' || fail2=1
import csv, sys
want = {"n1_real": "yes", "n2_inert": "no", "n3_worse": "no"}
rows = {r["step_id"]: r for r in csv.DictReader(open(sys.argv[1]))}
ok = True
for sid, conf in want.items():
    r = rows.get(sid)
    if not r:
        print(f"FAIL: no row {sid}"); ok = False; continue
    if r["conformant"] != conf:
        print(f"FAIL: {sid} conformant={r['conformant']}, expected {conf}"); ok = False
sys.exit(0 if ok else 1)
CSVEOF

[ "$fail2" -eq 0 ] && echo "PASS: a step that does not move the peak is rejected, and for the stated reason"
[ "$fail2" -ne 0 ] && fail=1


# ---------------------------------------------------------------------------
# Block 3: does the T2 gate discriminate? Tested both ways: within (jitter inside the tolerance ->
# builds), beyond (one replicate outside -> NONDETERMINISTIC), changed (output moved -> GATE FAILED),
# notol (undeclared tolerance -> refused). Tolerance is ABSOLUTE, the case relative cannot express
# (one value is near zero, large relative difference but small absolute).
G=$(mktemp -d)
trap 'rm -rf "$T" "$G"' EXIT

gate_sys() {  # dir tolerance_line
    mkdir -p "$1"
    cat > "$1/MANIFEST.md" <<EOF
target_host_kb: 1000
overhead_host_kb: 0
gate_tier: T2
$2
EOF
}

gstep() {  # sysdir step_id "v1 v2 v3" "v1 v2 v3" "v1 v2 v3"   (one per replicate)
    d="$1/steps/$2"; shift 2
    i=1
    for vals in "$@"; do
        mkdir -p "$d/logs/run$i"
        printf '# ts rss vmpeak\n1 2048 0\n' > "$d/logs/run$i/vmrss.log"
        echo "rc=0" > "$d/logs/run$i/PEAK.txt"
        echo 50 > "$d/logs/run$i/wall_seconds.txt"
        tr ' ' '\n' <<<"$vals" > "$d/logs/run$i/gate.txt"
        i=$((i + 1))
    done
    printf '# pool_attacked: host\nLEVER=x\n' > "$d/levers.env"
}

# The 0.0005 element is the near-zero case: an absolute tolerance must not care that its relative
# difference is large.
gate_sys "$G/within" "gate_tolerance_abs: 7.3e-3"
gstep "$G/within" b0 "15.200000 0.000500 74.200000" \
                     "15.203000 0.001500 74.203000" \
                     "15.197000 0.002500 74.197000"
out=$(python3 tools/make_steps.py "$G/within" 2>&1); rc=$?
grep -q "NO GATE" <<<"$out" && { echo "FAIL: T2 replicates inside the tolerance were called NONDETERMINISTIC"; fail=1; }
[ "$rc" -ne 0 ] && { echo "FAIL: a T2 line whose replicates agree within tolerance did not build"; echo "$out"; fail=1; }

# Pairwise case: run2 and run3 each agree with run1 within tolerance but are 0.008 apart, past it.
# Comparing each against run1 would accept it; comparing every pair rejects it.
gate_sys "$G/beyond" "gate_tolerance_abs: 7.3e-3"
gstep "$G/beyond" b0 "15.200000 0.000500 74.200000" \
                     "15.204000 0.001500 74.203000" \
                     "15.196000 0.002500 74.197000"
out=$(python3 tools/make_steps.py "$G/beyond" 2>&1); rc=$?
[ "$rc" -eq 0 ] && { echo "FAIL: a T2 line whose replicates disagree beyond tolerance built successfully"; fail=1; }
grep -q "NO GATE b0" <<<"$out" || { echo "FAIL: replicates beyond tolerance were not reported as NO GATE"; fail=1; }

# A step whose own replicates are consistent but whose output has moved away from the baseline.
gate_sys "$G/changed" "gate_tolerance_abs: 7.3e-3"
gstep "$G/changed" b0 "15.200000 0.000500 74.200000" \
                      "15.203000 0.001500 74.203000" \
                      "15.197000 0.002500 74.197000"
gstep "$G/changed" b1 "15.900000 0.000500 74.200000" \
                      "15.903000 0.001500 74.203000" \
                      "15.897000 0.002500 74.197000"
out=$(python3 tools/make_steps.py "$G/changed" 2>&1); rc=$?
[ "$rc" -eq 0 ] && { echo "FAIL: a step whose output moved past the tolerance built successfully"; fail=1; }
grep -q "GATE FAILED b1" <<<"$out" || { echo "FAIL: the changed step was not flagged"; echo "$out"; fail=1; }

# An undeclared tolerance is refused rather than treated leniently: a gate whose tolerance is
# whatever the step needed is not a gate.
gate_sys "$G/notol" ""
gstep "$G/notol" b0 "15.200000 0.000500 74.200000" \
                    "15.203000 0.001500 74.203000" \
                    "15.197000 0.002500 74.197000"
out=$(python3 tools/make_steps.py "$G/notol" 2>&1); rc=$?
[ "$rc" -eq 0 ] && { echo "FAIL: T2 with no declared tolerance built successfully"; fail=1; }

[ "$fail" -eq 0 ] && echo "PASS: the T2 gate accepts jitter inside its tolerance and rejects everything else"

# ---------------------------------------------------------------------------
# Block 4: a DECLARED pool skip builds and says so (block 1 already proved the undeclared divergence
# fails). Replays block 1's numbers with `# pool_skipped: host` and requires conformant=yes-skipped
# plus a "skipped" note, since a skip that reads like an ordinary pass would hide what it exposes.
P=$(mktemp -d)
trap 'rm -rf "$T" "$T2" "$G" "$P"' EXIT
cp "$T/MANIFEST.md" "$P/MANIFEST.md"
pmk() {  # step_id vram_MB host_MB declared_pool extra_lines
    d="$P/steps/$1"; mkdir -p "$d/logs/n1"
    echo "GPUPEAK|party=0|peak_alloc_kb=0|peak_reserved_kb=$(( $2 * 1024 ))" > "$d/logs/n1/all_markers.txt"
    printf '# ts rss vmpeak\n1 %d 0\n' $(( $3 * 1024 )) > "$d/logs/n1/vmrss.log"
    echo "rc=0" > "$d/logs/n1/PEAK.txt"
    echo 50 > "$d/logs/wall_seconds.txt"
    { printf '# pool_attacked: %s\n' "$4"; [ -n "${5:-}" ] && printf '%s\n' "$5"; \
      printf 'LEVER=%s\n' "$1"; } > "$d/levers.env"
}
pmk s0_default  7139 2904 ""
pmk s1_embed    1977 3428 vram
pmk s2_per_layer 817 1760 vram "# pool_skipped: host"

out=$(python3 tools/make_steps.py "$P" 2>&1); rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: a DECLARED pool skip did not build"; echo "$out"; fail=1; }
python3 - "$P/steps.csv" <<'PY' || fail=1
import csv, sys
r = {x["step_id"]: x for x in csv.DictReader(open(sys.argv[1]))}.get("s2_per_layer")
if r is None:
    print("FAIL: no row s2_per_layer"); sys.exit(1)
if r["conformant"] != "yes-skipped":
    print(f"FAIL: declared pool skip reads conformant={r['conformant']}, expected yes-skipped")
    sys.exit(1)
if "skipped" not in r["evidence"]:
    print("FAIL: the row does not say the pool was skipped, so a reader cannot see it")
    sys.exit(1)
sys.exit(0)
PY
[ "$fail" -eq 0 ] && echo "PASS: a declared pool skip builds and says so, an undeclared one fails"

# ---------------------------------------------------------------------------
# Block 5: phase B (a step whose output moved but whose protocol work is unchanged). All four ways:
# ok (same bytes + plaintext test -> gate reads equivalent), bytes (MORE bytes -> refused, BumbleBee
# x7), notest (no plaintext test -> refused), order (an exact step after an equivalent one -> refused).
E=$(mktemp -d)
trap 'rm -rf "$T" "$T2" "$G" "$P" "$E"' EXIT

estep() {  # sysdir step_id rss gate_values comm_bytes comm_rounds evidence_line plaintext_file
    d="$1/steps/$2"
    for i in 1 2 3; do
        mkdir -p "$d/logs/run$i"
        printf '# ts rss vmpeak\n1 %d 0\n' "$3" > "$d/logs/run$i/vmrss.log"
        echo "rc=0" > "$d/logs/run$i/PEAK.txt"
        echo 50 > "$d/logs/run$i/wall_seconds.txt"
        tr ' ' '\n' <<<"$4" > "$d/logs/run$i/gate.txt"
        printf 'TRANSCRIPT|party=0|bytes=%s|rounds=%s\n' "$5" "$6" > "$d/logs/run$i/stdout.log"
    done
    { printf '# pool_attacked: host\n'; [ -n "${7:-}" ] && printf '%s\n' "$7"; \
      printf 'LEVER=%s\n' "$2"; } > "$d/levers.env"
    [ -n "${8:-}" ] && { mkdir -p "$d/$(dirname "$8")"; echo "equivalence test" > "$d/$8"; }
}

# The accepting case.
gate_sys "$E/ok" "gate_tolerance: 1e-6"
estep "$E/ok" b0 2048 "1.0 2.0" 100000 1496 "" ""
estep "$E/ok" b1 1024 "9.9 8.8" 100000 1526 "# evidence: equivalent
# plaintext_equivalence: impl/equiv_test.py" "impl/equiv_test.py"
out=$(python3 tools/make_steps.py "$E/ok" 2>&1); rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: a declared phase B step with the same bytes did not build"; echo "$out"; fail=1; }
python3 - "$E/ok/steps.csv" <<'PY' || fail=1
import csv, sys
r = {x["step_id"]: x for x in csv.DictReader(open(sys.argv[1]))}.get("b1")
if r is None or r["correctness_gate"] != "equivalent":
    print(f"FAIL: phase B step reads gate={r and r['correctness_gate']}, expected equivalent"); sys.exit(1)
if "same" not in r["evidence"] or "NOT that the output is identical" not in r["evidence"]:
    print("FAIL: the row does not state what phase B claims and does not claim"); sys.exit(1)
sys.exit(0)
PY

# More bytes: other protocol work. This is the shape that retracted BumbleBee's x7.
gate_sys "$E/bytes" "gate_tolerance: 1e-6"
estep "$E/bytes" b0 2048 "1.0 2.0" 100000 1496 "" ""
estep "$E/bytes" b1 1024 "9.9 8.8" 141000 1526 "# evidence: equivalent
# plaintext_equivalence: impl/equiv_test.py" "impl/equiv_test.py"
out=$(python3 tools/make_steps.py "$E/bytes" 2>&1); rc=$?
[ "$rc" -eq 0 ] && { echo "FAIL: phase B accepted a step that exchanged MORE bytes"; fail=1; }
grep -q "other protocol work" <<<"$out" || { echo "FAIL: the byte difference was not the stated reason"; echo "$out"; fail=1; }

# No plaintext equivalence test: half the evidence is not the evidence.
gate_sys "$E/notest" "gate_tolerance: 1e-6"
estep "$E/notest" b0 2048 "1.0 2.0" 100000 1496 "" ""
estep "$E/notest" b1 1024 "9.9 8.8" 100000 1526 "# evidence: equivalent" ""
out=$(python3 tools/make_steps.py "$E/notest" 2>&1); rc=$?
[ "$rc" -eq 0 ] && { echo "FAIL: phase B accepted a step with no plaintext equivalence test"; fail=1; }
grep -q "plaintext_equivalence" <<<"$out" || { echo "FAIL: the missing test was not the stated reason"; echo "$out"; fail=1; }

# Ordering: exact evidence is exhausted before equivalence evidence is admitted.
gate_sys "$E/order" "gate_tolerance: 1e-6"
estep "$E/order" b0 2048 "1.0 2.0" 100000 1496 "" ""
estep "$E/order" b1 1024 "9.9 8.8" 100000 1526 "# evidence: equivalent
# plaintext_equivalence: impl/equiv_test.py" "impl/equiv_test.py"
estep "$E/order" b2 512 "1.0 2.0" 100000 1526 "" ""
out=$(python3 tools/make_steps.py "$E/order" 2>&1); rc=$?
[ "$rc" -eq 0 ] && { echo "FAIL: an exact step after an equivalent one built successfully"; fail=1; }
grep -q "EXACT AFTER EQUIVALENT b2" <<<"$out" || { echo "FAIL: the inverted phase ordering was not flagged"; echo "$out"; fail=1; }

[ "$fail" -eq 0 ] && echo "PASS: phase B needs both measurements, rejects extra traffic, and is ordered last"

# ------------------------------------------------------------------------------------------------
# Block 6: with replicates:1 (no wall spread) a step must not publish a blank cost class or a 0.0%
# spread. A declared class is used (s1_decl free), an undeclared one is named undetermined (not left
# blank, which rule 6 would skip), spreads are "" not 0.0%, and a rising peak is still caught.
R=$(mktemp -d)
cat > "$R/MANIFEST.md" <<'EOF'
target_host_kb: 100000
overhead_host_kb: 0
EOF
r1() {  # step host_MB wall_s extra_levers_line
    d="$R/steps/$1"; mkdir -p "$d/logs/run1"
    printf '# ts rss vmpeak\n1 %d 0\n' $(( $2 * 1024 )) > "$d/logs/run1/vmrss.log"
    echo "rc=0" > "$d/logs/run1/PEAK.txt"
    echo "$3" > "$d/logs/run1/wall_seconds.txt"
    printf '# pool_attacked: host\n%s\nLEVER=%s\n' "$4" "$1" > "$d/levers.env"
}
r1 s0_default 4000 100 ""
r1 s1_decl    3000 101 "# cost_class: free"
r1 s2_blank   2000 130 ""
r1 s3_worse   2100 131 ""
out=$(python3 tools/make_steps.py "$R" 2>&1)
rfail=0
python3 - "$R/steps.csv" <<'PYINNER' || rfail=1
import csv, sys
rows = {r["step_id"]: r for r in csv.DictReader(open(sys.argv[1]))}
ok = True


def want(sid, field, value):
    global ok
    got = rows.get(sid, {}).get(field)
    if got != value:
        print("FAIL: %s %s is %r, expected %r" % (sid, field, got, value))
        ok = False


# a single run establishes no spread, on either axis
for sid in rows:
    want(sid, "spread_host_pct", "")
    want(sid, "wall_spread_pct", "")
# a declared class is used; an undeclared one is NAMED rather than left blank
want("s1_decl", "cost_type", "free")
want("s2_blank", "cost_type", "undetermined")
# and a rising peak is still caught with no spread to judge it against
if "ROSE" not in rows.get("s3_worse", {}).get("evidence", ""):
    print("FAIL: a rising peak was not flagged without a spread")
    ok = False
sys.exit(0 if ok else 1)
PYINNER
grep -q "cost class 'free' is DECLARED" "$R/steps.csv" \
    || { echo "FAIL: a declared class was not marked as declared in the row"; rfail=1; }
rm -rf "$R"
[ "$rfail" -eq 0 ] && echo "PASS: one replicate names its cost class or calls it undetermined, and never reports 0.0% spread"
[ "$rfail" -ne 0 ] && fail=1

# ------------------------------------------------------------------------------------------------
# Block 7: multi-party runs publish the maximum over the parties, and the median over runs of that
# maximum. p1 has the non-leader heavier in two runs and one table per party (its own is used);
# p2 has the non-leader heavier in two runs and only the leader's table, so the row keeps the table of
# the median run by the leader's peak and declares decomposed_party=leader.
P=$(mktemp -d)
cat > "$P/MANIFEST.md" <<'EOF'
target_host_kb: 100000
overhead_host_kb: 0
EOF
prun() {  # step run leader_MB other_MB own_table(0|1)
    d="$P/steps/$1/logs/$2"; mkdir -p "$d/poll_100" "$d/poll_200"
    printf '# ts rss vmpeak\n1 %d 0\n' $(( $3 * 1024 )) > "$d/poll_100/vmrss.log"
    printf '# ts rss vmpeak\n1 %d 0\n' $(( $4 * 1024 )) > "$d/poll_200/vmrss.log"
    printf 'poller on pid 100 (leader=1) -> /x/poll_100\npoller on pid 200 (leader=0) -> /x/poll_200\n' \
        > "$d/poller_attach.log"
    echo "rc=0" > "$d/PEAK.txt"; echo 100 > "$d/wall_seconds.txt"
    printf '{"objects": {"heap:a.c:1:leader_obj": %d, "instrument:pmtable_100": 0}}\n' \
        $(( $3 * 1024 * 512 )) > "$d/objects_host.jsonl"
    [ "$5" -eq 1 ] && printf '{"objects": {"heap:a.c:1:other_obj": %d, "instrument:pmtable_200": 0}}\n' \
        $(( $4 * 1024 * 512 )) > "$d/objects_host_200.jsonl"
    printf '# pool_attacked: host\nLEVER=%s\n' "$1" > "$P/steps/$1/levers.env"
}
prun p0_default run1 4000 3900 1; prun p0_default run2 4010 3905 1; prun p0_default run3 4020 3910 1
prun p1_own     run1 3000 3100 1; prun p1_own     run2 3010 2990 1; prun p1_own     run3 3020 3150 1
prun p2_leader  run1 2000 2150 0; prun p2_leader  run2 2030 2100 0; prun p2_leader  run3 2010 1990 0
python3 tools/make_steps.py "$P" >/dev/null 2>&1
python3 - "$P/steps.csv" <<'PYP' || fail=1
import csv, sys
rows = {r["step_id"]: r for r in csv.DictReader(open(sys.argv[1]))}
bad = []
def want(sid, field, value):
    got = rows.get(sid, {}).get(field)
    if got != value:
        bad.append("%s %s is %r, expected %r" % (sid, field, got, value))
# per-run maxima 4000/4010/4020 -> median 4010 MB, the leader's
want("p0_default", "peak_host", str(4010 * 1024)); want("p0_default", "peak_party", "leader")
# per-run maxima 3100/3010/3150 -> median 3100 MB, the non-leader's, with its own table
want("p1_own", "peak_host", str(3100 * 1024)); want("p1_own", "peak_party", "non-leader")
want("p1_own", "decomposed_party", "non-leader")
want("p1_own", "object_targeted", "heap:other_obj")
# per-run maxima 2150/2100/2010 -> median 2100 MB (run2, non-leader); only the leader has a table,
# and the median run by the leader's peak (2000/2030/2010) is run3
want("p2_leader", "peak_host", str(2100 * 1024)); want("p2_leader", "peak_party", "non-leader")
want("p2_leader", "decomposed_party", "leader")
ev = rows.get("p2_leader", {}).get("evidence", "")
if "PARTY MISMATCH" not in ev or "run3" not in ev:
    bad.append("p2_leader does not declare the party mismatch and the run of its table")
for b in bad:
    print("FAIL: " + b)
if not bad:
    print("PASS: multi-party runs publish the median of per-run maxima and declare whose table decomposes it")
sys.exit(1 if bad else 0)
PYP
rm -rf "$P"

exit "$fail"
