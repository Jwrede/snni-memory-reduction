#!/bin/bash
# Is this system ready to be measured? Run before the first step and after any harness change.
# Exit 1 on any failure. What cannot be machine-checked is listed at the end as the manual gate.
#
#   preflight.sh <system> [remote]        remote defaults to `palma`
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

SYS="${1:?usage: preflight.sh <system> [remote]}"
REMOTE="${2:-palma}"
FAIL=0
ok()   { printf '  ok    %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; FAIL=$((FAIL+1)); }
warn() { printf '  warn  %s\n' "$1"; }

echo "preflight: $SYS"

# --- 1. the harness knows this system -----------------------------------------------------------
CONF="tools/palma/conf/$SYS.conf"
if [ -r "$CONF" ]; then
    ok "conf exists: $CONF"
    # shellcheck disable=SC1090
    ( . tools/palma/campaign.env && . "$CONF" && \
      [ -n "${WORKDIR:-}" ] && [ -n "${SBATCH:-}" ] && [ -n "${DERIVE:-}" ] && [ -n "${CHANNELS:-}" ] ) \
        && ok "conf declares WORKDIR, SBATCH, DERIVE, CHANNELS" \
        || bad "conf is missing one of WORKDIR, SBATCH, DERIVE, CHANNELS"
else
    bad "no conf at $CONF (copy tools/palma/conf/llama-dealer.conf and edit)"
fi

# --- 2. no system carries its own copy of the measurement policy ---------------------------------
if bash tools/check_policy.sh "systems/$SYS" >/dev/null 2>&1; then
    ok "no hardcoded measurement policy under systems/$SYS"
else
    bad "hardcoded measurement policy: run tools/check_policy.sh systems/$SYS"
fi

# --- 3. the MANIFEST pins what the procedure requires BEFORE the first run -----------------------
M="systems/$SYS/MANIFEST.md"
if [ -r "$M" ]; then
    ok "MANIFEST exists"
    grep -qE '^\s*target_(host|vram)_kb:\s*[0-9]+' "$M" \
        && ok "a protocol target is pinned" \
        || bad "no target_<pool>_kb in MANIFEST: W is not computable, so the pool choice cannot be verified"
    if grep -qE 'gate_tier:\s*T[0-3]' "$M"; then
        if grep -qE 'gate_tier:\s*T0' "$M"; then
            # T0 is a legitimate declaration but an illegitimate state to publish from.
            bad "gate_tier is T0: declared, which is correct, but this system has no output check,"
            printf '        so no step of it is shown to preserve semantics and none may be published.\n'
        else
            ok "gate tier declared"
        fi
    else
        bad "no gate_tier in MANIFEST: T0 is allowed but must be DECLARED, never silently absent"
    fi
    # A placeholder is not a declaration: `seed: TODO` is worse than an absent line (reads as settled).
    if grep -qE '^\s*seed:' "$M"; then
        if grep -qiE '^\s*seed:\s*(unknown|todo|tbd|none|\?+)' "$M"; then
            bad "seed is a placeholder, not a value: pin it and verify with two unchanged runs"
        else
            ok "seed declared"
        fi
    else
        bad "no seed in MANIFEST: see README.md, a gate cannot exist without determinism"
    fi
    if grep -qE 'gate_observable:' "$M"; then
        if grep -qiE 'gate_observable:\s*(none|todo|tbd|unknown)' "$M"; then
            bad "gate_observable is a placeholder: name what a step must not change, or state T0 and stop"
        else
            ok "gate observable named"
        fi
    else
        bad "no gate_observable in MANIFEST"
    fi
    # A T2 system declares EXACTLY ONE tolerance: gate_tolerance (relative) or gate_tolerance_abs
    # (absolute, the unit for a fixed-point ULP that does not scale). Both, or neither, is refused.
    if grep -qE 'gate_tier:\s*T2' "$M"; then
        has_rel=0; has_abs=0
        # Anchored to line start, so a MANIFEST discussing a retired tolerance in prose does not read
        # as a declaration (ARION's prose mentions gate_tolerance: 1e-6 while it declares only _abs).
        grep -qE '^[[:space:]]*gate_tolerance:' "$M" && has_rel=1
        grep -qE '^[[:space:]]*gate_tolerance_abs:' "$M" && has_abs=1
        if [ $((has_rel + has_abs)) -eq 0 ]; then
            bad "T2 without a tolerance: an undeclared tolerance is compared exactly, which is not what T2 means"
        elif [ $((has_rel + has_abs)) -eq 2 ]; then
            bad "T2 with BOTH gate_tolerance and gate_tolerance_abs: declare exactly one"
        else
            ok "T2 tolerance declared"
        fi
    fi
    if grep -qE '^\s*replicates:\s*[0-2]\b' "$M"; then
        # The reason may not sit on the same line as the word (SIGMA-GPU argues queue exposure over
        # four lines), so search a window after any mention with cost-argument vocabulary.
        grep -iE -A8 'replicates' "$M" \
            | grep -qiE '(because|reason|h per run|hours|per run|queue|slot|cost|given up|argument)' \
            && ok "replicates below 3, with a reason" \
            || bad "replicates below 3 without a stated reason in the MANIFEST"
    fi
else
    bad "no $M (targets, gate and seed must be pinned BEFORE the first run, not after)"
fi

# --- 4. off-path.md exists, even empty ----------------------------------------------------------
[ -r "systems/$SYS/off-path.md" ] \
    && ok "off-path.md exists" \
    || bad "no off-path.md: 'no accuracy/security/portability trades' is a result, not an omission"

# --- 5. determinism, per README.md ---------------------------------------------------------------
# READ THE STATUS COLUMN, NOT THE WHOLE ROW. This was `grep "^\| $SYS .*verified"`, and the
# substring is in the NEGATION too: `preflight.sh llama` printed "ok README.md records determinism
# as verified" while llama's row says, in full, "pinned, not yet verified". A check that passes on
# the words of its own failure is the failure shape this campaign keeps meeting -- not an error, a
# plausible pass -- and here it sat inside the preflight that exists to catch exactly that.
# The status is the register's BOLD field and it must OPEN with the word: `**verified ...**`.
# Splitting the row on pipes does not work and quietly gives the wrong column -- several rows
# carry pipes of their own, escaped (`SEEDED\|`) or literal (`|e| <= 1`) -- while "pinned, not yet
# verified" opens its bold with "pinned". So the pattern is anchored to the bold marker, which is
# how every verified row in README.md is already written and what that file tells its writers to do.
if grep -qE "^\| $SYS .*\*\*verified" README.md 2>/dev/null; then
    ok "README.md records determinism as verified"
else
    warn "README.md does not record this system as verified: run it twice unchanged and compare the"
    warn "gate observable byte for byte BEFORE the first step. Two runs is the cheapest check there"
    warn "is and it is the one that would have caught SHAFT-GPU immediately."
fi

# --- 6. cluster side ----------------------------------------------------------------------------
if [ -r "$CONF" ]; then
    # shellcheck disable=SC1090
    WD=$( . "$CONF" && echo "$WORKDIR" )
    SB=$( . "$CONF" && echo "$SBATCH" )
    # EACH CHECK RUNS INDEPENDENTLY, and that is a bug fix rather than a style choice.
    #
    # These four used to be one `&&` chain inside a command substitution. Any single miss made the
    # whole chain non-zero, so the script took the `else` branch and reported "could not reach
    # palma" -- skipping all four checks and blaming the network. The harness link was the miss:
    # deploy.sh installs ONE shared copy at <camp>/harness, while this looked for it under the
    # system's own workdir. So for EVERY system the cluster-side checks were silently not running,
    # and the warning pointed at the wrong cause. I read that warning several times this session
    # and put it down to a flaky login node.
    #
    # The lesson is the campaign's own: a check that reports the wrong reason for failing is worse
    # than one that fails loudly, because it redirects attention.
    CAMPROOT=${WD%/snni_campaign/*}/snni_campaign
    if ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" true 2>/dev/null; then
        rq() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" "$1" >/dev/null 2>&1; }
        rq "[ -d '$WD' ]"                 && ok "workdir exists on $REMOTE" || bad "no workdir $WD on $REMOTE"
        rq "[ -f '$WD/$SB' ]"             && ok "sbatch present"            || bad "no $SB in $WD"
        rq "[ -r '$CAMPROOT/harness/measure.sh' ]" \
            && ok "shared harness deployed at $CAMPROOT/harness" \
            || bad "shared harness missing: run tools/palma/deploy.sh $SYS"
        rq "find '$WD' -maxdepth 2 -name '*.sif' | grep -q ." \
            && ok "at least one image present" || bad "no .sif under $WD"
    else
        warn "could not reach $REMOTE (ssh itself failed); cluster-side checks skipped"
    fi
fi

echo
echo "NOT CHECKABLE HERE, and required before any step is published:"
echo "  - the recorder LOADS in this system's image. Prove it with a trivial program:"
echo "      tools/attrib_resident/build_in_image.sh --builder docker://gcc:9-buster <image>"
echo "    'it works on the other system' has never been evidence. A host toolchain newer than the"
echo "    container's emits a library the container's loader refuses, and every run exits at once."
echo "  - the recorder floor is stable for THIS system: halve it until the object order stops"
echo "    moving. 64 KiB is the campaign value, justified on the dealer, not proven to transfer."
echo "  - the first PM_INDEX has been eyeballed. A more expensive full scan means a larger gap"
echo "    between the sample and the peak, and on torch/CUDA that gap has been 36x."
echo "  - the payload pid is the process that HOLDS THE MEMORY, confirmed in poller_attach.log"
echo "    against the peak, not assumed from a name."

echo
if [ "$FAIL" -gt 0 ]; then
    echo "$FAIL check(s) failed. This system is not ready to measure."
    exit 1
fi
echo "preflight passed for $SYS."
