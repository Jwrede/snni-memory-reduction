#!/usr/bin/env python3
"""Lets poll_peak.sh sample the SEAL pool recorder's table inside the same freeze window
(SIGSTOP..SIGCONT); gated by SNNI_POOLREC_TABLE.

    patch_poller_poolrec.py <path to poll_peak.sh>

Edits: (1) %d in SNNI_POOLREC_TABLE -> pid; (2) sample into peaks/resident_pool.txt inside the freeze;
(3) pooltable_ counted as instrument mapping (subtracted).
"""
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
P = sys.argv[1]


def edit(old, new, what, already, count=1):
    src = open(P, encoding="utf-8").read()
    if already in src:
        print(f"  already present: {what}")
        return
    if src.count(old) != count:
        sys.exit(f"FAILED: '{what}' expected {count} match(es), found {src.count(old)}")
    src = src.replace(old, new, count)
    open(P, "w", encoding="utf-8").write(src)
    if already not in src:
        sys.exit(f"FAILED: applied '{what}' but its marker is absent")
    print(f"  applied: {what}")


# 1. The pid substitution, beside the one that already exists.
edit(
    """if [ -n "${SNNI_PM_TABLE:-}" ]; then
    SNNI_PM_TABLE="${SNNI_PM_TABLE//%d/$PID}"
    export SNNI_PM_TABLE
fi
""",
    """if [ -n "${SNNI_PM_TABLE:-}" ]; then
    SNNI_PM_TABLE="${SNNI_PM_TABLE//%d/$PID}"
    export SNNI_PM_TABLE
fi
# The SEAL pool recorder writes the SAME table format into its own shared mapping, so it takes the
# same pid substitution. Only MOAI-CPU sets this; everywhere else the variable is empty and every
# line that mentions it below is skipped.
if [ -n "${SNNI_POOLREC_TABLE:-}" ]; then
    SNNI_POOLREC_TABLE="${SNNI_POOLREC_TABLE//%d/$PID}"
    export SNNI_POOLREC_TABLE
fi
""",
    "resolve %d in SNNI_POOLREC_TABLE",
    already="SNNI_POOLREC_TABLE//%d",
)

# 2. The sample, inside the freeze, immediately after the host table. It runs only when the host
# table just ran, so the two describe one instant and the pause stays bounded by the same gate.
edit(
    """        pt1=$(date +%s%3N)
        pcost=$(( pt1 - pt0 ))
        PM_NEXT_TS=$(( pt1 + pcost * (100 - PM_DUTY_PCT) / (PM_DUTY_PCT > 0 ? PM_DUTY_PCT : 100) ))
        echo "$ts $rss $pcost" >> "$OUTDIR/peaks/PM_INDEX"
    fi
""",
    """        # THE POOL TABLE, IN THE SAME FROZEN WINDOW AND UNDER THE SAME GATE.
        #
        # On MOAI-CPU the program's frees never reach the allocator -- a destroyed Ciphertext goes
        # back to SEAL's pool -- so the host table names who made the POOL GROW rather than what is
        # ALIVE. The recording pool answers the second question, and the answer is only worth
        # having at the instant the first one describes, which is here.
        if [ -n "${SNNI_POOLREC_TABLE:-}" ] && [ -s "$SNNI_POOLREC_TABLE" ]; then
            if "$SNNI_PM_SAMPLE" "$PID" "$SNNI_POOLREC_TABLE" "$OUTDIR/peaks/pool.maps" \\
                    > "$OUTDIR/peaks/resident_pool.txt.tmp" 2>/dev/null; then
                sed -i "1i # sample_ts_ms=$ts sample_rss_kb=$rss source=seal_pool_recorder" \\
                    "$OUTDIR/peaks/resident_pool.txt.tmp"
                mv "$OUTDIR/peaks/resident_pool.txt.tmp" "$OUTDIR/peaks/resident_pool.txt"
            fi
        fi
        pt1=$(date +%s%3N)
        pcost=$(( pt1 - pt0 ))
        PM_NEXT_TS=$(( pt1 + pcost * (100 - PM_DUTY_PCT) / (PM_DUTY_PCT > 0 ? PM_DUTY_PCT : 100) ))
        echo "$ts $rss $pcost" >> "$OUTDIR/peaks/PM_INDEX"
    fi
""",
    "sample the pool table inside the freeze",
    already="resident_pool.txt.tmp",
)

# 3. The instrument subtraction: pooltable_ is a named mapping in the measured process, so it must
# be subtracted like pmtable_/devtable_ or the instrument lands in the published peak.
edit(
    """            /^[0-9a-f]+-[0-9a-f]+ / { inmap = ($0 ~ /\\/(pm|dev)table_/) ; next }""",
    """            /^[0-9a-f]+-[0-9a-f]+ / { inmap = ($0 ~ /\\/(pm|dev|pool)table_/) ; next }""",
    "count pooltable_ as an instrument mapping",
    already="(pm|dev|pool)table_",
)

print("patch_poller_poolrec.py: done")
