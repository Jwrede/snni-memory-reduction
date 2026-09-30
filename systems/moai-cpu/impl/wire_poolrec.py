#!/usr/bin/env python3
"""Wire the SEAL pool recorder's table (SNNI_POOLREC_TABLE) through measure.sh and the sbatch so the poller can sample it; both edits inert when SNNI_POOLREC is unset.

    wire_poolrec.py <snni_campaign root>
"""
import os
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
ROOT = sys.argv[1]
MEASURE = os.path.join(ROOT, "harness", "measure.sh")
SBATCH = os.path.join(ROOT, "moai-cpu", "moai_cpu_campaign.sbatch")


def edit(path, old, new, what, already, count=1):
    src = open(path, encoding="utf-8").read()
    if already in src:
        print(f"  already present: {what} ({os.path.basename(path)})")
        return
    if src.count(old) != count:
        sys.exit(f"FAILED: '{what}' expected {count} match(es) in {path}, found {src.count(old)}")
    src = src.replace(old, new, count)
    open(path, "w", encoding="utf-8").write(src)
    if already not in src:
        sys.exit(f"FAILED: applied '{what}' but its marker is absent from {path}")
    print(f"  applied: {what} ({os.path.basename(path)})")


# The pollers, both branches (lead and follower): a variable handed to only one is the bug shape
# that hit devtable_ on the follower.
edit(
    MEASURE,
    """            SNNI_PM_TABLE="${PMREC:+$res/pmtable_$p}" \\
            SNNI_PM_SAMPLE="${PMREC:+$AR/pmsample}" \\
            SNNI_PM_FULLSCAN="$SNNI_PM_FULLSCAN" \\
            SNNI_PM_FREEZE="$SNNI_PM_FREEZE" \\
            SNNI_PM_FREEZE_PIDS="$all" \\""",
    """            SNNI_PM_TABLE="${PMREC:+$res/pmtable_$p}" \\
            SNNI_PM_SAMPLE="${PMREC:+$AR/pmsample}" \\
            SNNI_PM_FULLSCAN="$SNNI_PM_FULLSCAN" \\
            SNNI_PM_FREEZE="$SNNI_PM_FREEZE" \\
            SNNI_PM_FREEZE_PIDS="$all" \\
            SNNI_POOLREC_TABLE="${SNNI_POOLREC:+$res/pooltable_$p}" \\""",
    "SNNI_POOLREC_TABLE for the lead poller",
    already="SNNI_POOLREC_TABLE=",
)

edit(
    MEASURE,
    """            SNNI_PM_TABLE="${PMREC:+$res/pmtable_$p}" \\
            SNNI_PM_SAMPLE="${PMREC:+$AR/pmsample}" \\
            SNNI_PM_FULLSCAN="$SNNI_PM_FULLSCAN" \\
                bash "$CAMP/poll_peak.sh" "$p" "$POLL_MS" "$out" \\""",
    """            SNNI_PM_TABLE="${PMREC:+$res/pmtable_$p}" \\
            SNNI_PM_SAMPLE="${PMREC:+$AR/pmsample}" \\
            SNNI_PM_FULLSCAN="$SNNI_PM_FULLSCAN" \\
            SNNI_POOLREC_TABLE="${SNNI_POOLREC:+$res/pooltable_$p}" \\
                bash "$CAMP/poll_peak.sh" "$p" "$POLL_MS" "$out" \\""",
    "SNNI_POOLREC_TABLE for the follower poller",
    already="SNNI_POOLREC_TABLE=\"${SNNI_POOLREC:+$res/pooltable_$p}\" \\\n                bash",
)

# Inside the container: the recorder now writes pmrec's table format and takes its floor from
# SNNI_PM_MIN, so it needs only the path (the old dump-dir + floor line is replaced).
OLD_INNER = (
    'INNER="$INNER${SNNI_POOLREC:+ SNNI_POOLREC=$SNNI_POOLREC SNNI_POOLREC_DIR=$RES '
    'SNNI_POOLREC_FLOOR=${SNNI_POOLREC_FLOOR:-1048576}}"\n'
)
NEW_INNER = (
    'INNER="$INNER${SNNI_POOLREC:+ SNNI_POOLREC=$SNNI_POOLREC '
    'SNNI_POOLREC_TABLE=$RES/pooltable_%d}"\n'
)
src = open(SBATCH, encoding="utf-8").read()
if "SNNI_POOLREC_TABLE=$RES/pooltable_%d" in src:
    print("  already present: SNNI_POOLREC_TABLE in the inner env (sbatch)")
elif OLD_INNER in src:
    open(SBATCH, "w", encoding="utf-8").write(src.replace(OLD_INNER, NEW_INNER, 1))
    print("  applied: SNNI_POOLREC_TABLE in the inner env, replacing the dump-dir version (sbatch)")
else:
    sys.exit("FAILED: neither the old nor the new POOLREC inner-env line is in the sbatch")

print("wire_poolrec.py: done")
