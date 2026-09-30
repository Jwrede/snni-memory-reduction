#!/usr/bin/env python3
"""Applies a1_btpeval_shallowcopy to a prepared ARION source tree (image build, after
pin_workload.py, before `go build`).
    patch_shallowcopy.py <src-tree>

Each worker rebuilt the read-only bootstrapping DFT matrices (65 x 3.24 GiB = 210.7 GiB); now shared.
Version 2: lattigo v6.1.1's bootstrapping ShallowCopy uses ResidualParameters where NewEvaluator uses
BootstrappingParameters (bare ShallowCopy panics "level cannot be larger than max level", job
45892530); the helper rebuilds the two affected fields the constructor's way. Two worker call sites.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/arion")

# --- the helper, in the package that already owns the deep copy -------------------------------

HELPER = '''

// ShallowCopyBootstrapEvaluator returns an evaluator that SHARES the read-only DFT matrices with
// the receiver while every buffer a worker must not share is reallocated.
//
// a1_btpeval_shallowcopy. It is lattigo's own ShallowCopy with two fields rebuilt, because
// ShallowCopy and NewEvaluator disagree about which parameters the sub-evaluators take:
// NewEvaluator uses BootstrappingParameters (evaluator.go:117-123), ShallowCopy uses
// ResidualParameters (evaluator.go:133,153-154). The DFT matrices live at the bootstrapping
// parameters' LevelQ, which is above the residual parameters' max level, so the unmodified
// ShallowCopy panics with "level cannot be larger than max level" on the first CoeffsToSlots.
// Measured, job 45892530. The two lines below are copied from NewEvaluator, not invented.
func ShallowCopyBootstrapEvaluator(origEval *bootstrapping.Evaluator) *bootstrapping.Evaluator {
	newEval := origEval.ShallowCopy()
	params := origEval.Parameters.BootstrappingParameters
	newEval.DFTEvaluator = dft.NewEvaluator(params, newEval.Evaluator)
	newEval.Mod1Evaluator = mod1.NewEvaluator(newEval.Evaluator,
		polynomial.NewEvaluator(params, newEval.Evaluator), newEval.Mod1Parameters)
	return newEval
}
'''

IMPORT_ANCHOR = '\t"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"\n'
IMPORT_NEW = IMPORT_ANCHOR + (
    '\t"github.com/tuneinsight/lattigo/v6/circuits/ckks/dft"\n'
    '\t"github.com/tuneinsight/lattigo/v6/circuits/ckks/mod1"\n'
    '\t"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"\n'
)

p = root / "pkg/btp/matbtp.go"
src = p.read_text()
if src.count(IMPORT_ANCHOR) != 1:
    sys.exit(f"FAILED: the bootstrapping import in pkg/btp/matbtp.go matched "
             f"{src.count(IMPORT_ANCHOR)} times, expected 1")
src = src.replace(IMPORT_ANCHOR, IMPORT_NEW, 1) + HELPER
p.write_text(src)
print("  patched: ShallowCopyBootstrapEvaluator added  (pkg/btp/matbtp.go)")

# --- the two worker sites ---------------------------------------------------------------------

SITES = [
    (
        "pkg/btp/matbtp.go",
        "\t\t\tlocalBtpEval := DeepCopyBootstrapEvaluator(btpEval, btpParams) "
        "// 每个线程使用自己的 evaluator\n"
        "\t\t\t// localBtpEval := btpEval.ShallowCopy()\n",
        "\t\t\t// a1_btpeval_shallowcopy: share the read-only DFT matrices instead of\n"
        "\t\t\t// rebuilding them per worker.\n"
        "\t\t\tlocalBtpEval := ShallowCopyBootstrapEvaluator(btpEval)\n",
    ),
    (
        "pkg/bert/attentionMT.go",
        "\t\t\t// localBtpEval := btpEval.ShallowCopy()\n"
        "\t\t\tlocalBtpEval := btp.DeepCopyBootstrapEvaluator(btpEval, btpEval.Parameters)\n",
        "\t\t\t// a1_btpeval_shallowcopy: share the read-only DFT matrices instead of\n"
        "\t\t\t// rebuilding them per worker.\n"
        "\t\t\tlocalBtpEval := btp.ShallowCopyBootstrapEvaluator(btpEval)\n",
    ),
]

for path, old, new in SITES:
    p = root / path
    src = p.read_text()
    n = src.count(old)
    if n != 1:
        sys.exit(f"FAILED: the deep-copy block in {path} matched {n} times, expected 1. "
                 f"Upstream moved; re-derive the patch.")
    p.write_text(src.replace(old, new))
    print(f"  patched: worker shares the bootstrapping evaluator's DFT matrices  ({path})")

# --- checks, over the whole tree and against the FILES rather than the intent ------------------

live = []
for p in root.rglob("*.go"):
    for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
        s = line.strip()
        if s.startswith("//"):
            continue
        if "DeepCopyBootstrapEvaluator(" in s and "func DeepCopyBootstrapEvaluator" not in s:
            live.append(f"{p.relative_to(root)}:{i}")
if live:
    sys.exit("FAILED: DeepCopyBootstrapEvaluator is still called at " + ", ".join(live))

t = (root / "pkg/btp/matbtp.go").read_text()
for must in ("func ShallowCopyBootstrapEvaluator(",
             "newEval.DFTEvaluator = dft.NewEvaluator(params, newEval.Evaluator)",
             "localBtpEval := ShallowCopyBootstrapEvaluator(btpEval)"):
    if must not in t:
        sys.exit(f"FAILED: {must!r} is not in the patched pkg/btp/matbtp.go")
if "btp.ShallowCopyBootstrapEvaluator(btpEval)" not in (root / "pkg/bert/attentionMT.go").read_text():
    sys.exit("FAILED: the attention worker does not call the shallow copy")

# The bare ShallowCopy is the version that panicked; refuse it at a worker site.
for path in ("pkg/btp/matbtp.go", "pkg/bert/attentionMT.go"):
    for i, line in enumerate((root / path).read_text().splitlines(), 1):
        s = line.strip()
        if s.startswith("//"):
            continue
        if "localBtpEval := btpEval.ShallowCopy()" in s:
            sys.exit(f"FAILED: the unmodified ShallowCopy is live at {path}:{i}; that is the "
                     f"version that panicked in job 45892530")
print("  verified against the files: helper present, no deep copy, no bare ShallowCopy")
