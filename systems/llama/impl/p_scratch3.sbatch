#!/bin/bash
#SBATCH --job-name=llscratch3
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/logs/llscratch3-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/logs/llscratch3-%j.err

# Source audit: unmatched new/delete in every top-level function of api.cpp (SlothGelu settled:
# five new[], zero delete[]; SlothARS 76.9 MB open). Over-reports functions that hand off ownership.
# Nothing measured or published.
set -uo pipefail

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

SIF="${SIF:-/scratch/tmp/j_wred02/snni_campaign/llama/sif/llama-online-o2_activation_free.sif}"
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }

apptainer exec "$SIF" bash -c '
A=/opt/EzPC/GPU-MPC/ext/sytorch/ext/llama/api.cpp
echo "api.cpp has $(wc -l < $A) lines"

echo
echo "=== every top-level function: allocations vs releases ==="
printf "%-52s %6s %6s   %s\n" "function" "new[]" "del[]" "verdict"
awk "
  /^[a-zA-Z_].*\\(/ && /^[a-z]/ { if (fn != \"\") emit(); fn=\$0; start=NR; n=0; d=0; next }
  fn != \"\" && /^}/ { emit(); fn=\"\"; next }
  fn != \"\" { n += gsub(/new GroupElement|new pair<|new osuCrypto|new [A-Za-z_]+\\[/, \"&\"); d += gsub(/delete\\[\\]|delete /, \"&\") }
  function emit(   name, v) {
    name = fn; sub(/\\(.*/, \"\", name); sub(/^[a-zA-Z_<>:*& ]* /, \"\", name)
    if (n > 0) {
      v = (d == 0) ? \"LEAKS ALL\" : (d < n ? \"leaks some\" : \"releases\")
      printf \"%-52s %6d %6d   %s  (line %d)\n\", name, n, d, v, start
    }
  }
  END { if (fn != \"\") emit() }
" "$A"

echo
echo "=== SlothARS in full, since it is the largest object at the current peak ==="
S=$(grep -n "^void SlothARS" "$A" | head -1 | cut -d: -f1)
E=$(awk -v s="$S" "NR>s && /^}/ { print NR; exit }" "$A")
sed -n "${S},${E}p" "$A" | grep -n "new \|delete\|return"
echo "SlothARS spans $S..$E"

echo
echo "=== and the sequence length, which fixes the per-call size ==="
grep -rn "seqlen\|n_seq\|128" /opt/EzPC/GPU-MPC/ext/sytorch/examples/bertbenchmark.cpp 2>/dev/null | grep -vi "bitlength" | head -8
'
RC=$?
echo "PROBE_EXIT_CODE=$RC"
exit "$RC"
