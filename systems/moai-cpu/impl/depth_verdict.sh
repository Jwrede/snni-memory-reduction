#!/bin/bash
# Verdict on depth probe 46373212 (fallback: m0 run 46371027): does pmrec at depth 7 on zen4 with
# m0's image rank encrypt_zero_symmetric or Ciphertext::operator= first, deciding m1's conformance?
# Addresses go through addr2line because MOAI-CPU links SEAL statically, so peek_live's caller names
# are objects, not symbols.
set -u
J_PROBE=46373212
J_M0=46371027
B=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/results

verdict() {   # <job> <results dir>
    local J="$1" D="$2" node pid out addrs base sym found=""
    node=$(ssh palma "squeue -j $J -h -o '%N' 2>/dev/null" 2>/dev/null | tr -d ' ')
    [ -n "$node" ] || return 1
    out=$(ssh palma "srun --jobid=$J --overlap -n1 python3 /scratch/tmp/j_wred02/snni_campaign/attrib_resident/peek_live.py $D 2>&1")
    echo "--- peek_live ($J, $node) ---"
    echo "$out" | head -16
    pid=$(ssh palma "srun --jobid=$J --overlap -n1 cat $D/moai.pid 2>/dev/null" | tr -dc '0-9')
    [ -n "$pid" ] || { echo "kein pid, kein Urteil"; return 1; }
    addrs=$(echo "$out" | grep -oE '0x[0-9a-f]+$' | head -6)
    [ -n "$addrs" ] || { echo "keine caller-Adressen, kein Urteil"; return 1; }
    # Symbolisieren auf dem Rechenknoten, roh und dann relativ zur Ladebasis.
    base=$(ssh palma "srun --jobid=$J --overlap -n1 sh -c 'grep -m1 -F \"\$(readlink /proc/$pid/exe)\" /proc/$pid/maps'" 2>/dev/null | cut -d- -f1)
    echo "--- addr2line (pid=$pid base=${base:-?}) ---"
    for a in $addrs; do
        sym=$(ssh palma "srun --jobid=$J --overlap -n1 addr2line -e /proc/$pid/exe -f -C -i $a 2>/dev/null" | head -4)
        case "$sym" in *'??'*|'') 
            if [ -n "${base:-}" ]; then
                rel=$(printf '0x%x' $(( a - 0x$base )))
                sym=$(ssh palma "srun --jobid=$J --overlap -n1 addr2line -e /proc/$pid/exe -f -C -i $rel 2>/dev/null" | head -4)
            fi ;;
        esac
        echo "$a -> $(echo "$sym" | tr '\n' ' | ')"
        found="$found $sym"
    done
    echo "=== URTEIL ==="
    if echo "$found" | grep -qi "encrypt_zero_symmetric"; then
        echo "GUT: encrypt_zero_symmetric erscheint bei Tiefe 7 auch auf zen4 mit m0s Image."
        echo "     -> m1s Objektdeklaration ist konform, m0s laufende Messung ist die richtige."
        return 0
    elif echo "$found" | grep -qi "Ciphertext::operator="; then
        echo "ALARM: Rang 1 ist Ciphertext::operator=, nicht encrypt_zero_symmetric."
        echo "     -> die beiden Namen folgen dem IMAGE, nicht der Tiefe. m1 greift nicht Rang 1 an"
        echo "        und braucht eine object_skipped-Deklaration, oder m0 muss auf bigsmp."
        return 0
    else
        echo "UNKLAR: weder Name symbolisiert. Rohausgabe oben lesen."
        return 1
    fi
}

for i in $(seq 1 720); do
    st=$(ssh palma "squeue -j $J_PROBE -h -o '%t' 2>/dev/null" 2>/dev/null | tr -d ' ')
    if [ "$st" = "R" ]; then
        n=$(ssh palma "ls $B/p_depthcheck/run1/pmtable_* 2>/dev/null | wc -l" 2>/dev/null)
        if [ "${n:-0}" -ge 1 ]; then
            sleep 240
            verdict "$J_PROBE" "$B/p_depthcheck/run1" && exit 0
        fi
    elif [ -z "$st" ]; then
        echo "Sonde $J_PROBE ist nicht mehr in der Queue. Endzustand:"
        ssh palma "sacct -X -j $J_PROBE -o JobID,State,Elapsed,ExitCode -n"
        # Ersatzweise den grossen Lauf befragen, der dieselbe Zelle misst.
        stm=$(ssh palma "squeue -j $J_M0 -h -o '%t' 2>/dev/null" 2>/dev/null | tr -d ' ')
        if [ "$stm" = "R" ]; then
            echo "Fallback auf m0 $J_M0:"
            verdict "$J_M0" "$B/m0_default/run1" && exit 0
        fi
        exit 1
    fi
    sleep 60
done
echo "Zeit abgelaufen ohne Urteil."
exit 1
