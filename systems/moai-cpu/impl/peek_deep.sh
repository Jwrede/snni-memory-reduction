#!/bin/bash
# When the depth-10 probe is running with a table: read live, symbolise addresses, and report
# whether program frames appear at effective depth 10 (at depth 6, 91.5% sits in two SEAL members).
set -u
J=46377184
D=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/results/p_depth10/run1
for i in $(seq 1 900); do
    st=$(ssh palma "squeue -j $J -h -o '%t' 2>/dev/null" 2>/dev/null | tr -d ' ')
    if [ "$st" = "R" ]; then
        n=$(ssh palma "ls $D/pmtable_* 2>/dev/null | wc -l" 2>/dev/null)
        if [ "${n:-0}" -ge 1 ]; then
            sleep 150
            echo "=== effektive Tiefe laut Recorder ==="
            ssh palma "grep -ho 'SNNI_PMREC|[^ ]*' /scratch/tmp/j_wred02/snni_campaign/moai-cpu/slurm-mcdeep-$J.err 2>/dev/null | sort -u | head -3"
            echo "=== peek_live ==="
            out=$(ssh palma "srun --jobid=$J --overlap -n1 python3 /scratch/tmp/j_wred02/snni_campaign/attrib_resident/peek_live.py $D 2>&1")
            echo "$out" | head -12
            PID=$(ssh palma "srun --jobid=$J --overlap -n1 cat $D/moai.pid 2>/dev/null" | tr -dc '0-9')
            B=$(ssh palma "srun --jobid=$J --overlap -n1 sh -c 'grep -m1 -F /root/moai/build/test /proc/$PID/maps'" 2>/dev/null | cut -d- -f1)
            echo "=== addr2line (pid=$PID base=0x$B) ==="
            for a in $(echo "$out" | grep -oE '0x[0-9a-f]+$' | head -6); do
                off=$(printf '0x%x' $(( a - 0x$B )))
                echo -n "$a off=$off -> "
                # The whole inline chain, not the first two lines: addr2line -i prints inner to
                # outer and the real derive takes the OUTERMOST; truncating reads the header
                # accessor instead of the program line.
                ssh palma "srun --jobid=$J --overlap -n1 addr2line -e /proc/$PID/exe -f -C -i $off 2>/dev/null" \
                    | paste - - | sed 's/^/      /'
                echo
            done
            echo "=== Lesehilfe ==="
            echo "Ein Name aus test_full_scheme.hpp / single_att_block.hpp / Bootstrapper heisst:"
            echo "Tiefe 10 kommt aus SEAL heraus, und moai-cpu bekommt eine benutzbare Zerlegung."
            echo "Nur seal::* heisst: auch zehn Frames reichen nicht, der Stapel ist tiefer."
            exit 0
        fi
    elif [ -z "$st" ]; then
        echo "Sonde $J nicht mehr in der Queue:"
        ssh palma "sacct -X -j $J -o JobID,State,Elapsed,ExitCode -n"
        exit 1
    fi
    sleep 45
done
