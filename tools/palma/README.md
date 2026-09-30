# tools/palma: shared harness

```
campaign.env          the measurement policy: recorder floor, fullscan, freeze, poll interval,
                      replicates. Changing a value here changes every system, which is the point.
run_step.sh           one step: every replicate, every channel, serial, then the derives
run_line.sh           a whole line, serial across steps as well as within them
deploy.sh             push the above plus one conf to a system's cluster workdir
conf/<system>.conf    workdir, sbatch name, and the run matrix for that system
```

Confs exist for 13 systems; `shaft-cpu-vit` and `shaft-gpu-vit` run through their parents' confs.

## Policy

- `tools/check_policy.sh` fails if a system carries its own copy of a policy value. `${VAR:-default}`
  is a violation even with a correct default; an sbatch requires the value (`${VAR:?...}`).
- Origin (2026-07-28): the LLAMA dealer's floor was corrected from 1 MiB to 64 KiB; SHAFT-GPU's sbatch
  kept `SNNI_PM_MIN=${PM_MIN:-1048576}` for a further day without symptom.

## Use

```
tools/check_policy.sh                          # must pass before anything is measured
tools/palma/deploy.sh <system>                 # between lines, never during one
ssh palma '<workdir>/harness/run_step.sh <system> <step_id>'
```

`run_step.sh` while a line is being derived (step N+1 is chosen after step N is measured);
`run_line.sh` to re-measure a line with fixed levers.

## Multi-party capture

`SNNI_PM_FREEZE_MODE=maxparty` (opt-in since 2026-09-29, `freeze_mode=maxparty` in RUN_META): the
leader's poller reads every party's VmRSS at each poll; at a new maximum over the parties it stops all
parties with one kill and captures the party holding the maximum into that party's
`poll_<pid>/peaks` (`snni_start_poller` in `measure.sh`, `maxparty_capture` in `poll_peak.sh`).
`peaks/CAPTURE` and the leader's `peaks/MAXPARTY_INDEX` record who held the maximum. Followers take
no captures.

## Channels

`CHANNELS` is the run matrix of one replicate, space-separated `suffix:VAR=VAL,...` (`-` = no suffix),
as in `tools/palma_chain.sh`:

```
CHANNELS="-:PMREC=1"                             single pool: one run, recorder rides along
CHANNELS="-:PMREC=1,ATTRIB=1,SNNI_COMBINED=1"    two pools: STILL one run, both recorders
```

- Recorder tables are named file-backed mappings, labelled `instrument:` by `smaps_objects.py` and
  subtracted from the peak.
- Retired three-run form `"-: _attrib:ATTRIB=1 _pmrec:PMREC=1"` (object list from another execution
  than the peak). Its directories can persist on scratch; a derive reads `combined_channel=1` from the
  run's `RUN_META` (`systems/sigma-gpu/impl/derive.sbatch`).
- `harvest_step.sh` suffixes: `<step>` -> `logs/`, `<step>_attrib` -> `attrib/device/`,
  `<step>_pmrec` -> `attrib/resident/`; under the combined channel only the first is written.
- Go (ARION) does not allocate through `malloc`: its host objects come from the Go heap census
  (`tools/attrib_go/`), not from `pmrec`.
