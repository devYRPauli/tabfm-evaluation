# Safety and Resource Limits

## Incident: MacBook restart from memory exhaustion (2026-06-30)

What happened: the 16 GB MacBook (M1 Pro) restarted uncleanly while running TabFM.

Evidence-based root cause:
1. Before the restart, free disk on the data volume fell from about 48 GB to
   9.1 GB. That roughly 39 GB was swap macOS created under memory pressure. On a
   16 GB RAM machine, ~39 GB of swap implies a working set on the order of 55 GB.
2. After the restart the swap was reclaimed (disk back to about 48 GB free) while
   the 11 GB weight cache remained. So the consumed space was swap, not data.
3. TabFM's default forward pass uses a 32-member ensemble. The same configuration
   allocated about 22.7 GiB on the workstation GPU. On a 16 GB CPU host this forces
   tens of GB of swap.
4. Compounding factor: a leftover environment-build example run was still active
   when a second job (Phase 1 conformance, which loads two models) was started.
   Multiple large TabFM processes ran at once on 16 GB.

No CPU kernel-panic report was written, consistent with a memory-pressure restart
rather than a logic panic.

This was avoidable. The full model should never have been run on the 16 GB machine,
and never concurrently with another TabFM job.

## Hard rules (apply to every TabFM run from now on)

1. Heavy TabFM runs (full config, Phases 1 to 4) default to the Studio (64 GB) and
   the workstation (125 GB or GPU). The 16 GB MacBook is memory-marginal for this
   model and is not a default compute target.
2. Any command that loads a TabFM model on the MacBook must run through
   `harness/safe_run.sh` with the 16 GB defaults (mem <= 9 GB RSS, disk >= 15 GB,
   swap <= 4 GB). safe_run kills the job before the OS is endangered. A killed job
   is an acceptable outcome; a machine restart is not.
3. Never run two TabFM jobs concurrently on one machine. safe_run preflight refuses
   to start if another TabFM job is detected. Always confirm that any
   subagent-launched background run has actually finished before starting another.
4. Before any MacBook run, ensure at least 15 GB free disk for swap headroom, and
   let the preflight confirm it.
5. Watch every heavy run. Start `harness/sysmon.sh 5 <log>` alongside it, on
   whichever machine it runs, and stop if swap climbs or free disk drops.
6. The MacBook "small-machine ceiling" datapoint is obtained by careful, monitored,
   capped probing only: small `n_estimators`, capped context, safe_run as the
   kill-switch. If TabFM cannot run within the cap, the honest finding is that it
   does not fit in 16 GB, reported as such. We never approach the ceiling by
   running the full config blind.
7. On the Studio and workstation, also run under sysmon, and use safe_run with a
   higher `--mem-gb` (for example 48 on the Studio) as a backstop, since a runaway
   can swap those machines too. Copy the harness scripts to the remote machine
   first (rsync into ~/tabfm-eval/harness).

## Tools

1. `harness/sysmon.sh [interval] [logfile]`: samples load, free RAM, swap used, and
   free disk per tick. Negligible overhead. Run beside any heavy job.
2. `harness/safe_run.sh [--mem-gb N] [--min-disk-gb N] [--swap-ceil-gb N]
   [--interval S] [--log F] -- CMD...`: preflight gate plus a watchdog that kills
   the job if its resident memory, system swap, or free disk crosses a limit.
   Defaults are tuned for 16 GB.

## Portability of the guardrail scripts

harness/sysmon.sh and harness/safe_run.sh are macOS-tuned (they use
`sysctl vm.swapusage`, `vm_stat`, `df -g` on the home volume). They are the guard
on the Macs. On the Linux workstation they do not run as-is. There the OS-restart
risk is negligible (125 GB RAM against a roughly 22 GB working set), so we rely on
the large RAM headroom, a `free -h` pre-check, and `nvidia-smi` for the real
constraint (24 GB GPU memory, where an overflow errors the process rather than
taking down the machine). A Linux port of the watchdog is a TODO if deeper
protection is wanted.

Per the user's decision after the 2026-06-30 incident, the 16 GB MacBook is
orchestration-only and never loads a TabFM model. Heavy runs go to the Studio
(64 GB, under safe_run with --mem-gb 48) and the workstation (single GPU via
CUDA_VISIBLE_DEVICES=0). The brief's laptop-CPU datapoint is dropped for safety;
the hardware comparison is Studio CPU versus single-4090 GPU.

## How heavy is TabFM, for planning

The 32-member ensemble allocated about 22.7 GiB on the workstation GPU. The weights
cache is about 11 to 12 GB on disk. Treat a default full forward pass as a
20 GB-plus working set. That fits comfortably on the Studio (64 GB) and workstation
(125 GB), and does not fit on the 16 GB MacBook without aggressive capping.
