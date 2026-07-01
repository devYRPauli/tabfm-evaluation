#!/bin/bash
# Guarded runner for heavy TabFM jobs (macOS has no cgroups, so we poll and kill).
#
# It refuses to start if a TabFM/JAX job is already running or if free disk is too
# low for safe swap headroom, then it watches the job's resident memory, system
# swap, and free disk. If any crosses its limit it kills the job BEFORE the OS is
# endangered. A killed job is a far better outcome than a machine restart.
#
# Usage:
#   safe_run.sh [--mem-gb N] [--min-disk-gb N] [--swap-ceil-gb N] [--interval S] [--log F] -- CMD [ARGS...]
# Defaults are tuned for a 16 GB machine. Raise --mem-gb on the Studio (64 GB) or
# the workstation (125 GB).
set -u

MEM_GB=9            # kill if the job's tree RSS exceeds this
MIN_DISK_GB=15      # refuse to start, and abort, below this free disk
SWAP_CEIL_GB=4      # abort if system swap usage exceeds this
INTERVAL=3
LOG="/tmp/tabfm_safe_run_$$.log"

die(){ echo "safe_run: $*" >&2; exit 2; }
usage(){ echo "usage: safe_run.sh [--mem-gb N] [--min-disk-gb N] [--swap-ceil-gb N] [--interval S] [--log F] -- CMD [ARGS...]" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --mem-gb) MEM_GB="$2"; shift 2;;
    --min-disk-gb) MIN_DISK_GB="$2"; shift 2;;
    --swap-ceil-gb) SWAP_CEIL_GB="$2"; shift 2;;
    --interval) INTERVAL="$2"; shift 2;;
    --log) LOG="$2"; shift 2;;
    --) shift; break;;
    -h|--help) usage;;
    *) die "unknown arg: $1";;
  esac
done
[ $# -ge 1 ] || usage

free_disk_gb(){ df -g "$HOME" | awk 'NR==2{print $4}'; }
swap_used_gb(){ sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used"){t=$(i+2); n=t; sub(/[A-Za-z]/,"",n); if(t ~ /G/) printf "%.2f", n; else printf "%.2f", n/1024}}'; }
gt(){ awk -v a="$1" -v b="$2" 'BEGIN{exit !(a>b)}'; }   # true (0) if a > b
lt(){ awk -v a="$1" -v b="$2" 'BEGIN{exit !(a<b)}'; }   # true (0) if a < b
tree_rss_gb(){ local pid="$1"; local kids; kids=$(pgrep -P "$pid" 2>/dev/null); ps -o rss= -p "$pid" $kids 2>/dev/null | awk '{s+=$1} END{printf "%.2f", s/1048576}'; }

# Preflight: no competing TabFM/JAX job, and enough disk for swap headroom.
# Match actual python jobs, but exclude our own wrapper lines (safe_run's own
# argv contains the python+tabfm path and would otherwise self-trigger).
existing=$(ps ax -o pid=,command= | grep -iE 'python[0-9.]* .*(tabfm|phase[0-9]|sanity|conformance)' | grep -v grep | grep -v safe_run)
[ -n "$existing" ] && die "refusing: a TabFM job already appears to be running:
$existing"
df0=$(free_disk_gb)
lt "$df0" "$MIN_DISK_GB" && die "refusing: only ${df0}GB free disk (< ${MIN_DISK_GB}GB needed for swap headroom)"

echo "safe_run: limits mem<=${MEM_GB}GB rss, disk>=${MIN_DISK_GB}GB, swap<=${SWAP_CEIL_GB}GB; log=$LOG"
echo "epoch,rss_gb,swap_gb,disk_free_gb" > "$LOG"

"$@" &
CHILD=$!
PEAK_RSS=0

while kill -0 "$CHILD" 2>/dev/null; do
  rss=$(tree_rss_gb "$CHILD"); [ -z "$rss" ] && rss=0
  sw=$(swap_used_gb); [ -z "$sw" ] && sw=0
  dk=$(free_disk_gb); [ -z "$dk" ] && dk=0
  echo "$(date +%s),$rss,$sw,$dk" >> "$LOG"
  gt "$rss" "$PEAK_RSS" && PEAK_RSS="$rss"
  REASON=""
  gt "$rss" "$MEM_GB" && REASON="job RSS ${rss}GB exceeded ${MEM_GB}GB"
  [ -z "$REASON" ] && gt "$sw" "$SWAP_CEIL_GB" && REASON="system swap ${sw}GB exceeded ${SWAP_CEIL_GB}GB"
  [ -z "$REASON" ] && lt "$dk" "$MIN_DISK_GB" && REASON="free disk ${dk}GB fell below ${MIN_DISK_GB}GB"
  if [ -n "$REASON" ]; then
    echo "safe_run: ABORT -> $REASON. killing job to protect the machine." >&2
    for p in $(pgrep -P "$CHILD" 2>/dev/null) "$CHILD"; do kill -TERM "$p" 2>/dev/null; done
    sleep 2
    for p in $(pgrep -P "$CHILD" 2>/dev/null) "$CHILD"; do kill -KILL "$p" 2>/dev/null; done
    wait "$CHILD" 2>/dev/null
    echo "safe_run: job killed. peak RSS was ${PEAK_RSS}GB." >&2
    exit 137
  fi
  sleep "$INTERVAL"
done

wait "$CHILD"; rc=$?
echo "safe_run: job exited rc=$rc, peak RSS ${PEAK_RSS}GB (trace in $LOG)"
exit $rc
