#!/bin/bash
# Lightweight health sampler for TabFM runs (macOS).
# Logs load, free RAM, swap used, and free disk once per interval. Negligible
# overhead. Run it alongside any heavy job to watch for memory pressure building.
#
# Usage: ./sysmon.sh [interval_seconds] [logfile]
#   ./sysmon.sh 5 /tmp/tabfm_sysmon.log
# Stop with Ctrl-C or by killing the process.
set -u
INTERVAL="${1:-5}"
LOG="${2:-/tmp/tabfm_sysmon.log}"
PAGE=$(sysctl -n hw.pagesize)

hdr="epoch,iso,load1,free_ram_gb,swap_used_gb,disk_free_gb"
echo "$hdr" | tee -a "$LOG"
while true; do
  pages=$(vm_stat | awk -F: '/Pages free/{f=$2} /Pages inactive/{i=$2} /Pages speculative/{s=$2} END{gsub(/[ .]/,"",f);gsub(/[ .]/,"",i);gsub(/[ .]/,"",s); print f+i+s}')
  free_ram_gb=$(awk -v p="$pages" -v ps="$PAGE" 'BEGIN{printf "%.2f", p*ps/1073741824}')
  swap_used_gb=$(sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used"){t=$(i+2); n=t; sub(/[A-Za-z]/,"",n); if(t ~ /G/) printf "%.2f", n; else printf "%.2f", n/1024}}')
  disk_free_gb=$(df -g "$HOME" | awk 'NR==2{print $4}')
  load1=$(sysctl -n vm.loadavg | awk '{print $2}')
  ep=$(date +%s); iso=$(date "+%Y-%m-%dT%H:%M:%S")
  echo "$ep,$iso,$load1,$free_ram_gb,$swap_used_gb,$disk_free_gb" | tee -a "$LOG"
  sleep "$INTERVAL"
done
