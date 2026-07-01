#!/bin/bash
# Waits for the Phase 3 sweep tmux session to finish, then aggregates the results
# and fires a macOS desktop notification with the headline. Runs independently of
# any Claude session (launch it in its own tmux). Also writes a DONE_headline.txt.
set -u
REPO="/Users/yashrajpandey/tabfm-evaluation"

while tmux has-session -t tabfm-sweep 2>/dev/null; do sleep 60; done
sleep 5  # let the final collect() rsync settle

OUT=$(python3 "$REPO/harness/aggregate_phase3.py" 2>&1)
HEAD=$(printf '%s\n' "$OUT" | tail -1)
printf '%s\n' "$HEAD" > "$REPO/results/phase3/DONE_headline.txt"
osascript -e "display notification \"$HEAD\" with title \"TabFM sweep complete\" sound name \"Glass\"" 2>/dev/null
