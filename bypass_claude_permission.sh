#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <tmux-session-name>"
    exit 1
fi

SESSION="$1"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists. Attaching..."
    tmux attach -t "$SESSION"
    exit 0
fi

tmux new-session -d -s "$SESSION" "IS_SANDBOX=1 claude --dangerously-skip-permissions"

tmux set-option -t "$SESSION" mouse on
tmux set-option -t "$SESSION" history-limit 100000

tmux attach -t "$SESSION"
