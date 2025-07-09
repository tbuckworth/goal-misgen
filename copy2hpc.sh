#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <relative_local_dir>"
  exit 1
fi

# e.g. "logs/other/tender/beings/files"
LOCAL_PATH="${1%/}"
REMOTE_HOST="login.cx3.hpc.imperial.ac.uk"
REMOTE_BASE="~/pyg/goal-misgen"

# Determine where on the remote to drop LOCAL_PATH
REMOTE_PARENT=$(dirname "$LOCAL_PATH")          # "logs/other/tender/beings"
REMOTE_BASENAME=$(basename "$LOCAL_PATH")       # "files"

# Ensure the full parent path exists remotely
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_BASE/$REMOTE_PARENT"

# Copy the directory (and all subdirs) into that parent
scp -r "$LOCAL_PATH" "$REMOTE_HOST:$REMOTE_BASE/$REMOTE_PARENT/"

echo "✅ '$LOCAL_PATH' copied to $REMOTE_HOST:$REMOTE_BASE/$REMOTE_PARENT/$REMOTE_BASENAME"
