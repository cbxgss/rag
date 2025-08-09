#!/bin/bash

sync_to_remote() {
  local remote="$1"
  local remote_dir="$2"
  local_dir=$(basename "$PWD")
  cd ..
  local exclude_file="${local_dir}/scripts/.sync_exclude"

  [ -f "$exclude_file" ] || touch "$exclude_file"

  rsync -avzh \
    --update \
    --exclude-from="$exclude_file" \
    --progress \
    "$local_dir/" "$remote:$remote_dir" \
    --delete \

  echo "-> $remote:$remote_dir"

  cd "$local_dir"
}

sync_to_remote f11 workspace/rag/dualrag
