#!/bin/bash
# Fetch the retargeting tool the six-robot study uses.
#
# General Motion Retargeting turns a human performance into a robot performance, and ships
# the descriptions of the six robots with it. It is MIT licensed, and each robot
# description carries its own licence inside the clone. Nothing is copied into this
# repository; the tool is fetched here at one fixed commit so everyone retargets with the
# same version.
#
#     bash robots/fetch_gmr.sh
#
# writes external/GMR/. Put it on the import path when running the study's steps:
#     PYTHONPATH=external/GMR python -m robots.lafan1_to_six_robots.retarget ...
set -euo pipefail

COMMIT=bb1bbe40774794fceb2a7c579a3464a28e68c844
REPO=https://github.com/YanjieZe/GMR.git
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/external/GMR"

if [ -d "$DEST/.git" ]; then
  echo "already present: $DEST"
else
  git clone "$REPO" "$DEST"
  git -C "$DEST" checkout -q "$COMMIT"
fi

echo "retargeting tool: $DEST (commit $COMMIT)"
echo "robot descriptions: $DEST/assets"
