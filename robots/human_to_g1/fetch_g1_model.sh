#!/bin/bash
# Fetch the Unitree G1 robot description this study uses.
#
# The description is not copied into this repository. It comes from HoloSoma, an open-source
# humanoid robot project from Amazon. HoloSoma is released under Apache-2.0, but its G1
# model folder has no licence file of its own, so we fetch it rather than copy it. It is
# fetched at one fixed commit (80f1221) so everyone gets the same robot. Only the G1 folder
# and HoloSoma's top-level files, including its LICENSE and NOTICE, are checked out. The
# download is about 105 MB.
#
#     bash robots/human_to_g1/fetch_g1_model.sh
#
# writes external/holosoma/, which the conversion step reads by default.
set -euo pipefail

COMMIT=80f12213aae718171ee836973b151fa8cdfe7ad3
REPO=https://github.com/amazon-far/holosoma
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$ROOT/external/holosoma"
MODELS=src/holosoma_retargeting/holosoma_retargeting/models/g1

if [ -d "$DEST/.git" ]; then
  echo "already present: $DEST"
else
  mkdir -p "$DEST"
  git -C "$DEST" init -q
  git -C "$DEST" remote add origin "$REPO"
  git -C "$DEST" config core.sparseCheckout true
  git -C "$DEST" sparse-checkout init --cone
  git -C "$DEST" sparse-checkout set "$MODELS"
  git -C "$DEST" fetch --depth 1 origin "$COMMIT"
  git -C "$DEST" checkout -q FETCH_HEAD
fi

echo "robot description: $DEST/$MODELS/g1_29dof.xml"
