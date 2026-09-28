#!/usr/bin/env bash
# Download the Motion2Motion authors' own program (Chen et al., arXiv:2508.13139).
#
# It is not included in this repository. It carries no licence, so it is not ours to pass on.
# This downloads the exact version the paper used into external/motion2motion, which git
# ignores.
#
# One line of it is changed afterwards. When the correspondence file names a joint the
# skeleton does not have, their program opens an interactive debugger and waits for someone to
# type at it. Over thousands of queries that stops everything. The line is replaced so that
# the query is simply recorded as unanswered and the rest carry on. Nothing else is touched.
#
#     bash methods/motion2motion/fetch_official.sh
#
# Nothing else needs installing: the repository's main environment already has what it uses.

set -euo pipefail

REPO="https://github.com/LinghaoChan/Motion2Motion_codes.git"
COMMIT="751d114785141491895b88719198128add1e2b61"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$ROOT/external/motion2motion"

if [ -d "$DEST/.git" ]; then
  echo "$DEST already exists; fetching $COMMIT"
  git -C "$DEST" fetch --quiet origin
else
  mkdir -p "$(dirname "$DEST")"
  git clone --quiet "$REPO" "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT"
echo "checked out $(git -C "$DEST" rev-parse HEAD)"

python3 - "$DEST/dataset/bvh/bvh_parser.py" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
old = "            import pdb; pdb.set_trace()\n"
new = ('            raise RuntimeError(  # replaces a call that opened a debugger here\n'
       '                "joint %r is not in this skeleton" % (target,))\n')
if old in text:
    path.write_text(text.replace(old, new, 1))
    print("changed", path)
elif new in text:
    print("already changed", path)
else:
    sys.exit("could not find the debugger call in %s; change it by hand" % path)
PY

echo "done: $DEST"
