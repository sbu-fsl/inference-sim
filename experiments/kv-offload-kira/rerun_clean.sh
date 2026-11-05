#!/bin/bash
# One clean rerun of this experiment from fresh clones, then a diff against the committed results.
# Empty `git status` at the end = every committed result reproduced byte for byte.
#
#   PY=python3 experiments/kv-offload-kira/rerun_clean.sh /tmp/kira-rerun1 [commit]   # PY needs pyarrow and pyyaml
set -euo pipefail
W=$1; REV=${2:-kira-kv-offload-repro}; PY=${PY:-python3}
rm -rf "$W"; mkdir -p "$W/blis-deps"; cd "$W"
git clone -q https://github.com/therealansh/inference-sim.git inference-sim
git -C inference-sim checkout -q "$REV"
git clone -q --branch 0.3.0 --depth 1 https://github.com/inference-sim/blis-catalog.git blis-deps/blis-catalog
git clone -q --branch v0.2.0 --depth 1 https://github.com/inference-sim/blis-registry.git blis-deps/blis-registry
E=inference-sim/experiments/kv-offload-kira
(cd inference-sim && go build -o blis main.go)
"$PY" $E/overlay/build_overlay.py                 # fetches the Qwen config.json at its pinned revision, sha256-checked
for c in calibrate checks cases; do "$PY" $E/kira_repro.py $c > "$W/$c.log"; done
(cd $E/gpubench && "$PY" fit_gpubench.py ibm1-102.json ibm1-103.json && "$PY" fit_gpubench.py ibm2-96.json) > "$W/fit.log"
git -C inference-sim status --short --untracked-files=no
echo "rerun at $(git -C inference-sim rev-parse --short HEAD) done; changed committed files listed above (none = reproduced)"
