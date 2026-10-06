#!/usr/bin/env bash
# Run the three SeaThru-NeRF baseline scenes serially (avoids the power/PSU overload
# of running them in parallel). Each scene trains for 30000 iters; logs go to
# outputs/baseline/<scene>_train.log.
set -eo pipefail

source /home/cxh/anaconda3/bin/activate APEX
export MPLBACKEND=Agg
cd "$(dirname "${BASH_SOURCE[0]}")/../src"

for scene in Curasao IUI3-RedSea JapaneseGradens-RedSea; do
  echo "===== TRAIN $scene ====="
  python train.py -s "../datasets/seathru_undist/$scene" \
                  -m "../outputs/baseline/$scene" --eval 2>&1 \
    | tee "../outputs/baseline/${scene}_train.log" \
    | grep -E "ITER|Evaluating|Number of point|Training complete|Error|Traceback|out of memory"
  echo "===== DONE $scene ====="
done
echo "ALL DONE"
