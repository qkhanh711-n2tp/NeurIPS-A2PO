#!/usr/bin/env bash
set -euo pipefail

# Batch runner for compare_gym_algorithms.py across multiple PettingZoo MPE envs.
# Usage:
#   bash run_compare_mpe_envs.sh
# Optional overrides:
#   N_AGENTS=5 ITERATIONS=100 DEVICE=cuda SEED=42 bash run_compare_mpe_envs.sh

N_AGENTS="${N_AGENTS:-3}"
ITERATIONS="${ITERATIONS:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"
HORIZON="${HORIZON:-25}"
SEED="${SEED:-42}"
MPE_LOCAL_RATIO="${MPE_LOCAL_RATIO:-0.0}"
A2PO_ETA="${A2PO_ETA:-0.003}"
A2PO_BETA="${A2PO_BETA:-0.9}"
A2PO_REG_LAMBDA="${A2PO_REG_LAMBDA:-0.01}"
PYTHON_CMD="${PYTHON_CMD:-python}"
OUTROOT="${OUTROOT:-results/dataset/mpe}"
export PYTHONUNBUFFERED=1

ENVS=(
  "simple_spread_v3"
  "simple_reference_v3"
  "simple_push_v3"
)

for ENV_NAME in "${ENVS[@]}"; do
  echo "============================================================"
  echo "Running compare_gym_algorithms.py on MPE env: ${ENV_NAME}"
  echo "============================================================"

  ${PYTHON_CMD} compare_gym_algorithms.py \
    --env_family mpe \
    --env_name "${ENV_NAME}" \
    --n_agents "${N_AGENTS}" \
    --iterations "${ITERATIONS}" \
    --device "${DEVICE}" \
    --batch_episodes "${BATCH_SIZE}" \
    --horizon "${HORIZON}" \
    --seed "${SEED}" \
    --mpe_local_ratio "${MPE_LOCAL_RATIO}" \
    --a2po_eta "${A2PO_ETA}" \
    --a2po_beta "${A2PO_BETA}" \
    --a2po_reg_lambda "${A2PO_REG_LAMBDA}" \
    --outdir "${OUTROOT}/${ENV_NAME}/n${N_AGENTS}/it${ITERATIONS}_seed${SEED}"
done

echo "All MPE environment runs completed."
