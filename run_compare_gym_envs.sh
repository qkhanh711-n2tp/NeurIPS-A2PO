#!/usr/bin/env bash
set -euo pipefail

# Batch runner for compare_gym_algorithms.py across multiple Gym/Gymnasium envs.
# Usage:
#   bash run_compare_gym_envs.sh
# Optional overrides:
#   N_AGENTS=5 ITERATIONS=100 DEVICE=cuda SEED=42 bash run_compare_gym_envs.sh

N_AGENTS="${N_AGENTS:-10}"
ITERATIONS="${ITERATIONS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cpu}"
HORIZON="${HORIZON:-15}"
SEED="${SEED:-42}"
A2PO_ETA="${A2PO_ETA:-0.003}"
A2PO_BETA="${A2PO_BETA:-0.9}"
A2PO_REG_LAMBDA="${A2PO_REG_LAMBDA:-0.01}"
PYTHON_CMD="${PYTHON_CMD:-python}"
export PYTHONUNBUFFERED=1

ENVS=(
  "LunarLander-v3"
  "Pendulum-v1"
  "MountainCar-v0"
  "HalfCheetah-v4"
  "Hopper-v4"
  "Walker2d-v4"
  "Ant-v4"
  "Humanoid-v4"
)

for ENV_NAME in "${ENVS[@]}"; do
  echo "============================================================"
  echo "Running compare_gym_algorithms.py on env: ${ENV_NAME}"
  echo "============================================================"

  ${PYTHON_CMD} compare_gym_algorithms.py \
    --env_name "${ENV_NAME}" \
    --n_agents "${N_AGENTS}" \
    --iterations "${ITERATIONS}" \
    --device "${DEVICE}" \
    --batch_episodes "${BATCH_SIZE}" \
    --horizon "${HORIZON}" \
    --seed "${SEED}" \
    --a2po_eta "${A2PO_ETA}" \
    --a2po_beta "${A2PO_BETA}" \
    --a2po_reg_lambda "${A2PO_REG_LAMBDA}"
done

echo "All environment runs completed."
