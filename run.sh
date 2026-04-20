#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/" && pwd)"
printf "Project root: %s\n" "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

MODE="${1:-full}"

# if [[ -x "${PY:-}" ]]; then
#   PYTHON_BIN="$PY"
# elif [[ -x "/home/khanh/miniconda3/envs/GAI/bin/python" ]]; then
#   PYTHON_BIN="/home/khanh/miniconda3/envs/GAI/bin/python"
# else
PYTHON_BIN="python"
# fi

DEVICE="${DEVICE:-cuda}"
SCALE_LIST="${SCALE_LIST:-10 20 30}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Paper-ready synthetic experiments
MATRIX_ITERS="${MATRIX_ITERS:-500}"
MATRIX_BATCH="${MATRIX_BATCH:-16}"
NAV_ITERS="${NAV_ITERS:-500}"
NAV_BATCH="${NAV_BATCH:-8}"
NAV_HORIZON="${NAV_HORIZON:-15}"
NAV_NUM_SEEDS="${NAV_NUM_SEEDS:-10}"
NAV_N_AGENTS="${NAV_N_AGENTS:-3}"
DATASET="${DATASET:-mujoco}"
MUJOCO_ENV="${MUJOCO_ENV:-halfcheetah-6x1}"
MUJOCO_ALGOS="${MUJOCO_ALGOS:-ippo,mappo,npg_uniform,a2po_diag,a2po_full}"
MUJOCO_SEEDS="${MUJOCO_SEEDS:-0,1,2,3,4}"
MUJOCO_SIGMA="${MUJOCO_SIGMA:-0.3}"
MUJOCO_BATCH_ENVS="${MUJOCO_BATCH_ENVS:-halfcheetah-6x1 ant-4x2 hopper-v4 walker2d-v4 humanoid-v4}"
MUJOCO_BATCH_ITERS="${MUJOCO_BATCH_ITERS:-500}"
MUJOCO_BATCH_BATCH_EPISODES="${MUJOCO_BATCH_BATCH_EPISODES:-4}"
MUJOCO_BATCH_HORIZON="${MUJOCO_BATCH_HORIZON:-15}"
MUJOCO_BATCH_ALGOS="${MUJOCO_BATCH_ALGOS:-ippo,mappo,npg_uniform,a2po_diag,a2po_full}"
MUJOCO_BATCH_SEEDS="${MUJOCO_BATCH_SEEDS:-0}"
MUJOCO_BATCH_SIGMA="${MUJOCO_BATCH_SIGMA:-0.3}"

# CartPole 4-algorithm benchmarks
CARTPOLE_ITERS="${CARTPOLE_ITERS:-500}"
CARTPOLE_BATCH_EPISODES="${CARTPOLE_BATCH_EPISODES:-6}"
CARTPOLE_HORIZON="${CARTPOLE_HORIZON:-50}"

# Matrix scaling alt plot runner
MG_ALT_ITERS="${MG_ALT_ITERS:-500}"
MG_ALT_N_SEEDS="${MG_ALT_N_SEEDS:-8}"
MG_ALT_HET="${MG_ALT_HET:-2.0}"
MG_ALT_LR="${MG_ALT_LR:-0.05}"
MG_ALT_SMOOTH="${MG_ALT_SMOOTH:-11}"
EXP06_AGENT_COUNTS="${EXP06_AGENT_COUNTS:-3 10 20}"

TS="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="logs/run_${TS}"
mkdir -p "$LOG_DIR"

run_logged() {
  local name="$1"
  shift
  local logfile="$LOG_DIR/${name}.log"
  echo
  echo "[RUN] ${name}"
  echo "      $*"
  "$@" 2>&1 | tee "$logfile"
}

run_paper_ready() {
  local outroot="${OUTROOT:-experiments/results/full_${TS}}"
  mkdir -p "$outroot"

  run_logged "exp_list" \
    "$PYTHON_BIN" experiments/run.py --list

  run_logged "exp01_matrix_game" \
    "$PYTHON_BIN" experiments/matrix_game_runner.py \
    --exp exp01 --outdir "$outroot" --iterations "$MATRIX_ITERS" --batch_size "$MATRIX_BATCH"

  run_logged "exp02_navigation" \
    "$PYTHON_BIN" experiments/navigation_runner.py \
    --preset paper --outdir "$outroot/exp02_navigation" \
    --iterations "$NAV_ITERS" --batch_size "$NAV_BATCH" \
    --horizon "$NAV_HORIZON" --num_seeds "$NAV_NUM_SEEDS" \
    --n_agents "$NAV_N_AGENTS"

run_logged "exp02_navigation" \
    "$PYTHON_BIN" experiments/navigation_runner.py \
    --preset local_bottleneck --outdir "$outroot/exp02_navigation_local_bottleneck" \
    --iterations "$NAV_ITERS" --batch_size "$NAV_BATCH" \
    --horizon "$NAV_HORIZON" --num_seeds "$NAV_NUM_SEEDS" \
    --n_agents "$NAV_N_AGENTS"

  run_logged "exp03_heterogeneity_ablation" \
    "$PYTHON_BIN" experiments/matrix_game_runner.py \
    --exp exp03 --outdir "$outroot" --iterations "$MATRIX_ITERS" --batch_size "$MATRIX_BATCH"

  run_logged "exp06_scaling_default_3_10_20" \
    "$PYTHON_BIN" experiments/matrix_game_runner.py \
    --exp exp06 --outdir "$outroot" --iterations "$MATRIX_ITERS" --batch_size "$MATRIX_BATCH" \
    --agent_counts $EXP06_AGENT_COUNTS
}

run_scaling_10_20_30() {
  local mg_out="${MG_ALT_OUTDIR:-results/mg/mg_scale_10_20_30_${TS}}"
  run_logged "mg_scaling_10_20_30" \
    "$PYTHON_BIN" core/mg_convergence_plot_alt.py \
    --n_agents $SCALE_LIST \
    --het "$MG_ALT_HET" \
    --n_seeds "$MG_ALT_N_SEEDS" \
    --seed_start 0 \
    --n_iters "$MG_ALT_ITERS" \
    --lr "$MG_ALT_LR" \
    --smooth "$MG_ALT_SMOOTH" \
    --output "$mg_out"

  if [[ "${DATASET,,}" == "cartpole" ]]; then
    for n in $SCALE_LIST; do
      run_logged "cartpole_4algo_n${n}" \
        "$PYTHON_BIN" plot_convergence.py \
        --dataset cartpole \
        --n_agents "$n" \
        --iterations "$CARTPOLE_ITERS" \
        --batch_episodes "$CARTPOLE_BATCH_EPISODES" \
        --horizon "$CARTPOLE_HORIZON" \
        --device "$DEVICE" \
        --num_workers "$NUM_WORKERS" \
        --parallel \
        --outdir "results/dataset/cartpole_4algo_n${n}_it${CARTPOLE_ITERS}_${TS}"
    done
  elif [[ "${DATASET,,}" == "mujoco" ]]; then
    run_logged "mujoco_benchmark_${MUJOCO_ENV}" \
      "$PYTHON_BIN" plot_convergence.py \
      --dataset mujoco \
      --mujoco_env "$MUJOCO_ENV" \
      --mujoco_algos "$MUJOCO_ALGOS" \
      --mujoco_seeds "$MUJOCO_SEEDS" \
      --mujoco_sigma "$MUJOCO_SIGMA" \
      --iterations "$CARTPOLE_ITERS" \
      --batch_episodes "$CARTPOLE_BATCH_EPISODES" \
      --horizon "$CARTPOLE_HORIZON" \
      --device "$DEVICE" \
      --outdir "results/dataset/mujoco_${MUJOCO_ENV}_it${CARTPOLE_ITERS}_${TS}"
  else
    echo "[ERROR] DATASET không hợp lệ: $DATASET (hỗ trợ: cartpole|mujoco)"
    exit 1
  fi
}

run_mujoco_3env_batch() {
  local outroot="${MUJOCO_BATCH_OUTROOT:-results/dataset/mujoco_3env_${TS}}"
  mkdir -p "$outroot"

  for env_name in $MUJOCO_BATCH_ENVS; do
    run_logged "mujoco_${env_name}" \
      "$PYTHON_BIN" plot_convergence.py \
      --dataset mujoco \
      --mujoco_env "$env_name" \
      --mujoco_algos "$MUJOCO_BATCH_ALGOS" \
      --mujoco_seeds "$MUJOCO_BATCH_SEEDS" \
      --mujoco_sigma "$MUJOCO_BATCH_SIGMA" \
      --iterations "$MUJOCO_BATCH_ITERS" \
      --batch_episodes "$MUJOCO_BATCH_BATCH_EPISODES" \
      --horizon "$MUJOCO_BATCH_HORIZON" \
      --device "$DEVICE" \
      --outdir "$outroot/$env_name"
  done
}

print_summary() {
  echo
  echo "Done."
  echo "- Python: $PYTHON_BIN"
  echo "- Mode: $MODE"
  echo "- Device: $DEVICE"
  echo "- Logs: $LOG_DIR"
}

case "$MODE" in
  full)
    run_paper_ready
    run_scaling_10_20_30
    ;;
  paper)
    run_paper_ready
    ;;
  scale)
    run_scaling_10_20_30
    ;;
  mujoco3|mujoco_batch)
    run_mujoco_3env_batch
    ;;
  cartpole)
    for n in $SCALE_LIST; do
      run_logged "cartpole_4algo_n${n}" \
        "$PYTHON_BIN" plot_convergence.py \
        --dataset cartpole \
        --n_agents "$n" \
        --iterations "$CARTPOLE_ITERS" \
        --batch_episodes "$CARTPOLE_BATCH_EPISODES" \
        --horizon "$CARTPOLE_HORIZON" \
        --device "$DEVICE" \
        --num_workers "$NUM_WORKERS" \
        --parallel \
        --outdir "results/dataset/cartpole_4algo_n${n}_it${CARTPOLE_ITERS}_${TS}"
    done
    ;;
  *)
    echo "[ERROR] MODE không hợp lệ: $MODE"
    echo "Usage: ./run.sh [full|paper|scale|mujoco3|mujoco_batch|cartpole]"
    exit 1
    ;;
esac

# cd /home/khanh/Khanh_stuff/Inprogress/A2PO && SCALE_LIST="10 20 30" MG_ALT_N_SEEDS=8 MG_ALT_ITERS=500 MG_ALT_HET=2.0 MG_ALT_LR=0.05 MG_ALT_SMOOTH=11 ./run.sh scale

# cd /home/khanh/Khanh_stuff/Inprogress/A2PO && MG_ALT_OUTDIR=results/mg_scale_10_20_30_20260418_163410 /home/khanh/miniconda3/envs/GAI/bin/python core/mg_convergence_plot_alt.py --n_agents 10 20 30 --het 2.0 --n_seeds 8 --seed_start 0 --n_iters 500 --lr 0.05 --smooth 11 --output results/mg_scaling_10_20_30

# /home/khanh/miniconda3/envs/GAI/bin/python plot_convergence.py   --parallel   --device cuda   --iterations 500   --batch_episodes 6   --horizon 50   --smooth 5   --csv_log_interval 10   --outdir results/dataset/"$DATASET"   --a2po_eta 0.003   --a2po_beta 0.9   --a2po_reg_lambda 0.01

# python compare_gym_algorithms.py --n_agents 5 --iterations 500 --device cuda --a2po_eta 0.003 --a2po_beta 0.9 --a2po_reg_lambda 0.01

print_summary
