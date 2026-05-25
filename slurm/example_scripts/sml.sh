EVAL_DIR=" "
cd "$EVAL_DIR" || exit 1

export VLLM_USE_V1=1
export VLLM_LOG_LEVEL=DEBUG
export HF_HOME=
export HF_HUB_CACHE=
export TORCH_HOME=

MODEL_NAME=

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m vllm.entrypoints.openai.api_server --model $MODEL_NAME --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --port 8000 \
  > "" &
VLLM_PID=$!

while ! nc -z localhost 8000; do
  sleep 1
done

interaction_model=
progress=0
steps=1000
max_history=0
text_info=0
with_game_area=0
epsilon_noise=0.0

# Parallel execution settings
MAX_PARALLEL_JOBS=

# 10 world-level pairs: (1,1) (1,2) (1,3) (2,1) (2,2) (3,1) (3,2) (3,3) (4,1) (4,2)
WORLD_LEVELS="1 1 1 2 1 3 2 1 2 2 3 1 3 2 3 3 4 1 4 2"

# Function to run a single evaluation
run_evaluation() {
    local run_count=$1
    local world=$2
    local level=$3
    local game_state_path=$4
    local progress_token=$5
    echo "Running evaluation for run ${run_count} (World ${world}, Level ${level}, progress ${progress_token})"
    python game_agent_evaluation_fromuser.py \
      --game_config_path ./configs/super_mario_land/${world}-${level}_env_blackwhite_wo_game_text_config.yaml \
      --interaction_model_config_path ./model_configs/interaction_model_config_${interaction_model}.yaml \
      --log_dir ./logs_evaluation/marioland-blackwhite-world${world}-level${level}-start${progress_token}-episode${steps}-input${max_history}-ram${text_info}-map${with_game_area}/model_${interaction_model} \
      --steps $steps \
      --max_history $max_history \
      --with_game_area $with_game_area \
      --with_text_info $text_info \
      --image_resize_ratio 8 \
      --game_state "$game_state_path" \
      --game_world $world --game_level $level \
      --use_old_prompt 1 --use_old_tick 0 \
      --epsilon_noise $epsilon_noise \
      --run_count $run_count
    echo "Completed run ${run_count} (World ${world}, Level ${level}, progress ${progress_token})"
}

# Loop over 10 world levels
set -- $WORLD_LEVELS
while [ $# -ge 2 ]; do
    world=$1
    level=$2
    shift 2
    echo "[INFO] Starting world ${world} level ${level} (64 runs)"
    eval_pids=()
    for run_count in {1..64}; do
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
            sleep 1
        done
        run_evaluation $run_count $world $level &
        eval_pids+=($!)
    done
    for pid in "${eval_pids[@]}"; do
        wait "$pid"
    done
    echo "[INFO] All 64 runs completed for World ${world}, Level ${level}!"
done

echo "All 10 world levels completed!"

if kill "${VLLM_PID}" >/dev/null 2>&1; then
  wait "${VLLM_PID}" || true
fi
