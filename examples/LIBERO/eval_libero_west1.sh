#!/bin/bash
set -euo pipefail

repo_root=/workspace/hongjia/Unified_Model/VLA-JEPA
libero_home=/workspace/hongjia/Unified_Model/LIBERO
sim_python=/workspace/hongjia/envs/libero/bin/python
vla_python=/workspace/hongjia/envs/vla-jepa/bin/python
checkpoint_root=/workspace/hongjia/Unified_Model/checkpoints/VLA-JEPA-official/LIBERO-west1
checkpoint=${checkpoint_root}/checkpoints/VLA-JEPA-LIBERO.pt

export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export LIBERO_HOME="${libero_home}"
export LIBERO_CONFIG_PATH="${libero_home}/libero"
export PYTHONPATH="${repo_root}:${libero_home}:${PYTHONPATH:-}"

cd "${repo_root}"

task_suites=(libero_10 libero_goal libero_object libero_spatial)
base_port=${BASE_PORT:-15083}
num_trials_per_task=${NUM_TRIALS_PER_TASK:-50}
with_state=${WITH_STATE:-true}
run_name=${RUN_NAME:-"official_libero_$(date +%Y%m%d_%H%M%S)"}
output_root=${OUTPUT_ROOT:-"${repo_root}/results/${run_name}"}
log_root=${LOG_ROOT:-"${repo_root}/logs/${run_name}"}
mkdir -p "${output_root}" "${log_root}"

server_pids=()
sim_pids=()
cleanup() {
    for pid in "${sim_pids[@]:-}" "${server_pids[@]:-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT INT TERM

for index in "${!task_suites[@]}"; do
    task_suite=${task_suites[$index]}
    gpu=$index
    port=$((base_port + index + 1))
    suite_output="${output_root}/${task_suite}"
    mkdir -p "${suite_output}"

    CUDA_VISIBLE_DEVICES=${gpu} "${vla_python}" deployment/model_server/server_policy.py \
        --ckpt_path "${checkpoint}" \
        --port "${port}" \
        --use_bf16 \
        --cuda 0 \
        >"${log_root}/${task_suite}_server.log" 2>&1 &
    server_pids+=("$!")

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${gpu} "${sim_python}" examples/LIBERO/eval_libero.py \
        --args.pretrained-path "${checkpoint}" \
        --args.host 127.0.0.1 \
        --args.port "${port}" \
        --args.task-suite-name "${task_suite}" \
        --args.num-trials-per-task "${num_trials_per_task}" \
        --args.video-out-path "${suite_output}" \
        --args.with-state "${with_state}" \
        >"${log_root}/${task_suite}_eval.log" 2>&1 &
    sim_pids+=("$!")
done

status=0
for pid in "${sim_pids[@]}"; do
    wait "${pid}" || status=$?
done
exit "${status}"
