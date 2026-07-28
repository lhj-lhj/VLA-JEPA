# VLA-JEPA Official Checkpoint LIBERO Evaluation — west1 (2026-07-28)

## Summary

This evaluation used the **official VLA-JEPA LIBERO checkpoint released by the
authors**, not a checkpoint trained in this repository fork.

- Evaluation date: 2026-07-28
- Server: Pyromind west1
- Repository commit: `f7751c6a99991db471096f3bdd3e6e0989976c38`
- Protocol: 4 suites × 10 tasks × 50 episodes = 2,000 rollouts
- Result: **97.9% overall success rate (1,958/2,000)**
- Paper result: **97.2% overall success rate**
- Difference from paper: **+0.7 percentage points**

All four evaluation processes exited with status `0`. The result was also
independently checked by counting the per-episode `Success: True/False` log
entries and the generated MP4 files.

## Checkpoint

The checkpoint came from the official Hugging Face release:

- Hugging Face repository: <https://huggingface.co/ginwind/VLA-JEPA>
- Release-relative path: `LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`
- Checkpoint filename: `VLA-JEPA-LIBERO.pt`
- File size: 6,163,579,855 bytes (approximately 5.74 GiB)

Path passed to both the policy server and evaluator on west1:

```text
/workspace/hongjia/Unified_Model/checkpoints/VLA-JEPA-official/LIBERO-west1/checkpoints/VLA-JEPA-LIBERO.pt
```

This path is a symbolic link to the downloaded official checkpoint:

```text
/workspace/hongjia/Unified_Model/checkpoints/VLA-JEPA-official/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
```

The `LIBERO-west1` directory provides west1-compatible configuration paths; the
weight file itself resolves directly to the official downloaded checkpoint.

## Evaluation protocol

The run used the repository's official evaluation implementation:

```text
examples/LIBERO/eval_libero.py
deployment/model_server/server_policy.py
```

Key settings:

- Suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
- Trials per task: 50
- Tasks per suite: 10
- Episodes per suite: 500
- Total episodes: 2,000
- Seed: 7
- State input: enabled (`--args.with-state true`)
- MuJoCo renderer: EGL
- Parallelism: four suites on GPUs 0–3
- Policy server ports: 15381–15384
- Policy inference: BF16

Each suite used the following evaluator arguments, with the suite name, port,
GPU, and output directory varied per process:

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<gpu> \
  /workspace/hongjia/envs/libero/bin/python \
  examples/LIBERO/eval_libero.py \
  --args.pretrained-path \
    /workspace/hongjia/Unified_Model/checkpoints/VLA-JEPA-official/LIBERO-west1/checkpoints/VLA-JEPA-LIBERO.pt \
  --args.host 127.0.0.1 \
  --args.port <port> \
  --args.task-suite-name <suite> \
  --args.num-trials-per-task 50 \
  --args.video-out-path <suite-output-directory> \
  --args.with-state true \
  --args.seed 7
```

## Results

| Suite | Successful | Failed | Total | This evaluation | Paper | Difference |
|---|---:|---:|---:|---:|---:|---:|
| LIBERO-Spatial | 491 | 9 | 500 | 98.2% | 96.2% | +2.0 pp |
| LIBERO-Object | 498 | 2 | 500 | 99.6% | 99.6% | 0.0 pp |
| LIBERO-Goal | 492 | 8 | 500 | 98.4% | 97.2% | +1.2 pp |
| LIBERO-10 | 477 | 23 | 500 | 95.4% | 95.8% | -0.4 pp |
| **Overall** | **1,958** | **42** | **2,000** | **97.9%** | **97.2%** | **+0.7 pp** |

The official checkpoint result is therefore reproduced within normal rollout
variation. Three suites match or exceed the paper result; LIBERO-10 is 0.4
percentage points below it, while the aggregate result is 0.7 percentage points
above the reported overall result.

## Verification

The following checks passed:

- Repository HEAD matched the recorded commit.
- All four suite processes returned exit status `0`.
- Each suite reported `Total episodes: 500`.
- Per-episode success and failure entries summed to 500 for every suite.
- Each suite produced exactly 500 MP4 files.
- Total output contained exactly 2,000 MP4 files.

At Python shutdown, completed suites printed ignored `EGL_NOT_INITIALIZED`
exceptions while destroying already-freed MuJoCo EGL contexts. These appeared
after the final metrics, did not change the exit status, and did not affect the
recorded rollouts.

## Artifacts on west1

Run identifier:

```text
official_libero_2000_f7751c6_20260728_02
```

Logs and launch record:

```text
/workspace/hongjia/Unified_Model/logs/vla_jepa_libero_official_2000_retry01
```

Rollout videos:

```text
/workspace/hongjia/Unified_Model/outputs/vla_jepa_libero_official_2000_retry01
```

Important files in the log directory:

```text
launch.sh
completed.done
exit_status.txt
libero_spatial.eval.log
libero_object.eval.log
libero_goal.eval.log
libero_10.eval.log
libero_spatial.server.log
libero_object.server.log
libero_goal.server.log
libero_10.server.log
```
