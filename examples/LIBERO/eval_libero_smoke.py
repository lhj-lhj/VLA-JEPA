from __future__ import annotations

import argparse
import collections
import logging
import pathlib
import time

import imageio
import numpy as np
from libero.libero import benchmark

from examples.LIBERO.eval_libero import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    M1Inference,
    _binarize_gripper_open,
    _get_libero_env,
    _quat2axisangle,
    short_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal VLA-JEPA + LIBERO evaluation smoke test")
    parser.add_argument("--pretrained-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15123)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--category-value", default=None)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-policy-steps", type=int, default=20)
    parser.add_argument("--video-out-path", default="results/libero_smoke")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--unnorm-key", default="franka")
    parser.add_argument("--with-state", default="true")
    parser.add_argument("--resize-size", type=int, nargs=2, default=[224, 224])
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    np.random.seed(args.seed)

    out_dir = pathlib.Path(args.video_out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_cls = benchmark.get_benchmark_dict()[args.task_suite_name]
    suite = bench_cls() if args.category_value is None else bench_cls(category_value=args.category_value)
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    model = M1Inference(
        policy_ckpt_path=args.pretrained_path,
        unnorm_key=args.unnorm_key,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )

    done = False
    replay_images = []
    full_actions = []
    try:
        logging.info("task_suite=%s task_id=%s episode_idx=%s", args.task_suite_name, args.task_id, args.episode_idx)
        logging.info("task_description=%s", task_description)
        logging.info("max_policy_steps=%s", args.max_policy_steps)

        model.reset(task_description=task_description)
        env.reset()
        obs = env.set_init_state(initial_states[args.episode_idx])

        t = 0
        step = 0
        while t < args.num_steps_wait + args.max_policy_steps:
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            replay_images.append(img)
            state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            )

            obs_input = {
                "images": [img, wrist_img],
                "task_description": task_description,
                "step": step,
            }
            if args.with_state == "true":
                obs_input["state"] = np.expand_dims(state, axis=0)

            start = time.time()
            response = model.step(**obs_input)
            logging.info("policy_step=%s latency_sec=%.3f", step, time.time() - start)

            raw_action = response["raw_action"]
            world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
            rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
            open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
            gripper = _binarize_gripper_open(open_gripper)
            if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and gripper.size == 1):
                raise ValueError(
                    f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                    f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                )
            delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)
            full_actions.append(delta_action)

            obs, _, done, _ = env.step(delta_action.tolist())
            if done:
                logging.info("episode ended with success at policy_step=%s", step)
                break
            t += 1
            step += 1

        suffix = "success" if done else "smoke"
        task_segment = short_name(task_description.replace(" ", "_"))
        video_path = out_dir / f"smoke_{task_segment}_episode{args.episode_idx}_{suffix}.mp4"
        action_path = out_dir / f"smoke_{task_segment}_episode{args.episode_idx}_{suffix}.npy"
        if replay_images:
            imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)
        if full_actions:
            np.save(action_path, np.stack(full_actions))
        logging.info(
            "SMOKE_DONE success=%s policy_steps=%s video=%s actions=%s",
            done,
            len(full_actions),
            video_path,
            action_path,
        )
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
