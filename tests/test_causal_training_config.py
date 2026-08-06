from pathlib import Path
import math

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_CONFIG = (
    REPO_ROOT
    / "scripts/config/vlajepa_cotrain_droid_ssv2_causal_adaln_kl_west1.yaml"
)
POSTTRAIN_CONFIG = (
    REPO_ROOT
    / "scripts/config/vlajepa_libero_v3_causal_adaln_kl_west1.yaml"
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _effective_route_weights(config: dict, *, robot: bool) -> dict[str, float]:
    model = config["framework"]
    vj2 = model["vj2_model"]
    latent = model["latent_alignment"]
    route_scale = config.get("trainer", {}).get("loss_scale", {}).get(
        "vla" if robot else "vlm",
        1.0,
    )
    weights = {
        "wm": route_scale
        * (
            vj2["vla_wm_loss_weight"]
            if robot
            else vj2["vlm_wm_loss_weight"]
        ),
        "dyn_kl": route_scale
        * latent["kl_weight"]
        * latent["dynamics_scale"],
        "rep_kl": route_scale
        * latent["kl_weight"]
        * latent["representation_scale"],
    }
    if robot:
        weights["action"] = route_scale
    return weights


def test_pretrain_loss_and_batch_contract() -> None:
    config = _load(PRETRAIN_CONFIG)

    expected_robot = {
        "wm": 0.2,
        "dyn_kl": 0.05,
        "rep_kl": 0.005,
        "action": 1.0,
    }
    expected_video = {
        "wm": 0.2,
        "dyn_kl": 0.05,
        "rep_kl": 0.005,
    }
    actual_robot = _effective_route_weights(config, robot=True)
    actual_video = _effective_route_weights(config, robot=False)
    assert actual_robot.keys() == expected_robot.keys()
    assert actual_video.keys() == expected_video.keys()
    for name, expected in expected_robot.items():
        assert math.isclose(actual_robot[name], expected)
    for name, expected in expected_video.items():
        assert math.isclose(actual_video[name], expected)

    accumulation = config["trainer"]["gradient_accumulation_steps"]
    assert accumulation == 2
    assert config["datasets"]["vla_data"]["per_device_batch_size"] == 8
    assert config["datasets"]["video_data"]["per_device_batch_size"] == 8
    assert 8 * 8 * accumulation == 128
    assert (8 + 8) * 8 * accumulation == 256


def test_causal_media_contract_is_direct_256() -> None:
    for path in (PRETRAIN_CONFIG, POSTTRAIN_CONFIG):
        config = _load(path)
        qwen = config["framework"]["qwenvl"]
        vla_data = config["datasets"]["vla_data"]
        assert qwen["preprocessed_visual_inputs"] is True
        assert (qwen["visual_resized_height"], qwen["visual_resized_width"]) == (
            256,
            256,
        )
        assert vla_data["resolution_size"] == 256
        assert vla_data["video_resolution_size"] == 256

    pretrain = _load(PRETRAIN_CONFIG)
    assert pretrain["datasets"]["video_data"]["resolution_size"] == 256
    assert pretrain["datasets"]["video_data"]["video_resolution_size"] == 256


def test_repeated_diffusion_and_posttrain_batch_contract() -> None:
    for path in (PRETRAIN_CONFIG, POSTTRAIN_CONFIG):
        config = _load(path)
        assert (
            config["framework"]["action_model"]["repeated_diffusion_steps"] == 8
        )

    posttrain = _load(POSTTRAIN_CONFIG)
    accumulation = posttrain["trainer"]["gradient_accumulation_steps"]
    per_device = posttrain["datasets"]["vla_data"]["per_device_batch_size"]
    assert per_device * 8 * accumulation == 256
