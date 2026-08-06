import torch

from starVLA.model.modules.world_model.causal_adaln_predictor import (
    balanced_diagonal_gaussian_kl,
    CausalAdaLNWorldModel,
    GaussianCodeProjector,
    diagonal_gaussian_kl,
    split_video_into_tubelets,
)


def _tiny_world_model(*, use_activation_checkpointing=False):
    torch.manual_seed(7)
    return CausalAdaLNWorldModel(
        img_size=(32, 32),
        patch_size=16,
        embed_dim=16,
        predictor_embed_dim=16,
        latent_dim=4,
        code_tokens_per_step=2,
        depth=2,
        num_heads=4,
        use_activation_checkpointing=use_activation_checkpointing,
    )


def test_gaussian_projector_and_identical_kl():
    projector = GaussianCodeProjector(input_dim=12, latent_dim=4)
    inputs = torch.randn(2, 8, 12)
    mean, logvar = projector(inputs)

    assert mean.shape == (2, 8, 4)
    assert logvar.shape == (2, 8, 4)
    objective, raw = diagonal_gaussian_kl(
        mean,
        logvar,
        mean,
        logvar,
        free_bits=0.01,
    )
    torch.testing.assert_close(raw, torch.zeros_like(raw), atol=1.0e-6, rtol=0)
    torch.testing.assert_close(
        objective,
        torch.full_like(objective, 0.01),
        atol=1.0e-6,
        rtol=0,
    )


def test_tubelet_split_preserves_batch_view_and_time_order():
    batch_size, num_views, num_frames = 2, 2, 8
    videos = torch.arange(
        batch_size * num_views * num_frames,
        dtype=torch.float32,
    ).view(batch_size * num_views, num_frames, 1, 1, 1)
    tubelets = split_video_into_tubelets(
        videos,
        batch_size=batch_size,
        num_views=num_views,
        tubelet_size=2,
    )

    assert tubelets.shape == (16, 2, 1, 1, 1)
    expected = []
    for batch_index in range(batch_size):
        for view_index in range(num_views):
            base = (batch_index * num_views + view_index) * num_frames
            for tubelet_index in range(num_frames // 2):
                first = base + 2 * tubelet_index
                expected.append([first, first + 1])
    torch.testing.assert_close(
        tubelets[:, :, 0, 0, 0],
        torch.tensor(expected, dtype=torch.float32),
    )


def test_rollout_shape_and_future_code_causality():
    model = _tiny_world_model().eval()
    initial_state = torch.randn(2, 4, 16)
    codes = torch.randn(2, 3, 2, 4)

    baseline = model.rollout(initial_state, codes)
    changed_codes = codes.clone()
    changed_codes[:, 2, 0, 0] += 10.0
    changed = model.rollout(initial_state, changed_codes)

    assert baseline.shape == (2, 3, 4, 16)
    # A future code group cannot affect any earlier recursive prediction.
    torch.testing.assert_close(baseline[:, :2], changed[:, :2])
    assert not torch.allclose(baseline[:, 2], changed[:, 2])


def test_training_teacher_forcing_differs_from_inference_rollout():
    model = _tiny_world_model().eval()
    ground_truth_contexts = torch.randn(2, 3, 4, 16)
    codes = torch.randn(2, 3, 2, 4)

    teacher_forced = model.teacher_forced(ground_truth_contexts, codes)
    autoregressive = model.rollout(ground_truth_contexts[:, 0], codes)

    assert teacher_forced.shape == autoregressive.shape == (2, 3, 4, 16)
    # Step 1 sees the same GT initial state in both schedules.
    torch.testing.assert_close(
        teacher_forced[:, 0],
        autoregressive[:, 0],
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    # Step 2 uses GT_23 in teacher forcing but predicted_23 in rollout.
    assert not torch.allclose(teacher_forced[:, 1], autoregressive[:, 1])


def test_balanced_kl_splits_prior_and_posterior_gradients():
    posterior_mean = torch.randn(2, 3, 4, requires_grad=True)
    posterior_logvar = torch.randn(2, 3, 4, requires_grad=True)
    prior_mean = torch.randn(2, 3, 4, requires_grad=True)
    prior_logvar = torch.randn(2, 3, 4, requires_grad=True)

    _, dynamics_kl, representation_kl, _ = balanced_diagonal_gaussian_kl(
        posterior_mean,
        posterior_logvar,
        prior_mean,
        prior_logvar,
    )
    dynamics_kl.backward(retain_graph=True)
    assert posterior_mean.grad is None
    assert posterior_logvar.grad is None
    assert prior_mean.grad is not None
    assert prior_logvar.grad is not None

    prior_mean.grad = None
    prior_logvar.grad = None
    representation_kl.backward()
    assert posterior_mean.grad is not None
    assert posterior_logvar.grad is not None
    assert prior_mean.grad is None
    assert prior_logvar.grad is None


def test_wm_loss_reaches_code_condition():
    model = _tiny_world_model(use_activation_checkpointing=True).train()
    state = torch.randn(2, 4, 16)
    code = torch.randn(2, 2, 4, requires_grad=True)
    prediction = model(state, code)
    prediction.square().mean().backward()

    assert code.grad is not None
    assert torch.isfinite(code.grad).all()
    assert code.grad.abs().sum() > 0
