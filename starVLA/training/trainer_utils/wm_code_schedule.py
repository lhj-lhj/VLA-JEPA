"""World-model code-source scheduling helpers."""


def use_prior_world_model_code(
    *,
    optimization_step: int,
    posterior_only_steps: int,
    alternate_after_warmup: bool,
) -> bool:
    """Select prior codes on even steps after a posterior-only warmup."""

    if optimization_step <= 0:
        raise ValueError("optimization_step must be positive.")
    if posterior_only_steps < 0:
        raise ValueError("posterior_only_steps must be non-negative.")
    return (
        bool(alternate_after_warmup)
        and optimization_step > posterior_only_steps
        and optimization_step % 2 == 0
    )
