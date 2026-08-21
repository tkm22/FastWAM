from __future__ import annotations

import torch


def sample_jit_sigma(
    batch_size: int,
    *,
    device: torch.device | str,
    p_mean: float = -0.8,
    p_std: float = 0.8,
) -> torch.Tensor:
    """Sample JiT noise levels using sigma = 1 - sigmoid(N(P_mean, P_std))."""
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}")
    if p_std < 0:
        raise ValueError(f"`p_std` must be non-negative, got {p_std}")
    data_t = torch.sigmoid(
        torch.randn((batch_size,), device=device, dtype=torch.float32) * float(p_std)
        + float(p_mean)
    )
    return 1.0 - data_t


def add_jit_noise(
    clean: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    if clean.shape != noise.shape:
        raise ValueError(
            f"`clean` and `noise` must have identical shapes, got {clean.shape} and {noise.shape}"
        )
    sigma_view = _batch_scalar_view(sigma, clean)
    return (1.0 - sigma_view) * clean + sigma_view * noise


def jit_velocity_from_x0(
    noisy: torch.Tensor,
    x0: torch.Tensor,
    sigma: torch.Tensor,
    *,
    t_eps: float = 0.05,
) -> torch.Tensor:
    """Convert clean-data prediction to FastWAM's noise-minus-data velocity."""
    if noisy.shape != x0.shape:
        raise ValueError(
            f"`noisy` and `x0` must have identical shapes, got {noisy.shape} and {x0.shape}"
        )
    if t_eps <= 0:
        raise ValueError(f"`t_eps` must be positive, got {t_eps}")
    denominator = _batch_scalar_view(sigma, noisy).clamp_min(float(t_eps))
    return (noisy - x0) / denominator


def _batch_scalar_view(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        value = value.reshape(1)
    if value.ndim != 1 or value.shape[0] not in (1, reference.shape[0]):
        raise ValueError(
            f"Batch scalar must be [1] or [B={reference.shape[0]}], got {tuple(value.shape)}"
        )
    return value.to(device=reference.device, dtype=reference.dtype).view(
        -1, *([1] * (reference.ndim - 1))
    )
