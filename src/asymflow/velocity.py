"""Asymmetric Flow calibration and implicit P=A A^T velocity recovery."""

from __future__ import annotations

import torch


def asymflow_calibration(sigma: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional video form of upstream ``asymflow_calibration``.

    Returns calibrated sigma ``k(sigma)*sigma`` and ``k(sigma)``, where
    ``k=1/[s+(1-s)sigma]``.  The upstream method receives timesteps in model
    units; FastWAM has already converted them to ``sigma in [0,1]``.
    """
    sigma = sigma.float()
    s = scale.to(device=sigma.device, dtype=torch.float32)
    while s.ndim < sigma.ndim:
        s = s.unsqueeze(0)
    k = 1.0 / (s + (1.0 - s) * sigma)
    return sigma * k, k


def project_to_subspace(x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Implicitly apply P=A A^T to the final feature dimension."""
    A = A.to(device=x.device, dtype=x.dtype)
    return (x @ A) @ A.transpose(0, 1)


def asymflow_velocity(
    u_asym: torch.Tensor,
    x_sigma: torch.Tensor,
    sigma: torch.Tensor,
    scale: torch.Tensor,
    A: torch.Tensor,
    sigma_min: float = 1e-6,
) -> torch.Tensor:
    """Recover full ``epsilon-x0`` velocity from the calibrated AsymFlow output.

    With scale calibration, the network target is
    ``u_A_cal = P epsilon - x0 / s`` (not ``P epsilon - x0`` unless
    ``s == 1``).  This implements AsymFlow Eq. 17 while applying
    ``P = A A.T`` implicitly. ``sigma`` is [B] and features are last.
    """
    output_dtype = x_sigma.dtype
    with torch.autocast(device_type=x_sigma.device.type, enabled=False):
        u_asym = u_asym.float()
        x_sigma = x_sigma.float()
        A = A.to(device=x_sigma.device, dtype=torch.float32)
        sigma = sigma.to(device=x_sigma.device, dtype=torch.float32)
        view = (sigma.shape[0],) + (1,) * (x_sigma.ndim - 1)
        sigma_v = sigma.reshape(view).clamp_min(float(sigma_min))
        _, k = asymflow_calibration(sigma, scale)
        k_v = k.reshape(view)
        s = scale.to(device=x_sigma.device, dtype=torch.float32)
        p_u = project_to_subspace(u_asym, A)
        p_x = project_to_subspace(x_sigma, A)
        u_perp = u_asym - p_u
        x_perp = x_sigma - p_x
        u_p = s * k_v * p_u + (1.0 - s * k_v) * p_x / sigma_v
        velocity = u_p + (x_perp + s * u_perp) / sigma_v
    return velocity.to(output_dtype)
