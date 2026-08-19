"""Training-only AsymFlow helpers for the pixel FastWAM path."""

from __future__ import annotations

import torch

from .video_packing import (
    patchify_future_tubes,
    unpatchify_future_tubes,
)


def calc_shifted_signal_ratio(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """AsymFlow's smooth VR/LPIPS time gate."""
    alpha = 1.0 - sigma
    alpha_sq = alpha.square()
    return alpha_sq / (alpha_sq + (float(shift) * sigma).square())


def sample_logit_normal_sigma(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    shift: float = 17.0,
) -> torch.Tensor:
    """Sample the shifted logit-normal sigma distribution used by AsymFlow."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    u = torch.sigmoid(torch.randn(batch_size, device=device, dtype=torch.float32))
    sigma = float(shift) * u / (1.0 + (float(shift) - 1.0) * u)
    return sigma.to(dtype=dtype)


def wan_future_latent_patches(z: torch.Tensor) -> torch.Tensor:
    """Pack a Wan 9-frame VAE latent into its eight future 192-D tokens."""
    if z.ndim != 5 or z.shape[1] != 48 or z.shape[2] < 2:
        raise ValueError("Wan VAE latent must be [B,48,T,H,W] with T >= 2")
    future = z[:, :, 1:].unfold(3, 2, 2).unfold(4, 2, 2)
    return future.permute(0, 2, 3, 4, 1, 5, 6).reshape(
        z.shape[0], -1, z.shape[1] * 2 * 2
    )


def lift_future_latents(
    latent_tokens: torch.Tensor,
    a_future: torch.Tensor,
    scale_future: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Lift frozen Wan VAE tokens into the calibrated pixel-flow subspace."""
    packed = latent_tokens.float() @ a_future.float().T
    packed = packed * scale_future.float()
    return unpatchify_future_tubes(
        packed.to(dtype=latent_tokens.dtype), frames, height, width
    )


def compute_vr_coefficient(
    full_x0: torch.Tensor,
    pred_x0: torch.Tensor,
    low_x0: torch.Tensor,
    ref_low_x0: torch.Tensor,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the upstream channel-wise AsymFlow VR coefficient.

    The source implementation's ``patchify(..., pack_channels=False)`` keeps
    each color channel separate and averages only within one patch.  For video,
    one patch is a ``4*32*32`` tube, so the returned tensors have shapes
    ``coefficient=[B,3,1,F,H,W]`` and ``low_diff=[B,3,4096,F,H,W]``.
    """
    low_diff = patchify_future_tubes(low_x0 - ref_low_x0, pack_channels=False)
    full_diff = patchify_future_tubes(
        full_x0 - pred_x0.detach(), pack_channels=False
    )
    numerator = (full_diff * low_diff).mean(dim=2, keepdim=True)
    denominator = low_diff.square().mean(dim=2, keepdim=True).clamp_min(eps)
    return (numerator / denominator).clamp_(0.0, 1.0), low_diff


def build_vr_target(
    full_x0: torch.Tensor,
    coefficient: torch.Tensor,
    low_diff: torch.Tensor,
    signal_ratio: torch.Tensor,
) -> torch.Tensor:
    """Apply the upstream VR control variate after video tube unpatchifying."""
    correction = unpatchify_future_tubes(
        coefficient * low_diff,
        frames=full_x0.shape[2],
        height=full_x0.shape[3],
        width=full_x0.shape[4],
        packed_channels=False,
    )
    signal_ratio = signal_ratio.reshape(-1, 1, 1, 1, 1).to(full_x0)
    return full_x0 - (1.0 - signal_ratio) * correction


def build_vr_lpips_gate(
    coefficient: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Match upstream channel-RMS VR coefficient gate in video coordinates."""
    patch_elements = 4 * 32 * 32
    channel_rms = coefficient.expand(-1, -1, patch_elements, -1, -1, -1)
    channel_rms = channel_rms.square().mean(dim=1, keepdim=True).sqrt()
    return unpatchify_future_tubes(
        channel_rms,
        frames=frames,
        height=height,
        width=width,
        packed_channels=False,
    )
