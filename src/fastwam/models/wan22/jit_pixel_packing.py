from __future__ import annotations

import torch
from einops import rearrange


def patchify_first_frame(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Pack [B, 3, H, W] RGB frames into non-overlapping spatial patches."""
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"`x` must be [B,3,H,W], got {tuple(x.shape)}")
    if patch_size <= 0 or x.shape[-2] % patch_size or x.shape[-1] % patch_size:
        raise ValueError(
            f"H/W must be divisible by positive patch_size={patch_size}, "
            f"got {tuple(x.shape[-2:])}"
        )
    return rearrange(
        x,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=patch_size,
        pw=patch_size,
    )


def unpatchify_first_frame(
    tokens: torch.Tensor,
    *,
    height: int,
    width: int,
    patch_size: int,
) -> torch.Tensor:
    """Restore packed RGB spatial tokens to [B, 3, H, W]."""
    if tokens.ndim != 3:
        raise ValueError(f"`tokens` must be [B,N,D], got {tuple(tokens.shape)}")
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"height/width must be divisible by patch_size={patch_size}, got {height}, {width}"
        )
    expected_tokens = (height // patch_size) * (width // patch_size)
    expected_dim = 3 * patch_size * patch_size
    if tokens.shape[1:] != (expected_tokens, expected_dim):
        raise ValueError(
            f"Packed token shape must be [B,{expected_tokens},{expected_dim}], got {tuple(tokens.shape)}"
        )
    return rearrange(
        tokens,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=height // patch_size,
        w=width // patch_size,
        c=3,
        ph=patch_size,
        pw=patch_size,
    )


def patchify_future_tubes(
    x: torch.Tensor,
    patch_size: int,
    tube_size: int,
) -> torch.Tensor:
    """Pack [B, 3, T, H, W] RGB video into non-overlapping pixel tubes."""
    if x.ndim != 5 or x.shape[1] != 3:
        raise ValueError(f"`x` must be [B,3,T,H,W], got {tuple(x.shape)}")
    if patch_size <= 0 or tube_size <= 0:
        raise ValueError(
            f"patch_size and tube_size must be positive, got {patch_size}, {tube_size}"
        )
    if x.shape[2] % tube_size or x.shape[-2] % patch_size or x.shape[-1] % patch_size:
        raise ValueError(
            "T/H/W must be divisible by tube_size/patch_size, got "
            f"shape={tuple(x.shape)}, tube_size={tube_size}, patch_size={patch_size}"
        )
    return rearrange(
        x,
        "b c (f pt) (h ph) (w pw) -> b (f h w) (c pt ph pw)",
        pt=tube_size,
        ph=patch_size,
        pw=patch_size,
    )


def unpatchify_future_tubes(
    tokens: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
    patch_size: int,
    tube_size: int,
) -> torch.Tensor:
    """Restore packed RGB tube tokens to [B, 3, T, H, W]."""
    if tokens.ndim != 3:
        raise ValueError(f"`tokens` must be [B,N,D], got {tuple(tokens.shape)}")
    if frames % tube_size or height % patch_size or width % patch_size:
        raise ValueError(
            "frames/height/width must be divisible by tube_size/patch_size, got "
            f"{frames}, {height}, {width}"
        )
    expected_tokens = (frames // tube_size) * (height // patch_size) * (width // patch_size)
    expected_dim = 3 * tube_size * patch_size * patch_size
    if tokens.shape[1:] != (expected_tokens, expected_dim):
        raise ValueError(
            "Packed token shape mismatch: expected "
            f"[B,{expected_tokens},{expected_dim}], got {tuple(tokens.shape)}"
        )
    return rearrange(
        tokens,
        "b (f h w) (c pt ph pw) -> b c (f pt) (h ph) (w pw)",
        f=frames // tube_size,
        h=height // patch_size,
        w=width // patch_size,
        c=3,
        pt=tube_size,
        ph=patch_size,
        pw=patch_size,
    )
