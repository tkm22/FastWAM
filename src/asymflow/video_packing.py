"""Generic non-overlapping pixel patch/tube packing helpers."""

from __future__ import annotations

import torch
from einops import rearrange


def patchify_first_frame(x: torch.Tensor, patch: int = 32) -> torch.Tensor:
    """[B,3,H,W] -> [B,H/p*W/p,3*p*p]."""
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)


def patchify_future_tubes(
    x: torch.Tensor,
    temporal: int = 4,
    patch: int = 32,
    *,
    pack_channels: bool = True,
) -> torch.Tensor:
    """Video adaptation of upstream AsymFlow ``patchify``.

    With ``pack_channels=True`` this produces Wan input/output tokens
    ``[B,F*H*W,3*t*p*p]``.  With ``False`` it retains RGB channels as upstream
    VR does: ``[B,3,t*p*p,F,H,W]``.
    """
    if pack_channels:
        return rearrange(
            x,
            "b c (f pt) (h ph) (w pw) -> b (f h w) (c pt ph pw)",
            pt=temporal,
            ph=patch,
            pw=patch,
        )
    return rearrange(
        x,
        "b c (f pt) (h ph) (w pw) -> b c (pt ph pw) f h w",
        pt=temporal,
        ph=patch,
        pw=patch,
    )


def unpatchify_future_tubes(
    tokens: torch.Tensor,
    frames: int,
    height: int,
    width: int,
    temporal: int = 4,
    patch: int = 32,
    *,
    packed_channels: bool = True,
) -> torch.Tensor:
    if not packed_channels:
        return rearrange(
            tokens,
            "b c (pt ph pw) f h w -> b c (f pt) (h ph) (w pw)",
            pt=temporal,
            ph=patch,
            pw=patch,
        )
    return rearrange(
        tokens,
        "b (f h w) (c pt ph pw) -> b c (f pt) (h ph) (w pw)",
        f=frames // temporal,
        h=height // patch,
        w=width // patch,
        pt=temporal,
        ph=patch,
        pw=patch,
    )
