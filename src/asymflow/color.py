"""Differentiable sRGB <-> Oklab conversion with dataset affine statistics."""

from __future__ import annotations

import warnings

import torch
from torch import nn


# These match artifacts/2026-08-04_stride4_bal100k.pt, the
# projection artifact selected by the FastWAM model configs.  Production models
# still load these values from their artifact; the defaults keep direct uses of
# this transform aligned with the configured LIBERO pixel space.
DEFAULT_LIBERO_OKLAB_MEAN = (0.5660378933, 0.0081772227, 0.0162523333)
DEFAULT_LIBERO_OKLAB_STD = (0.2106077075, 0.0115914568, 0.0160523932)


class OklabColorEncoder(nn.Module):
    """Map normalized sRGB ``[-1, 1]`` tensors to standardized Oklab.

    The affine statistics are deliberately buffers, not learnable parameters:
    fitting them is part of the offline projection artifact and they must be
    identical in A fitting, training and inference.
    """

    def __init__(
        self,
        use_affine_norm: bool = True,
        mean=DEFAULT_LIBERO_OKLAB_MEAN,
        std=DEFAULT_LIBERO_OKLAB_STD,
    ):
        super().__init__()
        self.use_affine_norm = bool(use_affine_norm)
        self.register_buffer("lrgb_to_lms", torch.tensor([
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ], dtype=torch.float32))
        self.register_buffer("lms_to_oklab", torch.tensor([
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ], dtype=torch.float32))
        self.register_buffer("oklab_to_lms", torch.linalg.inv(self.lms_to_oklab.float()))
        self.register_buffer("lms_to_lrgb", torch.linalg.inv(self.lrgb_to_lms.float()))
        if self.use_affine_norm:
            self.register_buffer("affine_mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("affine_std", torch.tensor(std, dtype=torch.float32))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Read pre-Oklab-fix training states without reverting the fix.

        The first pixel-training checkpoints serialized an earlier colour
        transform as ``rgb_to_lms``/``lms_to_rgb`` plus ``mean``/``std``.
        Those values are not merely renamed: they precede the corrected
        linear-RGB Oklab transform and the statistics used to fit the current
        projection artifact.  On resume, retain this module's configured
        current buffers while loading every trainable model and optimizer
        state exactly from the checkpoint.
        """
        legacy_marker = prefix + "rgb_to_lms"
        if legacy_marker in state_dict:
            legacy_names = (
                "mean",
                "std",
                "rgb_to_lms",
                "lms_to_oklab",
                "oklab_to_lms",
                "lms_to_rgb",
            )
            current_names = (
                "lrgb_to_lms",
                "lms_to_oklab",
                "oklab_to_lms",
                "lms_to_lrgb",
            )
            if self.use_affine_norm:
                current_names += ("affine_mean", "affine_std")

            for name in legacy_names:
                state_dict.pop(prefix + name, None)
            for name in current_names:
                state_dict[prefix + name] = getattr(self, name).detach().clone()
            warnings.warn(
                "Loading a pre-Oklab-fix checkpoint: retaining the current "
                "linear-RGB Oklab transform and projection-artifact statistics.",
                UserWarning,
                stacklevel=2,
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.lrgb_to_lms.dtype

    @staticmethod
    def srgb_to_lrgb(srgb: torch.Tensor) -> torch.Tensor:
        """Upstream AsymFlow sRGB-to-linear-RGB conversion."""
        a = 0.055
        srgb = srgb.clamp(0, 1)
        return torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + a) / (1 + a)).pow(2.4))

    @staticmethod
    def lrgb_to_srgb(lrgb: torch.Tensor) -> torch.Tensor:
        """Upstream AsymFlow linear-RGB-to-sRGB conversion."""
        lrgb = lrgb.clamp(0, 1)
        a = 0.055
        return torch.where(
            lrgb <= 0.0031308,
            12.92 * lrgb,
            (1 + a) * lrgb.clamp_min(0.0031308).pow(1 / 2.4) - a,
        )

    def lrgb_to_oklab(self, lrgb: torch.Tensor) -> torch.Tensor:
        lms = self._linear(lrgb, self.lrgb_to_lms)
        return self._linear(lms.clamp_min(0).pow(1.0 / 3.0), self.lms_to_oklab)

    def oklab_to_lrgb(self, oklab: torch.Tensor) -> torch.Tensor:
        lms = self._linear(oklab, self.oklab_to_lms).pow(3)
        return self._linear(lms, self.lms_to_lrgb).clamp(0, 1)

    @staticmethod
    def _linear(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        # Color channels are always dimension 1 for both [B,C,H,W] and [B,C,T,H,W].
        return torch.einsum("ij,bj...->bi...", matrix.to(dtype=x.dtype), x)

    def encode(self, srgb_minus_one_to_one: torch.Tensor) -> torch.Tensor:
        oklab = self.lrgb_to_oklab(self.srgb_to_lrgb((srgb_minus_one_to_one + 1) * 0.5))
        if self.use_affine_norm:
            affine_shape = (1, 3) + (1,) * (oklab.ndim - 2)
            mean = self.affine_mean.to(oklab).reshape(affine_shape)
            std = self.affine_std.to(oklab).reshape(affine_shape)
            oklab = (oklab - mean) / std
        return oklab

    def decode(self, standardized_oklab: torch.Tensor) -> torch.Tensor:
        oklab = standardized_oklab
        if self.use_affine_norm:
            affine_shape = (1, 3) + (1,) * (standardized_oklab.ndim - 2)
            mean = self.affine_mean.to(standardized_oklab).reshape(affine_shape)
            std = self.affine_std.to(standardized_oklab).reshape(affine_shape)
            oklab = standardized_oklab * std + mean
        # Oklab states produced during denoising are not guaranteed to be in
        # the sRGB gamut.  The upstream AsymFlow encoder clamps *linear* RGB
        # before applying the sRGB transfer function.  Clamping only the
        # final display tensor leaves arbitrarily large intermediate values
        # which are fed back into the next denoising step.
        srgb = self.lrgb_to_srgb(self.oklab_to_lrgb(oklab))
        return srgb * 2 - 1

    forward = encode
