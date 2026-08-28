from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
from PIL import Image


def pil_frames_to_video_tensor(frames: Sequence[Image.Image]) -> torch.Tensor:
    if len(frames) == 0:
        raise ValueError("`frames` must be non-empty.")

    frame_tensors = []
    for frame in frames:
        arr = np.array(frame.convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3, H, W]
        frame_tensors.append(x)
    return torch.stack(frame_tensors, dim=1)  # [3, T, H, W]


def _gaussian_kernel_2d(kernel_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    return kernel_2d.repeat(channels, 1, 1, 1)


def video_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, eps: float = 1e-8) -> float:
    """
    Compute average PSNR over all frames.
    Expects `pred` and `target` in shape [3, T, H, W] with values in [0, 1].
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")

    pred = pred.float()
    target = target.float()
    mse = (pred - target).pow(2).mean(dim=(0, 2, 3))  # [T]
    psnr = 10.0 * torch.log10((data_range * data_range) / (mse + eps))
    return float(psnr.mean().item())


def video_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Compute average SSIM over all frames.
    Expects `pred` and `target` in shape [3, T, H, W] with values in [0, 1].
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if pred.ndim != 4 or pred.shape[0] != 3:
        raise ValueError(f"Expected [3, T, H, W], got {tuple(pred.shape)}")
    if kernel_size % 2 == 0:
        raise ValueError("`kernel_size` must be odd.")

    pred = pred.float().permute(1, 0, 2, 3).contiguous()  # [T, 3, H, W]
    target = target.float().permute(1, 0, 2, 3).contiguous()  # [T, 3, H, W]
    channels = pred.shape[1]
    kernel = _gaussian_kernel_2d(
        kernel_size=kernel_size,
        sigma=sigma,
        channels=channels,
        device=pred.device,
        dtype=pred.dtype,
    )

    pad = kernel_size // 2
    mu_x = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=pad, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean().item())


@torch.no_grad()
def video_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_model: torch.nn.Module,
) -> float:
    """Compute frame-averaged LPIPS for videos in ``[3,T,H,W]``/``[0,1]``."""
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )
    if pred.ndim != 4 or pred.shape[0] != 3:
        raise ValueError(f"Expected [3, T, H, W], got {tuple(pred.shape)}")
    try:
        metric_device = next(lpips_model.parameters()).device
    except StopIteration:
        metric_device = pred.device
    pred_images = pred.float().permute(1, 0, 2, 3).to(metric_device)
    target_images = target.float().permute(1, 0, 2, 3).to(metric_device)
    distance = lpips_model(pred_images * 2.0 - 1.0, target_images * 2.0 - 1.0)
    return float(distance.float().mean().cpu().item())


def video_haar_wavelet_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Return fixed one-level orthonormal Haar band errors.

    Inputs use ``[3,T,H,W]`` and ``[0,1]``. Spatial bands are computed per
    frame. Temporal low/high bands are also reported when at least two future
    frames are available. The protocol intentionally has no learned weights.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )
    if pred.ndim != 4 or pred.shape[0] != 3:
        raise ValueError(f"Expected [3, T, H, W], got {tuple(pred.shape)}")

    pred = pred.float()
    target = target.float()
    height = pred.shape[-2] - pred.shape[-2] % 2
    width = pred.shape[-1] - pred.shape[-1] % 2
    if height == 0 or width == 0:
        raise ValueError("Haar metrics require spatial dimensions of at least 2")

    def spatial_bands(video: torch.Tensor) -> tuple[torch.Tensor, ...]:
        video = video[..., :height, :width]
        x00 = video[..., 0::2, 0::2]
        x01 = video[..., 0::2, 1::2]
        x10 = video[..., 1::2, 0::2]
        x11 = video[..., 1::2, 1::2]
        return (
            (x00 + x01 + x10 + x11) * 0.5,
            (x00 - x01 + x10 - x11) * 0.5,
            (x00 + x01 - x10 - x11) * 0.5,
            (x00 - x01 - x10 + x11) * 0.5,
        )

    pred_bands = spatial_bands(pred)
    target_bands = spatial_bands(target)
    names = ("ll", "lh", "hl", "hh")
    metrics = {
        f"wavelet_{name}_mse": float((p - t).square().mean().item())
        for name, p, t in zip(names, pred_bands, target_bands)
    }
    metrics["wavelet_high_mse"] = float(
        np.mean([metrics["wavelet_lh_mse"], metrics["wavelet_hl_mse"], metrics["wavelet_hh_mse"]])
    )

    temporal_frames = pred.shape[1] - pred.shape[1] % 2
    if temporal_frames >= 2:
        pred_even = pred[:, :temporal_frames:2]
        pred_odd = pred[:, 1:temporal_frames:2]
        target_even = target[:, :temporal_frames:2]
        target_odd = target[:, 1:temporal_frames:2]
        scale = 2.0**-0.5
        metrics["wavelet_temporal_low_mse"] = float(
            (
                (pred_even + pred_odd) * scale
                - (target_even + target_odd) * scale
            )
            .square()
            .mean()
            .item()
        )
        metrics["wavelet_temporal_high_mse"] = float(
            (
                (pred_even - pred_odd) * scale
                - (target_even - target_odd) * scale
            )
            .square()
            .mean()
            .item()
        )
    return metrics


def load_i3d_fvd_detector(
    checkpoint: str | Path,
    device: str | torch.device,
) -> torch.jit.ScriptModule:
    """Load the Kinetics-400 I3D TorchScript detector used by StyleGAN-V FVD."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"I3D FVD detector not found: {checkpoint}")
    detector = torch.jit.load(str(checkpoint), map_location=device).eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    return detector


@torch.no_grad()
def video_i3d_features(
    videos: torch.Tensor,
    detector: torch.nn.Module,
    *,
    num_frames: int = 16,
) -> torch.Tensor:
    """Extract I3D-400 features after deterministic nearest time resampling.

    ``videos`` must be ``[B,3,T,H,W]`` in ``[0,1]``.  The released I3D
    TorchScript graph requires at least 16 input frames; short robot rollout
    clips are therefore sampled at evenly spaced nearest frame indices.
    """
    if videos.ndim != 5 or videos.shape[1] != 3 or videos.shape[2] < 1:
        raise ValueError(f"Expected [B,3,T,H,W] with T >= 1, got {tuple(videos.shape)}")
    if num_frames < 16:
        raise ValueError(f"I3D requires at least 16 frames, got {num_frames}")
    frame_indices = torch.linspace(
        0,
        videos.shape[2] - 1,
        num_frames,
        device=videos.device,
    ).round().long()
    videos = videos.index_select(2, frame_indices)
    videos = videos.float().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    try:
        detector_device = next(detector.parameters()).device
    except StopIteration:
        detector_device = videos.device
    features = detector(
        videos.to(detector_device),
        rescale=True,
        resize=True,
        return_features=True,
    )
    return features.float().cpu()


def frechet_feature_distance(
    pred_features: torch.Tensor | np.ndarray,
    target_features: torch.Tensor | np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    """Compute the Fréchet distance between two feature distributions."""
    pred = np.asarray(pred_features, dtype=np.float64)
    target = np.asarray(target_features, dtype=np.float64)
    if pred.ndim != 2 or target.ndim != 2 or pred.shape[1] != target.shape[1]:
        raise ValueError(
            "Expected feature matrices [N,D] with equal D, got "
            f"pred={pred.shape} target={target.shape}"
        )
    if pred.shape[0] < 2 or target.shape[0] < 2:
        raise ValueError("Fréchet distance requires at least two samples per distribution")
    pred_mean = pred.mean(axis=0)
    target_mean = target.mean(axis=0)
    pred_cov = np.cov(pred, rowvar=False)
    target_cov = np.cov(target, rowvar=False)
    covariance_mean = scipy.linalg.sqrtm(pred_cov @ target_cov)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(pred_cov.shape[0]) * eps
        covariance_mean = scipy.linalg.sqrtm((pred_cov + offset) @ (target_cov + offset))
    if np.iscomplexobj(covariance_mean):
        imaginary_max = float(np.max(np.abs(covariance_mean.imag)))
        if imaginary_max > 1e-3:
            raise ValueError(
                "Fréchet covariance product has a large imaginary component: "
                f"{imaginary_max}"
            )
        covariance_mean = covariance_mean.real
    mean_distance = np.square(pred_mean - target_mean).sum()
    covariance_distance = np.trace(pred_cov + target_cov - 2.0 * covariance_mean)
    # Finite-sample singular covariance matrices can produce a tiny negative
    # round-off value even though the analytic distance is non-negative.
    return max(float(np.real(mean_distance + covariance_distance)), 0.0)
