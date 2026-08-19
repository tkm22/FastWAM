"""Orthogonal Procrustes projection fitting and portable artifact I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class ProjectionArtifact:
    """CPU serializable inputs required by an asymmetric pixel model."""

    A_first: torch.Tensor
    A_future: torch.Tensor
    scale_first: torch.Tensor
    scale_future: torch.Tensor
    oklab_mean: torch.Tensor
    oklab_std: torch.Tensor
    metadata: dict[str, Any]

    def state_dict(self) -> dict[str, Any]:
        return {
            "A_first": self.A_first.cpu(),
            "A_future": self.A_future.cpu(),
            "scale_first": self.scale_first.cpu(),
            "scale_future": self.scale_future.cpu(),
            "oklab_mean": self.oklab_mean.cpu(),
            "oklab_std": self.oklab_std.cpu(),
            "metadata": dict(self.metadata),
        }

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> "ProjectionArtifact":
        d = torch.load(path, map_location=map_location, weights_only=False)
        return cls(**d)


def validate_projection_artifact(
    artifact: ProjectionArtifact,
    *,
    require_task_balanced_100k: bool = False,
) -> None:
    """Fail closed when a pixel model is paired with an incompatible A fit."""
    expected = {"A_first": (3072, 192), "A_future": (12288, 192)}
    for name, shape in expected.items():
        value = getattr(artifact, name)
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
        gram = value.float().T @ value.float()
        error = (gram - torch.eye(shape[1], dtype=gram.dtype)).abs().max()
        if error > 5e-4:
            raise ValueError(f"{name} is not column-orthonormal (max error {error:.3e})")
    for name in ("scale_first", "scale_future"):
        value = torch.as_tensor(getattr(artifact, name))
        if value.numel() != 1 or not torch.isfinite(value).all() or not bool((value > 0).all()):
            raise ValueError(f"{name} must be one finite positive value")
    for name in ("oklab_mean", "oklab_std"):
        value = torch.as_tensor(getattr(artifact, name))
        if tuple(value.shape) != (3,) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain three finite channel values")
    if not bool((torch.as_tensor(artifact.oklab_std) > 0).all()):
        raise ValueError("oklab_std must be positive")
    if require_task_balanced_100k:
        metadata = artifact.metadata
        if metadata.get("clips_per_task") != 2500 or metadata.get("total_clips") != 100000:
            raise ValueError("artifact is not the required 40-task, 100k task-balanced fit")
        sampling = str(metadata.get("sampling", ""))
        if "task-balanced" not in sampling or "observation_stride=4" not in sampling:
            raise ValueError("artifact metadata does not describe task-balanced stride-4 sampling")


def fit_orthogonal_procrustes(cross_gram: torch.Tensor) -> torch.Tensor:
    """Return A[D_pixel,D_latent] maximizing ``tr(A.T @ cross_gram)``.

    ``cross_gram`` is ``X.T @ Z`` accumulated over paired pixel/latent tokens.
    Reduced SVD avoids ever allocating a D_pixel² covariance.
    """
    u, _, vh = torch.linalg.svd(cross_gram, full_matrices=False)
    return (u @ vh).contiguous()


def scale_from_projected_energy(projected_sq_sum: torch.Tensor, latent_sq_sum: torch.Tensor) -> torch.Tensor:
    """Return AsymFlow calibration s=||A.T X||_F / ||Z||_F."""
    return torch.sqrt(
        projected_sq_sum
        / latent_sq_sum.clamp_min(torch.finfo(torch.float32).eps)
    )
