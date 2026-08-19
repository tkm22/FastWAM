#!/usr/bin/env python3
"""Fit a real LIBERO Asym-FastWAM projection artifact.

This is intentionally an offline-only program: it decodes complete
episodes, applies the *same* horizontal camera concat + 224x448 resize/crop as
``RobotVideoDataset``, and loads a frozen Wan VAE only to obtain source latent
tokens.  Runtime pixel FastWAM never imports or loads this VAE.

Every candidate clip uses observation indices ``start + [0,4,...,32]`` and
must fit completely inside one episode.  By default the tool uses every valid
clip.  ``--clips-per-task`` reproduces task-balanced sampling without
replacement, such as the 40 x 2,500 clips used for the 100k artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision.transforms.functional as transforms_F
from tqdm import tqdm

from asymflow.color import OklabColorEncoder
from asymflow.projection import (
    ProjectionArtifact,
    fit_orthogonal_procrustes,
    scale_from_projected_energy,
)
from asymflow.video_packing import patchify_first_frame, patchify_future_tubes
from fastwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDatasetMetadata
from fastwam.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames, get_safe_default_codec
from fastwam.models.wan22.helpers.io import load_state_dict
from fastwam.models.wan22.helpers.state_dict_converters import wan_video_vae_state_dict_converter
from fastwam.models.wan22.wan_video_vae import WanVideoVAE38


FIRST_DIM, FUTURE_DIM, LATENT_DIM = 3 * 32 * 32, 3 * 4 * 32 * 32, 48 * 2 * 2


def _select_valid_clips(
    roots: list[Path],
    clips_per_task: int | None,
    seed: int,
) -> tuple[list[tuple[Path, int, int, list[int]]], dict[str, int], str]:
    """Select full stride-4 clip starts, optionally balanced by task.

    A candidate is ``(episode, start)`` with a full 33-observation horizon;
    no endpoint replication is used. Optional sampling is directly over clips
    rather than hierarchically over episodes.
    """
    candidates: dict[str, list[tuple[Path, int, int, int]]] = defaultdict(list)
    for root in roots:
        for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
            episode = json.loads(line)
            task = episode["tasks"][0]
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            candidates[task].extend(
                (root, episode_index, length, start)
                for start in range(max(0, length - 32))
            )

    selected: list[tuple[Path, int, int, int]] = []
    task_counts = {}
    for task, pool in sorted(candidates.items()):
        if clips_per_task is not None and len(pool) < clips_per_task:
            raise ValueError(
                f"task {task!r} has only {len(pool)} valid clips, requested "
                f"{clips_per_task}"
            )
        if clips_per_task is None:
            task_selected = pool
        else:
            task_hash = int(hashlib.sha1(task.encode()).hexdigest()[:8], 16)
            order = torch.randperm(
                len(pool),
                generator=torch.Generator().manual_seed(seed + task_hash),
            ).tolist()
            task_selected = [pool[i] for i in order[:clips_per_task]]
        selected.extend(task_selected)
        task_counts[task] = len(task_selected)

    by_episode: dict[tuple[Path, int, int], list[int]] = defaultdict(list)
    for root, episode_index, length, start in selected:
        by_episode[(root, episode_index, length)].append(start)
    grouped = [
        (root, episode_index, length, sorted(starts))
        for (root, episode_index, length), starts in sorted(
            by_episode.items(), key=lambda item: (str(item[0][0]), item[0][1])
        )
    ]
    selection_text = "\n".join(
        f"{root}|{episode_index}|{start}"
        for root, episode_index, _, starts in grouped
        for start in starts
    )
    return grouped, task_counts, hashlib.sha256(selection_text.encode()).hexdigest()


class EpisodeReader:
    def __init__(self, root: Path):
        self.root = root
        self.meta = LeRobotDatasetMetadata(repo_id=str(root), root=root)
        if len(self.meta.video_keys) != 2:
            raise ValueError(
                "LIBERO two-camera fitting expects exactly two video keys, "
                f"got {self.meta.video_keys}"
            )
        self.resize = ResizeSmallestSideAspectPreserving({"img_h": 224, "img_w": 448})
        self.crop = CenterCrop({"img_h": 224, "img_w": 448})
        self.normalize = Normalize({"mean": .5, "std": .5})
        self.backend = get_safe_default_codec()

    def read(self, episode: int, length: int) -> torch.Tensor:
        timestamps = [i / self.meta.fps for i in range(length)]
        cams = []
        for key in self.meta.video_keys:
            path = self.root / self.meta.get_video_file_path(episode, key)
            frames = decode_video_frames(
                path, timestamps, 1e-4, self.backend
            ).squeeze(0)
            # BaseLerobotDataset._get_image followed by the configured
            # ToTensor transform: decoder float -> uint8 -> float [0,1].
            frames = (frames * 255).to(torch.uint8).to(torch.float32) / 255
            # FastWAMProcessor applies torchvision.transforms.Resize([224,224])
            # independently to each 512x512 camera before RobotVideoDataset
            # concatenates the cameras.  Resize defaults to bilinear.
            cams.append(
                transforms_F.resize(
                    frames,
                    [224, 224],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )
            )
        # [T,3,224,448], exact training camera preprocessing + concat
        video = torch.cat(cams, dim=-1)
        # These are the same final transforms used by RobotVideoDataset.  At
        # 224x448 they are spatial no-ops, but retaining them keeps this tool
        # aligned if the training configuration is later parameterized.
        video = self.crop(self.resize(video))
        # [3,T,224,448], RGB [-1,1]
        return self.normalize(video).permute(1, 0, 2, 3)


def _clips(video: torch.Tensor, starts: list[int], batch_size: int):
    """Yield full-horizon stride-4 clips at explicitly selected starts."""
    base = torch.arange(0, 33, 4)
    for offset in range(0, len(starts), batch_size):
        clip_starts = torch.tensor(starts[offset:offset + batch_size])
        ids = clip_starts[:, None] + base[None]
        yield video[:, ids].permute(1, 0, 2, 3, 4).contiguous()


def _latent_patches(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Wan VAE: [B,48,3,14,28] -> first [B,98,192], future [B,2*98,192].
    first = z[:, :, 0].unfold(2, 2, 2).unfold(3, 2, 2)
    first = first.permute(0, 2, 3, 1, 4, 5).reshape(z.shape[0], 98, LATENT_DIM)
    future = z[:, :, 1:].unfold(3, 2, 2).unfold(4, 2, 2)
    future = future.permute(0, 2, 3, 4, 1, 5, 6).reshape(z.shape[0], (z.shape[2] - 1) * 98, LATENT_DIM)
    return first, future


def _load_vae(
    path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> WanVideoVAE38:
    vae = WanVideoVAE38()
    state_dict = load_state_dict(str(path), torch_dtype=dtype)
    vae.load_state_dict(
        wan_video_vae_state_dict_converter(state_dict),
        strict=True,
    )
    return vae.to(device=device, dtype=dtype).eval().requires_grad_(False)


def _init_distributed(args) -> tuple[int, int]:
    """Use torchrun ranks when present; otherwise retain single-GPU behavior."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        args.device = torch.device(f"cuda:{local_rank}")
    return rank, world_size


def _all_reduce_sum(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _fit(args, rank: int, world_size: int):
    """Fit one full-horizon artifact across torchrun ranks.

    Each rank decodes a disjoint subset of episodes and accumulates only fixed
    ``X.T @ Z`` / energy sufficient statistics. NCCL sums them before rank 0
    runs the two small SVDs and writes the artifact.
    """
    grouped, task_counts, selection_hash = _select_valid_clips(
        args.dataset_roots, args.clips_per_task, args.seed
    )
    local_groups = grouped[rank::world_size]
    total_clips = sum(task_counts.values())
    if args.clips_per_task is None:
        fit_label = "all-valid"
        sampling = (
            "all episodes; all valid clip starts; observation_stride=4; "
            "no endpoint replication"
        )
    else:
        fit_label = "task-balanced random"
        sampling = (
            "all episodes; task-balanced uniform valid clip starts; "
            "observation_stride=4; no endpoint replication"
        )
    if rank == 0:
        print(
            f"{fit_label} fitting: {len(task_counts)} tasks, "
            f"{total_clips} full-horizon clips, {len(grouped)} episodes"
        )

    raw_color = OklabColorEncoder(mean=(0, 0, 0), std=(1, 1, 1)).to(args.device)
    stats_sum = torch.zeros(3, dtype=torch.float64, device=args.device)
    stats_sq = torch.zeros(3, dtype=torch.float64, device=args.device)
    stats_n = torch.zeros((), dtype=torch.float64, device=args.device)
    readers = {}
    for root, episode_index, length, starts in tqdm(
        local_groups, desc=f"rank {rank} Oklab stats", disable=rank != 0
    ):
        reader = readers.setdefault(root, EpisodeReader(root))
        video = reader.read(episode_index, length)
        for rgb in _clips(video, starts, args.vae_batch):
            lab = raw_color.encode(rgb.to(args.device, torch.float32)).double()
            stats_sum += lab.sum((0, 2, 3, 4))
            stats_sq += lab.square().sum((0, 2, 3, 4))
            stats_n += lab.numel() // 3
    _all_reduce_sum(stats_sum, world_size)
    _all_reduce_sum(stats_sq, world_size)
    _all_reduce_sum(stats_n, world_size)
    mean = stats_sum / stats_n
    std = (stats_sq / stats_n - mean.square()).clamp_min(1e-12).sqrt()
    color = OklabColorEncoder(mean=mean.tolist(), std=std.tolist()).to(args.device)

    vae = _load_vae(args.vae_path, args.device, args.dtype)
    first_cross = torch.zeros((FIRST_DIM, LATENT_DIM), dtype=torch.float64, device=args.device)
    future_cross = torch.zeros((FUTURE_DIM, LATENT_DIM), dtype=torch.float64, device=args.device)
    readers = {}
    for root, episode_index, length, starts in tqdm(
        local_groups, desc=f"rank {rank} Procrustes", disable=rank != 0
    ):
        reader = readers.setdefault(root, EpisodeReader(root))
        video = reader.read(episode_index, length)
        for rgb in _clips(video, starts, args.vae_batch):
            pixel = color.encode(rgb.to(args.device, torch.float32))
            with torch.inference_mode():
                z = vae.model.encode(rgb.to(args.device, args.dtype), vae.scale)
            xp = patchify_first_frame(pixel[:, :, 0]).reshape(-1, FIRST_DIM)
            xf = patchify_future_tubes(pixel[:, :, 1:]).reshape(-1, FUTURE_DIM)
            zp, zf = _latent_patches(z)
            first_cross += xp.double().T @ zp.reshape(-1, LATENT_DIM).double()
            future_cross += xf.double().T @ zf.reshape(-1, LATENT_DIM).double()
    _all_reduce_sum(first_cross, world_size)
    _all_reduce_sum(future_cross, world_size)

    if rank == 0:
        A_first = fit_orthogonal_procrustes(first_cross).float()
        A_future = fit_orthogonal_procrustes(future_cross).float()
    else:
        A_first = torch.empty((FIRST_DIM, LATENT_DIM), dtype=torch.float32, device=args.device)
        A_future = torch.empty((FUTURE_DIM, LATENT_DIM), dtype=torch.float32, device=args.device)
    if world_size > 1:
        dist.broadcast(A_first, src=0)
        dist.broadcast(A_future, src=0)

    p1 = torch.zeros((), dtype=torch.float64, device=args.device)
    z1 = torch.zeros((), dtype=torch.float64, device=args.device)
    p4 = torch.zeros((), dtype=torch.float64, device=args.device)
    z4 = torch.zeros((), dtype=torch.float64, device=args.device)
    readers = {}
    for root, episode_index, length, starts in tqdm(
        local_groups, desc=f"rank {rank} scale", disable=rank != 0
    ):
        reader = readers.setdefault(root, EpisodeReader(root))
        video = reader.read(episode_index, length)
        for rgb in _clips(video, starts, args.vae_batch):
            pixel = color.encode(rgb.to(args.device, torch.float32))
            with torch.inference_mode():
                z = vae.model.encode(rgb.to(args.device, args.dtype), vae.scale)
            xp = patchify_first_frame(pixel[:, :, 0]).reshape(-1, FIRST_DIM)
            xf = patchify_future_tubes(pixel[:, :, 1:]).reshape(-1, FUTURE_DIM)
            zp, zf = _latent_patches(z)
            p1 += (xp.float() @ A_first).double().square().sum()
            z1 += zp.reshape(-1, LATENT_DIM).double().square().sum()
            p4 += (xf.float() @ A_future).double().square().sum()
            z4 += zf.reshape(-1, LATENT_DIM).double().square().sum()
    for value in (p1, z1, p4, z4):
        _all_reduce_sum(value, world_size)

    if rank == 0:
        s_first = scale_from_projected_energy(p1.float(), z1.float()).cpu()
        s_future = scale_from_projected_energy(p4.float(), z4.float()).cpu()
        metadata = {
            "dataset_roots": [str(x) for x in args.dataset_roots],
            "sampling": sampling,
            "total_clips": total_clips,
            "task_clip_counts": task_counts,
            "selection_sha256": selection_hash,
            "selected_episodes": len(grouped),
            "pixel_space": "oklab",
            "oklab_dtype": "float32",
            "vae_dtype": str(args.dtype).removeprefix("torch."),
            "cross_gram_and_svd_dtype": "float64",
            "pixel_patch_size": 32,
            "future_tube_frames": 4,
            "first_pixel_dim": FIRST_DIM,
            "future_pixel_dim": FUTURE_DIM,
            "source_latent_token_dim": LATENT_DIM,
        }
        if args.clips_per_task is not None:
            metadata["clips_per_task"] = args.clips_per_task
            metadata["selection_seed"] = args.seed
        ProjectionArtifact(
            A_first.cpu(), A_future.cpu(), s_first, s_future,
            mean.float().cpu(), std.float().cpu(), metadata,
        ).save(args.output)
        print(
            f"saved {args.output}: A_first={tuple(A_first.shape)} "
            f"s_first={s_first:.6g}; A_future={tuple(A_future.shape)} "
            f"s_future={s_future:.6g}"
        )
    if world_size > 1:
        dist.barrier()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-roots", nargs="+", type=Path, required=True)
    p.add_argument(
        "--vae-path",
        type=Path,
        default=Path(
            "checkpoints/DiffSynth-Studio/"
            "Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Dated output artifact path, for example artifacts/YYYY-MM-DD_stride4_<sampling>.pt.",
    )
    p.add_argument(
        "--clips-per-task",
        type=int,
        default=None,
        help=(
            "Sample this many valid clips per task without replacement. "
            "Omit to fit every valid clip from every episode."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vae-batch", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = p.parse_args()
    args.device = torch.device(args.device)
    args.dtype = (
        torch.bfloat16 if args.dtype == "bf16" else torch.float32
    )
    rank, world_size = _init_distributed(args)
    try:
        _fit(args, rank, world_size)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
