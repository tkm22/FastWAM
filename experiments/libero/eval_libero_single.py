import copy
import hashlib
import inspect
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import mujoco
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.utils.video_metrics import (
    frechet_feature_distance,
    load_i3d_fvd_detector,
    pil_frames_to_video_tensor,
    video_haar_wavelet_mse,
    video_i3d_features,
    video_lpips,
    video_psnr,
    video_ssim,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from libero.libero import benchmark
from action_ensembler import ActionEnsembler

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return

    # deprecated legacy checkpoint loading
    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy checkpoint payload must be dict, got: {type(payload)}")

    if "mot" in payload and hasattr(model, "mot"):
        missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
        logging.warning(
            "Loaded fallback `mot` state_dict with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        return

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    if state_dict is None and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        raise ValueError(f"Cannot parse legacy checkpoint keys from: {ckpt}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.warning(
        "Loaded fallback model state_dict with strict=False. Missing=%d Unexpected=%d",
        len(missing),
        len(unexpected),
    )


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _get_video_metric_horizon_steps(cfg: DictConfig) -> int:
    model_horizon_steps = int(cfg.data.train.num_frames) - 1
    configured = cfg.EVALUATION.get("video_metric_horizon_steps")
    horizon_steps = model_horizon_steps if configured is None else int(configured)
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    if horizon_steps < action_video_freq_ratio:
        raise ValueError(
            "EVALUATION.video_metric_horizon_steps must include at least one "
            f"future video frame, got {horizon_steps}."
        )
    if horizon_steps > model_horizon_steps:
        raise ValueError(
            "EVALUATION.video_metric_horizon_steps exceeds the model action "
            f"horizon: {horizon_steps} > {model_horizon_steps}."
        )
    if horizon_steps % action_video_freq_ratio != 0:
        raise ValueError(
            "EVALUATION.video_metric_horizon_steps must be divisible by "
            "data.train.action_video_freq_ratio, got "
            f"{horizon_steps} and {action_video_freq_ratio}."
        )
    return horizon_steps


def _get_video_metric_replan_indices(cfg: DictConfig) -> Optional[set[int]]:
    configured = cfg.EVALUATION.get("video_metric_replan_indices")
    if configured is None:
        return None
    indices = {int(index) for index in configured}
    if any(index < 0 for index in indices):
        raise ValueError("EVALUATION.video_metric_replan_indices must be non-negative.")
    return indices


def _compute_video_metrics_enabled(cfg: DictConfig) -> bool:
    configured = cfg.EVALUATION.get("compute_video_metrics")
    if configured is not None:
        return bool(configured)
    # Preserve older resolved configs and launch commands.
    return bool(cfg.EVALUATION.get("visualize_future_video", False))


def _save_prediction_videos_enabled(cfg: DictConfig) -> bool:
    configured = cfg.EVALUATION.get("save_prediction_videos")
    if configured is not None:
        return bool(configured)
    # Before save_prediction_videos existed, visualize_future_video implied
    # that prediction videos would be encoded.
    return bool(cfg.EVALUATION.get("visualize_future_video", False))


def _validate_video_metric_cfg(cfg: DictConfig) -> None:
    if not _compute_video_metrics_enabled(cfg):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.compute_video_metrics=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    metric_horizon_steps = _get_video_metric_horizon_steps(cfg)
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = metric_horizon_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    if len(pred_video) < keep_frames:
        raise ValueError(
            "`infer_joint` returned fewer video frames than the configured "
            f"metric horizon requires: {len(pred_video)} < {keep_frames}."
        )
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    metric_horizon_steps = _get_video_metric_horizon_steps(cfg)
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = metric_horizon_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _aligned_future_video_tensors(
    gt_frames: list[Any],
    pred_frames: list[Any],
    *,
    exclude_conditioning_frame: bool = True,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None, None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for future-video metrics: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    start = 1 if exclude_conditioning_frame else 0
    if len(gt_frames) <= start:
        return None, None

    aligned_gt = []
    aligned_pred = []
    for gt_frame, pred_frame in zip(gt_frames[start:], pred_frames[start:]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )
        aligned_gt.append(Image.fromarray(gt_image.astype(np.uint8)))
        aligned_pred.append(Image.fromarray(pred_image.astype(np.uint8)))

    return (
        pil_frames_to_video_tensor(aligned_gt),
        pil_frames_to_video_tensor(aligned_pred),
    )


def _compute_clip_metrics(
    gt_frames: list[Any],
    pred_frames: list[Any],
    *,
    lpips_model: Optional[torch.nn.Module] = None,
    exclude_conditioning_frame: bool = True,
) -> dict[str, float]:
    gt_video, pred_video = _aligned_future_video_tensors(
        gt_frames,
        pred_frames,
        exclude_conditioning_frame=exclude_conditioning_frame,
    )
    if gt_video is None or pred_video is None:
        return {}
    metrics = {
        "psnr": video_psnr(pred_video, gt_video),
        "ssim": video_ssim(pred_video, gt_video),
        "num_future_frames": float(pred_video.shape[1]),
    }
    metrics.update(video_haar_wavelet_mse(pred_video, gt_video))
    if lpips_model is not None:
        metrics["lpips"] = video_lpips(pred_video, gt_video, lpips_model)

    full_gt_video, full_pred_video = _aligned_future_video_tensors(
        gt_frames,
        pred_frames,
        exclude_conditioning_frame=False,
    )
    if (
        full_gt_video is not None
        and full_pred_video is not None
        and full_gt_video.shape[1] >= 2
    ):
        gt_boundary_delta = full_gt_video[:, 1] - full_gt_video[:, 0]
        pred_boundary_delta = full_pred_video[:, 1] - full_pred_video[:, 0]
        metrics["boundary_temporal_delta_mse"] = float(
            (pred_boundary_delta - gt_boundary_delta).square().mean().item()
        )
    return metrics


def _score_future_video_clip(
    clip: dict[str, Any],
    cfg: DictConfig,
    *,
    lpips_model: Optional[torch.nn.Module],
    fvd_detector: Optional[torch.nn.Module],
) -> dict[str, Any]:
    assert len(clip["gt_frames"]) == len(clip["pred_frames"]), (
        "GT/pred frame count mismatch before scoring: "
        f"len(gt_frames)={len(clip['gt_frames'])} "
        f"len(pred_frames)={len(clip['pred_frames'])} "
        f"replan={clip['replan_idx']}."
    )
    exclude_conditioning_frame = bool(
        cfg.EVALUATION.get("metrics_exclude_conditioning_frame", True)
    )
    clip_metrics = _compute_clip_metrics(
        clip["gt_frames"],
        clip["pred_frames"],
        lpips_model=lpips_model,
        exclude_conditioning_frame=exclude_conditioning_frame,
    )
    clip["metrics"] = clip_metrics
    if clip_metrics and fvd_detector is not None:
        gt_video, pred_video = _aligned_future_video_tensors(
            clip["gt_frames"],
            clip["pred_frames"],
            exclude_conditioning_frame=exclude_conditioning_frame,
        )
        if gt_video is not None and pred_video is not None:
            features = video_i3d_features(
                torch.stack([gt_video, pred_video]),
                fvd_detector,
                num_frames=int(cfg.EVALUATION.get("gfvd_num_frames", 16)),
            )
            clip["_gt_fvd_feature"] = features[0]
            clip["_pred_fvd_feature"] = features[1]
    return clip


def _rollout_metric_gt_frames(
    env,
    initial_frame: Any,
    action_chunk: np.ndarray,
    cfg: DictConfig,
) -> tuple[list[Any], Optional[int]]:
    """Execute one full action chunk in a dedicated video-metric rollout."""
    capture_steps = _get_future_frame_capture_steps(cfg)
    metric_horizon_steps = capture_steps[-1]
    if len(action_chunk) < metric_horizon_steps:
        raise ValueError(
            "Predicted action chunk is shorter than the video metric horizon: "
            f"{len(action_chunk)} < {metric_horizon_steps}."
        )

    gt_frames = [initial_frame]
    capture_step_set = set(capture_steps[1:])
    last_frame = initial_frame
    terminal_step = None
    done = False
    for step_idx in range(1, metric_horizon_steps + 1):
        if not done:
            obs, _, done, _ = env.step(action_chunk[step_idx - 1])
            last_frame = get_libero_image(obs)
            if done:
                terminal_step = step_idx
        if step_idx in capture_step_set:
            gt_frames.append(last_frame)

    assert len(gt_frames) == len(capture_steps), (
        "Metric rollout did not capture every requested future frame: "
        f"captured={len(gt_frames)} expected={len(capture_steps)}."
    )
    return gt_frames, terminal_step


def _capture_libero_branch_state(env) -> dict[str, Any]:
    """Capture enough MuJoCo and robosuite state to discard a metric branch."""
    inner_env = env.env
    sim_model = inner_env.sim.model._model
    sim_data = inner_env.sim.data._data
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    integration_state = np.empty(
        mujoco.mj_stateSize(sim_model, state_spec), dtype=np.float64
    )
    mujoco.mj_getState(
        sim_model,
        sim_data,
        integration_state,
        state_spec,
    )

    robot = inner_env.robots[0]
    controller = robot.controller
    controller_state = {
        key: copy.deepcopy(value)
        for key, value in controller.__dict__.items()
        if key != "sim"
    }
    robot_state = {
        key: copy.deepcopy(value)
        for key, value in robot.__dict__.items()
        if key.startswith("recent_") or key == "torques"
    }
    observable_state = {
        name: copy.deepcopy(
            {
                key: value
                for key, value in observable.__dict__.items()
                if key not in {"_sensor", "_corrupter", "_delayer"}
            }
        )
        for name, observable in inner_env._observables.items()
    }
    mutable_model_state = {
        name: np.array(getattr(inner_env.sim.model, name), copy=True)
        for name in ("body_pos", "body_quat", "geom_rgba", "site_rgba")
    }
    object_properties = {
        name: copy.deepcopy(obj.object_properties)
        for name, obj in {
            **inner_env.objects_dict,
            **inner_env.fixtures_dict,
        }.items()
        if hasattr(obj, "object_properties")
    }
    return {
        "integration_state": integration_state,
        "timestep": int(inner_env.timestep),
        "cur_time": float(inner_env.cur_time),
        "done": bool(inner_env.done),
        "controller": controller_state,
        "robot": robot_state,
        "obs_cache": copy.deepcopy(inner_env._obs_cache),
        "observables": observable_state,
        "mutable_model_state": mutable_model_state,
        "object_properties": object_properties,
        "numpy_random_state": np.random.get_state(),
        "python_random_state": random.getstate(),
    }


def _restore_libero_branch_state(env, snapshot: dict[str, Any]) -> None:
    """Restore a state captured by :func:`_capture_libero_branch_state`."""
    inner_env = env.env
    sim_model = inner_env.sim.model._model
    sim_data = inner_env.sim.data._data
    mujoco.mj_setState(
        sim_model,
        sim_data,
        snapshot["integration_state"],
        mujoco.mjtState.mjSTATE_INTEGRATION,
    )
    inner_env.timestep = snapshot["timestep"]
    inner_env.cur_time = snapshot["cur_time"]
    inner_env.done = snapshot["done"]
    inner_env._obs_cache = copy.deepcopy(snapshot["obs_cache"])
    for name, value in snapshot["mutable_model_state"].items():
        getattr(inner_env.sim.model, name)[:] = value
    for name, value in snapshot["object_properties"].items():
        inner_env.get_object(name).object_properties = copy.deepcopy(value)

    robot = inner_env.robots[0]
    controller = robot.controller
    controller_sim = controller.sim
    controller.__dict__.clear()
    controller.__dict__.update(copy.deepcopy(snapshot["controller"]))
    controller.sim = controller_sim
    for key, value in snapshot["robot"].items():
        setattr(robot, key, copy.deepcopy(value))

    for name, state in snapshot["observables"].items():
        observable = inner_env._observables[name]
        callables = {
            key: getattr(observable, key)
            for key in ("_sensor", "_corrupter", "_delayer")
        }
        observable.__dict__.clear()
        observable.__dict__.update(copy.deepcopy(state))
        observable.__dict__.update(callables)

    np.random.set_state(snapshot["numpy_random_state"])
    random.setstate(snapshot["python_random_state"])


def _run_video_metric_branch(
    env,
    *,
    initial_frame: Any,
    action_chunk: np.ndarray,
    predicted_frames: list[Image.Image],
    replan_idx: int,
    cfg: DictConfig,
    lpips_model: Optional[torch.nn.Module],
    fvd_detector: Optional[torch.nn.Module],
) -> dict[str, Any]:
    """Save state, execute the metric horizon for GT frames, then restore."""
    snapshot = _capture_libero_branch_state(env)
    try:
        gt_frames, terminal_step = _rollout_metric_gt_frames(
            env=env,
            initial_frame=initial_frame,
            action_chunk=action_chunk,
            cfg=cfg,
        )
    finally:
        _restore_libero_branch_state(env, snapshot)

    clip = {
        "replan_idx": replan_idx,
        "gt_frames": gt_frames,
        "pred_frames": predicted_frames,
        "metric_terminal_step": terminal_step,
    }
    return _score_future_video_clip(
        clip,
        cfg,
        lpips_model=lpips_model,
        fvd_detector=fvd_detector,
    )


def _mean_metric_dicts(values: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for value in values for key in value})
    return {
        key: float(np.mean([value[key] for value in values if key in value]))
        for key in keys
        if any(key in value for value in values)
    }


def _build_lpips_metric(cfg: DictConfig, device: str) -> Optional[torch.nn.Module]:
    if not bool(cfg.EVALUATION.get("compute_lpips", True)):
        return None
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "EVALUATION.compute_lpips=true requires the installed `lpips` package"
        ) from exc
    return lpips.LPIPS(
        net=str(cfg.EVALUATION.get("lpips_net", "vgg")),
        spatial=False,
        eval_mode=True,
        pnet_tune=False,
    ).to(device=device, dtype=torch.float32).eval().requires_grad_(False)


def _resolve_fvd_detector_path(cfg: DictConfig) -> Path:
    configured = Path(str(cfg.EVALUATION.get("fvd_detector_path")))
    if not configured.is_absolute():
        configured = project_root / configured
    return configured.resolve()


def _build_fvd_detector(
    cfg: DictConfig,
    device: str,
) -> Optional[torch.jit.ScriptModule]:
    if not bool(cfg.EVALUATION.get("compute_gfvd", True)):
        return None
    return load_i3d_fvd_detector(_resolve_fvd_detector_path(cfg), device)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--short")
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=root
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "diff_sha256": None}
    return {
        "commit": commit,
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff).hexdigest() if status else None,
    }


def _inference_provenance(cfg: DictConfig) -> dict[str, Any]:
    return {
        "schedule": str(
            cfg.EVALUATION.get("inference_schedule", "logit_normal")
        ),
        "schedule_shift": (
            None
            if cfg.EVALUATION.get("inference_schedule_shift", 17.0) is None
            else float(cfg.EVALUATION.get("inference_schedule_shift", 17.0))
        ),
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "num_steps": int(cfg.EVALUATION.get("num_inference_steps")),
    }


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]]]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)

    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "inference_schedule": str(
            cfg.EVALUATION.get("inference_schedule", "logit_normal")
        ),
        "inference_schedule_shift": (
            None
            if cfg.EVALUATION.get("inference_schedule_shift", 17.0) is None
            else float(cfg.EVALUATION.get("inference_schedule_shift", 17.0))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    compute_video_metrics = _compute_video_metrics_enabled(cfg)
    predicted_future_frames = None
    if compute_video_metrics:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    with torch.no_grad():
        if compute_video_metrics:
            if "test_action_with_infer_action" in inspect.signature(
                model.infer_joint
            ).parameters:
                infer_kwargs["test_action_with_infer_action"] = False
            pred = model.infer_joint(**infer_kwargs)
            predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
        else:
            pred = model.infer_action(**infer_kwargs)
    action = pred["action"]  # [T, D]

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action, imgs, predicted_future_frames


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    lpips_model: Optional[torch.nn.Module],
    fvd_detector: Optional[torch.nn.Module],
) -> tuple[bool, list, list[dict[str, Any]], dict[str, float]]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    save_rollout_videos = bool(cfg.EVALUATION.get("save_rollout_videos", True))
    compute_video_metrics = _compute_video_metrics_enabled(cfg)
    metric_replan_indices = _get_video_metric_replan_indices(cfg)

    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_metrics: list[dict[str, float]] = []
    pending_actions: list[list[float]] = []
    current_replan_idx = -1

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            action_chunk, imgs, predicted_future_frames = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            current_replan_idx += 1
            evaluate_this_replan = (
                metric_replan_indices is None
                or current_replan_idx in metric_replan_indices
            )
            if compute_video_metrics and evaluate_this_replan:
                if predicted_future_frames is None:
                    raise ValueError(
                        "Video metrics require infer_joint video "
                        "predictions."
                    )
                metric_clip = _run_video_metric_branch(
                    env,
                    initial_frame=imgs.copy(),
                    action_chunk=action_chunk,
                    predicted_frames=predicted_future_frames,
                    replan_idx=current_replan_idx,
                    cfg=cfg,
                    lpips_model=lpips_model,
                    fvd_detector=fvd_detector,
                )
                predicted_future_video_clips.append(metric_clip)
                if metric_clip.get("metrics"):
                    episode_future_clip_metrics.append(metric_clip["metrics"])
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            if save_rollout_videos:
                replay_images.append(imgs.copy())
        elif save_rollout_videos:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        obs, _, done, _ = env.step(pending_actions.pop(0))
        if done:
            break
        t += 1
    pbar.close()

    return (
        bool(done),
        replay_images,
        predicted_future_video_clips,
        _mean_metric_dicts(episode_future_clip_metrics),
    )


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    lpips_model: Optional[torch.nn.Module],
    fvd_detector: Optional[torch.nn.Module],
) -> dict:
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    compute_video_metrics = _compute_video_metrics_enabled(cfg)
    save_prediction_videos = _save_prediction_videos_enabled(cfg)
    save_rollout_videos = bool(cfg.EVALUATION.get("save_rollout_videos", True))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }
    if compute_video_metrics:
        results["episode_future_video_metrics"] = []
        results["future_video_clip_metrics"] = []
        results["prediction_video_files"] = []
        results["future_video_psnr_mean"] = None
        results["future_video_metrics_mean"] = {}
    results["rollout_video_files"] = []
    gt_fvd_features = []
    pred_fvd_features = []

    trial_indices_cfg = cfg.EVALUATION.get("trial_indices")
    if trial_indices_cfg is None:
        trial_indices = list(range(int(cfg.EVALUATION.num_trials)))
    else:
        trial_indices = [int(index) for index in trial_indices_cfg]
        if not trial_indices:
            raise ValueError("EVALUATION.trial_indices must be non-empty when set")
        if len(set(trial_indices)) != len(trial_indices):
            raise ValueError("EVALUATION.trial_indices must not contain duplicates")
        if min(trial_indices) < 0 or max(trial_indices) >= len(initial_states):
            raise ValueError(
                "EVALUATION.trial_indices are outside the available initial "
                f"state range [0, {len(initial_states) - 1}]"
            )
    results["trial_indices"] = trial_indices

    for trial_idx in trial_indices:
        success, replay_images, predicted_future_video_clips, episode_metrics = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            lpips_model=lpips_model,
            fvd_detector=fvd_detector,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if compute_video_metrics:
            results["episode_future_video_metrics"].append(
                {"trial_index": trial_idx, **episode_metrics}
            )

        if save_rollout_videos:
            rollout_video_path = save_rollout_video(
                video_dir,
                replay_images,
                f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                success=success,
                task_description=task_description,
            )
            results["rollout_video_files"].append(rollout_video_path)
        if compute_video_metrics:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    if "_gt_fvd_feature" in clip:
                        gt_fvd_features.append(clip["_gt_fvd_feature"])
                        pred_fvd_features.append(clip["_pred_fvd_feature"])
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    if save_prediction_videos and bool(
                        cfg.EVALUATION.get(
                            "save_prediction_clip_videos", True
                        )
                    ):
                        prediction_video_path = save_prediction_video(
                            predicted_video_dir,
                            clip["gt_frames"],
                            clip["pred_frames"],
                            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                            clip["replan_idx"],
                            success=success,
                            task_description=task_description,
                        )
                        results["prediction_video_files"].append(
                            prediction_video_path
                        )
                    results["future_video_clip_metrics"].append(
                        {
                            "trial_index": trial_idx,
                            "replan_index": int(clip["replan_idx"]),
                            "metric_terminal_step": clip.get(
                                "metric_terminal_step"
                            ),
                            **clip.get("metrics", {}),
                        }
                    )
                if save_prediction_videos:
                    prediction_video_path = save_prediction_video(
                        predicted_video_dir,
                        all_gt_frames,
                        all_pred_frames,
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        "all",
                        success=success,
                        task_description=task_description,
                    )
                    results["prediction_video_files"].append(
                        prediction_video_path
                    )

    if compute_video_metrics:
        metric_rows = [
            {
                key: value
                for key, value in row.items()
                if key != "trial_index"
            }
            for row in results["episode_future_video_metrics"]
        ]
        results["future_video_metrics_mean"] = _mean_metric_dicts(metric_rows)
        results["future_video_psnr_mean"] = results[
            "future_video_metrics_mean"
        ].get("psnr")
        results["gfvd_num_clips"] = len(gt_fvd_features)
        if len(gt_fvd_features) >= 2:
            stacked_gt_fvd_features = torch.stack(gt_fvd_features)
            stacked_pred_fvd_features = torch.stack(pred_fvd_features)
            results["future_video_metrics_mean"]["gfvd"] = (
                frechet_feature_distance(
                    stacked_pred_fvd_features,
                    stacked_gt_fvd_features,
                )
            )
            results["_gt_fvd_features"] = stacked_gt_fvd_features.numpy()
            results["_pred_fvd_features"] = stacked_pred_fvd_features.numpy()
        elif fvd_detector is not None:
            logging.warning(
                "gFVD requires at least two valid clips; collected %s",
                len(gt_fvd_features),
            )
        metric_horizon_steps = _get_video_metric_horizon_steps(cfg)
        results["video_metric_terminal_padded_clips"] = sum(
            row.get("metric_terminal_step") is not None
            and int(row["metric_terminal_step"]) < metric_horizon_steps
            for row in results["future_video_clip_metrics"]
        )
    close = getattr(env, "close", None)
    if callable(close):
        close()
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_video_metric_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel_test.sh for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()
    lpips_model = (
        _build_lpips_metric(cfg, model_device)
        if _compute_video_metrics_enabled(cfg)
        else None
    )
    fvd_detector = (
        _build_fvd_detector(cfg, model_device)
        if _compute_video_metrics_enabled(cfg)
        else None
    )

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = (
        local_log_dir
        / "resolved_configs"
        / (
            f"{cfg.EVALUATION.task_suite_name}_task{int(cfg.EVALUATION.task_id)}"
            f"_gpu{int(cfg.gpu_id)}.yaml"
        )
    )
    resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        config=cfg,
        f=str(resolved_config_path),
        resolve=True,
    )
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    if bool(cfg.EVALUATION.get("save_rollout_videos", True)):
        video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    if _save_prediction_videos_enabled(cfg):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)

    trial_indices_cfg = cfg.EVALUATION.get("trial_indices")
    if trial_indices_cfg is None:
        requested_episode_count = int(cfg.EVALUATION.num_trials)
        required_initial_states = requested_episode_count
    else:
        trial_indices = [int(index) for index in trial_indices_cfg]
        requested_episode_count = len(trial_indices)
        required_initial_states = max(trial_indices) + 1 if trial_indices else 0
    while len(initial_states) < required_initial_states:
        initial_states.extend(
            initial_states[: (required_initial_states - len(initial_states))]
        )

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": requested_episode_count,
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
        "ckpt": str(Path(str(cfg.ckpt)).resolve()),
        "checkpoint_type": (
            "ema" if Path(str(cfg.ckpt)).stem.endswith("_ema") else "raw"
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "inference": _inference_provenance(cfg),
        "evaluation_protocol": {
            "scope": "closed_loop_libero",
            "control_replan_steps": int(
                cfg.EVALUATION.get("replan_steps", 5)
            ),
            "visualize_future_video": bool(
                cfg.EVALUATION.get("visualize_future_video", False)
            ),
            "compute_video_metrics": _compute_video_metrics_enabled(cfg),
            "save_rollout_videos": bool(
                cfg.EVALUATION.get("save_rollout_videos", True)
            ),
            "save_prediction_videos": _save_prediction_videos_enabled(cfg),
            "save_prediction_clip_videos": bool(
                cfg.EVALUATION.get("save_prediction_clip_videos", True)
            ),
            "compute_lpips": bool(cfg.EVALUATION.get("compute_lpips", True)),
            "compute_gfvd": bool(cfg.EVALUATION.get("compute_gfvd", True)),
        },
        "metric_protocol": {
            "enabled": _compute_video_metrics_enabled(cfg),
            "gt_source": "metric_rollout_from_each_closed_loop_replan_state",
            "control_rollout_isolation": (
                "restore_saved_simulator_and_controller_state_before_control"
            ),
            "extra_joint_model_inferences_per_episode": 0,
            "video_metric_horizon_steps": _get_video_metric_horizon_steps(cfg),
            "video_metric_replan_indices": (
                sorted(_get_video_metric_replan_indices(cfg))
                if _get_video_metric_replan_indices(cfg) is not None
                else None
            ),
            "video_frame_steps": _get_future_frame_capture_steps(cfg),
            "terminal_policy": "hold_last_observation",
            "future_frames_only": bool(
                cfg.EVALUATION.get(
                    "metrics_exclude_conditioning_frame", True
                )
            ),
            "psnr_data_range": 1.0,
            "ssim_kernel": 11,
            "lpips_net": (
                str(cfg.EVALUATION.get("lpips_net", "vgg"))
                if lpips_model is not None
                else None
            ),
            "wavelet": "one_level_orthonormal_haar_spatial_and_temporal",
            "boundary_metric": "t0_to_first_future_temporal_delta_mse",
            "gfvd": (
                {
                    "feature_detector": "kinetics_400_i3d_torchscript",
                    "detector_path": str(_resolve_fvd_detector_path(cfg)),
                    "detector_sha256": _sha256_file(
                        _resolve_fvd_detector_path(cfg)
                    ),
                    "num_frames": int(
                        cfg.EVALUATION.get("gfvd_num_frames", 16)
                    ),
                    "temporal_resample": "evenly_spaced_nearest",
                    "distribution_samples": "closed_loop_replan_metric_branches",
                }
                if fvd_detector is not None
                else None
            ),
        },
        "code": _git_provenance(project_root),
        "resolved_config": str(resolved_config_path.resolve()),
    }

    logging.info("Running LIBERO evaluation with env_num=1")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
        lpips_model=lpips_model,
        fvd_detector=fvd_detector,
    )
    gt_fvd_features = task_results.pop("_gt_fvd_features", None)
    pred_fvd_features = task_results.pop("_pred_fvd_features", None)
    results.update(task_results)
    if results["metric_protocol"]["gfvd"] is not None:
        results["metric_protocol"]["gfvd"]["num_clips"] = results.get(
            "gfvd_num_clips", 0
        )

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if gt_fvd_features is not None and pred_fvd_features is not None:
        fvd_feature_file = (
            output_dir
            / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_fvd_features.npz"
        )
        np.savez_compressed(
            fvd_feature_file,
            gt=np.asarray(gt_fvd_features, dtype=np.float32),
            pred=np.asarray(pred_fvd_features, dtype=np.float32),
        )
        results["fvd_feature_file"] = str(fvd_feature_file.resolve())
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{results['total_episodes']} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
