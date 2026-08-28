#!/usr/bin/env python3
"""Run the 400-episode Pixel WAM schedule/solver video matrix.

Each matrix cell uses the same checkpoint and evaluates all 40 LIBERO tasks
with trials 0..9.  The only inference variables are the timestep schedule and
ODE solver; the denoising step count remains fixed at 10.  The underlying
LIBERO manager is limited to one worker per GPU.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


DEFAULT_ROOT = Path("/raid/kaiming/FastWAM")
PYTHON = Path("/home/admin/miniconda3/envs/fastwam/bin/python")
SUITES = ("libero_10", "libero_goal", "libero_spatial", "libero_object")
TASK_IDS = tuple(range(10))
TRIAL_INDICES = tuple(range(10))
SCHEDULES = ("uniform", "logit_normal")
SOLVERS = ("euler", "heun", "midpoint")
NUM_INFERENCE_STEPS = 10
CORE_VIDEO_METRICS = (
    "gfvd",
    "lpips",
    "num_future_frames",
    "psnr",
    "ssim",
    "wavelet_hh_mse",
    "wavelet_high_mse",
    "wavelet_hl_mse",
    "wavelet_lh_mse",
    "wavelet_ll_mse",
    "wavelet_temporal_high_mse",
    "wavelet_temporal_low_mse",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    root: Path
    kind: str
    run_dir: Path
    checkpoint: Path
    dataset_stats: Path
    projection_artifact: Path | None = None
    prediction_type: str | None = None
    vr_enabled: bool | None = None
    lpips_enabled: bool | None = None


@dataclass(frozen=True)
class Cell:
    model: ModelSpec
    schedule: str
    solver: str
    output_dir: Path

    @property
    def cell_id(self) -> str:
        return (
            f"{self.model.name}__{self.schedule}__{self.solver}"
            f"__n{NUM_INFERENCE_STEPS}"
        )


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model-kind", choices=("pixel", "latent"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--final-step", type=int, default=21700)
    parser.add_argument("--projection-artifact", type=Path)
    parser.add_argument(
        "--prediction-type",
        choices=("asym_velocity", "x0"),
    )
    parser.add_argument("--vr-enabled", type=parse_bool)
    parser.add_argument("--lpips-enabled", type=parse_bool)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def expected_model_evaluations(solver: str) -> int:
    if solver == "euler":
        return NUM_INFERENCE_STEPS
    if solver == "heun":
        return 2 * NUM_INFERENCE_STEPS - 1
    if solver == "midpoint":
        return 2 * NUM_INFERENCE_STEPS
    raise ValueError(f"Unsupported solver: {solver}")


def schedule_shift(schedule: str) -> float | None:
    return 17.0 if schedule == "logit_normal" else None


def build_cells(output_dir: Path, model: ModelSpec) -> list[Cell]:
    return [
        Cell(
            model=model,
            schedule=schedule,
            solver=solver,
            output_dir=output_dir
            / f"{model.name}__{schedule}__{solver}__n{NUM_INFERENCE_STEPS}",
        )
        for schedule in SCHEDULES
        for solver in SOLVERS
    ]


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _path_matches(value: Any, expected: Path) -> bool:
    try:
        return Path(str(value)).resolve() == expected.resolve()
    except (OSError, TypeError, ValueError):
        return False


def _nonempty_files(values: Any, expected_count: int) -> bool:
    if not isinstance(values, list) or len(values) != expected_count:
        return False
    try:
        return all(Path(value).is_file() and Path(value).stat().st_size > 0 for value in values)
    except (OSError, TypeError, ValueError):
        return False


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _result_is_complete(path: Path, cell: Cell, suite: str, task_id: int) -> bool:
    result = _load_json(path)
    if result is None:
        return False
    inference = result.get("inference", {})
    evaluation = result.get("evaluation_protocol", {})
    metrics = result.get("future_video_metrics_mean", {})
    code = result.get("code", {})
    resolved_config = Path(str(result.get("resolved_config", "")))
    return (
        _path_matches(result.get("ckpt"), cell.model.checkpoint)
        and result.get("checkpoint_type") == "raw"
        and result.get("seed") == 42
        and result.get("task_suite") == suite
        and result.get("task_id") == task_id
        and result.get("total_episodes") == len(TRIAL_INDICES)
        and result.get("trial_indices") == list(TRIAL_INDICES)
        and inference.get("schedule") == cell.schedule
        and inference.get("schedule_shift") == schedule_shift(cell.schedule)
        and inference.get("sigma_shift") is None
        and inference.get("ode_solver") == cell.solver
        and inference.get("endpoint_policy")
        == ("terminal_euler" if cell.solver == "heun" else None)
        and inference.get("num_steps") == NUM_INFERENCE_STEPS
        and inference.get("num_model_evaluations")
        == expected_model_evaluations(cell.solver)
        and evaluation.get("scope") == "closed_loop_libero"
        and evaluation.get("visualize_future_video") is True
        and evaluation.get("save_rollout_videos") is True
        and evaluation.get("save_prediction_clip_videos") is False
        and evaluation.get("compute_lpips") is True
        and evaluation.get("compute_gfvd") is True
        and isinstance(metrics, dict)
        and all(_is_finite_number(metrics.get(name)) for name in CORE_VIDEO_METRICS)
        and isinstance(result.get("episode_future_video_metrics"), list)
        and len(result["episode_future_video_metrics"]) == len(TRIAL_INDICES)
        and isinstance(result.get("future_video_clip_metrics"), list)
        and len(result["future_video_clip_metrics"]) > 0
        and _nonempty_files(result.get("rollout_video_files"), len(TRIAL_INDICES))
        and _nonempty_files(result.get("prediction_video_files"), len(TRIAL_INDICES))
        and resolved_config.is_file()
        and resolved_config.stat().st_size > 0
        and isinstance(code, dict)
        and bool(re.fullmatch(r"[0-9a-f]{40}", str(code.get("commit", ""))))
        and isinstance(code.get("dirty"), bool)
    )


def summary_is_complete(cell: Cell) -> bool:
    summary = _load_json(cell.output_dir / "summary.json")
    if summary is None:
        return False
    inference = summary.get("inference", {})
    evaluation = summary.get("evaluation_protocol", {})
    overall = summary.get("overall", {})
    suites = summary.get("suite_stats", {})
    tasks = summary.get("task_results", {})
    metrics = overall.get("future_video_metrics_mean", {})
    expected_task_keys = {
        f"{suite}_{task_id}" for suite in SUITES for task_id in TASK_IDS
    }
    summary_matches = (
        _path_matches(summary.get("ckpt"), cell.model.checkpoint)
        and summary.get("checkpoint_type") == "raw"
        and summary.get("seed") == 42
        and inference.get("schedule") == cell.schedule
        and inference.get("schedule_shift") == schedule_shift(cell.schedule)
        and inference.get("sigma_shift") is None
        and inference.get("ode_solver") == cell.solver
        and inference.get("endpoint_policy")
        == ("terminal_euler" if cell.solver == "heun" else None)
        and inference.get("num_steps") == NUM_INFERENCE_STEPS
        and inference.get("num_model_evaluations")
        == expected_model_evaluations(cell.solver)
        and evaluation.get("visualize_future_video") is True
        and evaluation.get("save_rollout_videos") is True
        and evaluation.get("save_prediction_clip_videos") is False
        and evaluation.get("compute_lpips") is True
        and evaluation.get("compute_gfvd") is True
        and overall.get("total_trials") == 400
        and isinstance(overall.get("total_successes"), int)
        and 0 <= overall["total_successes"] <= 400
        and isinstance(metrics, dict)
        and all(_is_finite_number(metrics.get(name)) for name in CORE_VIDEO_METRICS)
        and set(suites) == set(SUITES)
        and all(
            suites[suite].get("total_tasks") == 10
            and suites[suite].get("total_trials") == 100
            and isinstance(suites[suite].get("total_successes"), int)
            for suite in SUITES
        )
        and set(tasks) == expected_task_keys
        and all(
            tasks[key].get("total_episodes") == len(TRIAL_INDICES)
            and tasks[key].get("trial_indices") == list(TRIAL_INDICES)
            for key in expected_task_keys
        )
    )
    if not summary_matches:
        return False
    for suite in SUITES:
        for task_id in TASK_IDS:
            result_path = Path(tasks[f"{suite}_{task_id}"]["result_file"])
            if not _result_is_complete(result_path, cell, suite, task_id):
                return False
    return True


def build_command(cell: Cell, num_gpus: int) -> list[str]:
    model = cell.model
    overrides = []
    if model.kind == "pixel":
        assert model.projection_artifact is not None
        assert model.prediction_type is not None
        assert model.vr_enabled is not None
        assert model.lpips_enabled is not None
        overrides.extend(
            (
                f"model.projection_artifact_path={model.projection_artifact}",
                "model.asymflow.enabled=true",
                f"model.asymflow.prediction_type={model.prediction_type}",
                f"model.asymflow.vr_enabled={str(model.vr_enabled).lower()}",
                f"model.asymflow.lpips_enabled={str(model.lpips_enabled).lower()}",
                "model.asymflow.timestep_sampling=logit_normal",
                "model.asymflow.timestep_shift=17.0",
            )
        )
        if not model.vr_enabled:
            overrides.extend(
                (
                    "model.asymflow.vae_path=null",
                    "model.asymflow.lpips_loss_weight=0.0",
                )
            )
    shift = "17.0" if cell.schedule == "logit_normal" else "null"
    return [
        str(PYTHON),
        "experiments/libero/run_libero_manager.py",
        "task=libero_joint_2cam224_1e-4",
        f"ckpt={model.checkpoint}",
        f"EVALUATION.dataset_stats_path={model.dataset_stats}",
        f"EVALUATION.output_dir={cell.output_dir}",
        f"EVALUATION.num_trials={len(TRIAL_INDICES)}",
        "EVALUATION.trial_indices=[0,1,2,3,4,5,6,7,8,9]",
        "EVALUATION.visualize_future_video=true",
        "EVALUATION.save_rollout_videos=true",
        "EVALUATION.save_prediction_clip_videos=false",
        "EVALUATION.compute_lpips=true",
        "EVALUATION.compute_gfvd=true",
        "EVALUATION.metrics_exclude_conditioning_frame=true",
        "EVALUATION.sigma_shift=null",
        f"EVALUATION.inference_schedule={cell.schedule}",
        f"EVALUATION.inference_schedule_shift={shift}",
        f"EVALUATION.ode_solver={cell.solver}",
        f"EVALUATION.num_inference_steps={NUM_INFERENCE_STEPS}",
        "seed=42",
        f"MULTIRUN.num_gpus={num_gpus}",
        "MULTIRUN.max_tasks_per_gpu=1",
        *overrides,
    ]


def validate_model(args: argparse.Namespace) -> ModelSpec:
    if args.final_step <= 0:
        raise ValueError("--final-step must be positive")
    if args.num_gpus <= 0:
        raise ValueError("--num-gpus must be positive")
    if args.workers_per_gpu != 1:
        raise ValueError("This protocol requires exactly one worker per GPU")
    if args.model_kind == "pixel":
        if args.projection_artifact is None:
            raise ValueError("Pixel models require --projection-artifact")
        if args.prediction_type is None:
            raise ValueError("Pixel models require --prediction-type")
        if args.vr_enabled is None or args.lpips_enabled is None:
            raise ValueError("Pixel models require explicit VR and LPIPS settings")
        if args.lpips_enabled and not args.vr_enabled:
            raise ValueError("LPIPS requires VR to be enabled")
    elif any(
        value is not None
        for value in (
            args.projection_artifact,
            args.prediction_type,
            args.vr_enabled,
            args.lpips_enabled,
        )
    ):
        raise ValueError("Latent models do not accept Pixel AsymFlow overrides")

    repository_root = args.repository_root.resolve()
    run_dir = args.run_dir.resolve()
    checkpoint = run_dir / "checkpoints/weights" / f"step_{args.final_step:06d}.pt"
    dataset_stats = run_dir / "dataset_stats.json"
    run_config = run_dir / "config.yaml"
    projection = (
        None
        if args.projection_artifact is None
        else args.projection_artifact.resolve()
    )
    required_paths = [repository_root, checkpoint, dataset_stats, run_config]
    if projection is not None:
        required_paths.append(projection)
    for path in required_paths:
        if not path.is_file() or path.stat().st_size <= 0:
            if path == repository_root and path.is_dir():
                continue
            raise FileNotFoundError(f"Required artifact is missing or empty: {path}")
    if "ema" in checkpoint.name.lower():
        raise ValueError(f"EMA checkpoints are forbidden: {checkpoint}")

    cfg = OmegaConf.load(run_config)
    checks = {
        "seed": int(cfg.seed) == 42,
        "resume": cfg.resume is None,
    }
    if args.model_kind == "pixel":
        checks.update(
            {
                "prediction_type": str(cfg.model.asymflow.get("prediction_type", "asym_velocity"))
                == args.prediction_type,
                "vr_enabled": bool(cfg.model.asymflow.vr_enabled)
                == args.vr_enabled,
                "lpips_enabled": bool(cfg.model.asymflow.lpips_enabled)
                == args.lpips_enabled,
                "timestep_sampling": str(cfg.model.asymflow.timestep_sampling)
                == "logit_normal",
                "timestep_shift": float(cfg.model.asymflow.timestep_shift)
                == 17.0,
                "projection_artifact": Path(
                    cfg.model.projection_artifact_path
                ).resolve()
                == projection,
            }
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Run configuration mismatch: {failed}")

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if int(payload.get("step", -1)) != args.final_step:
        raise ValueError(
            f"Checkpoint step mismatch: {payload.get('step')} != {args.final_step}"
        )
    if not isinstance(payload.get("mot"), dict) or not payload["mot"]:
        raise ValueError(f"Checkpoint has no non-empty raw MoT state: {checkpoint}")
    del payload
    gc.collect()

    return ModelSpec(
        name=args.model_name,
        root=repository_root,
        kind=args.model_kind,
        run_dir=run_dir,
        checkpoint=checkpoint,
        dataset_stats=dataset_stats,
        projection_artifact=projection,
        prediction_type=args.prediction_type,
        vr_enabled=args.vr_enabled,
        lpips_enabled=args.lpips_enabled,
    )


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_combined_report(output_dir: Path, cells: list[Cell]) -> None:
    rows = []
    for cell in cells:
        if not summary_is_complete(cell):
            continue
        summary = _load_json(cell.output_dir / "summary.json")
        assert summary is not None
        overall = summary["overall"]
        metrics = overall["future_video_metrics_mean"]
        rows.append(
            {
                "cell_id": cell.cell_id,
                "model": cell.model.name,
                "checkpoint": str(cell.model.checkpoint),
                "schedule": cell.schedule,
                "schedule_shift": schedule_shift(cell.schedule),
                "solver": cell.solver,
                "steps": NUM_INFERENCE_STEPS,
                "model_evaluations": expected_model_evaluations(cell.solver),
                "successes": overall["total_successes"],
                "episodes": overall["total_trials"],
                "success_rate": overall["success_rate"],
                **{name: metrics[name] for name in CORE_VIDEO_METRICS},
                "summary_file": str(cell.output_dir / "summary.json"),
            }
        )

    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "checkpoint_type": "raw",
            "seed": 42,
            "task_suites": list(SUITES),
            "tasks_per_suite": 10,
            "trial_indices": list(TRIAL_INDICES),
            "trials_per_task": 10,
            "episodes_per_cell": 400,
            "schedules": list(SCHEDULES),
            "solvers": list(SOLVERS),
            "steps": NUM_INFERENCE_STEPS,
            "visualize_future_video": True,
            "save_rollout_videos": True,
            "save_prediction_clip_videos": False,
            "workers_per_gpu": 1,
        },
        "expected_cells": len(cells),
        "completed_cells": len(rows),
        "expected_episodes": 400 * len(cells),
        "completed_episodes": 400 * len(rows),
        "rows": rows,
    }
    _write_json_atomic(output_dir / "matrix_summary.json", payload)

    fieldnames = [
        "cell_id",
        "model",
        "checkpoint",
        "schedule",
        "schedule_shift",
        "solver",
        "steps",
        "model_evaluations",
        "successes",
        "episodes",
        "success_rate",
        *CORE_VIDEO_METRICS,
        "summary_file",
    ]
    with (output_dir / "matrix_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Pixel WAM 400-episode sampler/solver evaluation",
        "",
        "Each row is 40 LIBERO tasks x trials 0..9 = 400 closed-loop episodes.",
        "All episodes use joint video/action inference and save rollout and predicted video.",
        "",
        f"Progress: {len(rows)}/{len(cells)} cells, {400 * len(rows)}/{400 * len(cells)} episodes.",
        "",
        "| Schedule | Solver | N | MFE | Success | Rate | gFVD | LPIPS | PSNR | SSIM |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['schedule']} | {row['solver']} | {row['steps']} | "
            f"{row['model_evaluations']} | {row['successes']}/{row['episodes']} | "
            f"{row['success_rate']:.2f}% | {row['gfvd']:.3f} | "
            f"{row['lpips']:.4f} | {row['psnr']:.3f} | {row['ssim']:.4f} |"
        )
    (output_dir / "matrix_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.dry_run and args.validate_only:
        raise ValueError("--dry-run and --validate-only are mutually exclusive")
    if args.dry_run:
        run_dir = args.run_dir.resolve()
        model = ModelSpec(
            name=args.model_name,
            root=args.repository_root.resolve(),
            kind=args.model_kind,
            run_dir=run_dir,
            checkpoint=run_dir
            / "checkpoints/weights"
            / f"step_{args.final_step:06d}.pt",
            dataset_stats=run_dir / "dataset_stats.json",
            projection_artifact=(
                None
                if args.projection_artifact is None
                else args.projection_artifact.resolve()
            ),
            prediction_type=args.prediction_type,
            vr_enabled=args.vr_enabled,
            lpips_enabled=args.lpips_enabled,
        )
    else:
        model = validate_model(args)

    if args.validate_only:
        print(f"validated model={model.name} checkpoint={model.checkpoint}")
        return 0

    output_dir = args.output_dir.resolve()
    cells = build_cells(output_dir, model)
    print(
        f"model={model.name} cells={len(cells)} episodes_per_cell=400 "
        f"total_episodes={400 * len(cells)} workers_per_gpu=1"
    )
    if args.dry_run:
        for cell in cells:
            print(f"\n[{cell.cell_id}]")
            print(" ".join(build_command(cell, args.num_gpus)))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    write_combined_report(output_dir, cells)
    environment = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "src",
        "NCCL_NVLS_ENABLE": "0",
        "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(args.num_gpus)),
    }
    for index, cell in enumerate(cells, start=1):
        if summary_is_complete(cell):
            print(f"[{index}/{len(cells)}] complete, skipping {cell.cell_id}")
            continue
        print(f"[{index}/{len(cells)}] starting {cell.cell_id}")
        cell_environment = {**environment, "EXP_NAME": cell.cell_id}
        subprocess.run(
            build_command(cell, args.num_gpus),
            cwd=cell.model.root,
            env=cell_environment,
            check=True,
        )
        if not summary_is_complete(cell):
            raise RuntimeError(f"Cell failed completeness validation: {cell.cell_id}")
        write_combined_report(output_dir, cells)

    write_combined_report(output_dir, cells)
    combined = _load_json(output_dir / "matrix_summary.json")
    assert combined is not None
    if combined["completed_cells"] != combined["expected_cells"]:
        raise RuntimeError(f"Matrix is incomplete: {combined}")
    print(
        f"completed {combined['completed_cells']} cells and "
        f"{combined['completed_episodes']} episodes: {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
