#!/usr/bin/env python3
"""Run the fixed Pixel WAM sampler/solver evaluation matrix.

The supervisor is restart-safe: a job is skipped only when its result JSON
matches the requested checkpoint, trials, schedule, solver, and step count.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PIXEL_ROOT = Path("/raid/kaiming/FastWAM")
LATENT_ROOT = Path("/raid/kaiming/FastWAM-jit")
PYTHON = Path("/home/admin/miniconda3/envs/fastwam/bin/python")
PROJECTION = PIXEL_ROOT / "artifacts/2026-08-18_stride4_all222929.pt"
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
    checkpoint: Path
    dataset_stats: Path
    overrides: tuple[str, ...] = ()


@dataclass
class Job:
    model: ModelSpec
    schedule: str
    solver: str
    steps: int
    output_dir: Path
    job_id: str
    attempts: int = 0
    status: str = "pending"
    gpu: int | None = None
    returncode: int | None = None
    duration: float | None = None
    command: list[str] = field(default_factory=list)


MODELS = (
    ModelSpec(
        name="pixel_asym_aall_no_vr_lpips",
        root=PIXEL_ROOT,
        checkpoint=PIXEL_ROOT
        / "runs/asym_libero_joint_2cam224_1e-4/2026-08-19_190513_all_LN_10ep/checkpoints/weights/step_021700.pt",
        dataset_stats=PIXEL_ROOT
        / "runs/asym_libero_joint_2cam224_1e-4/2026-08-19_190513_all_LN_10ep/dataset_stats.json",
        overrides=(
            f"model.projection_artifact_path={PROJECTION}",
            "model.asymflow.enabled=true",
            "model.asymflow.vr_enabled=false",
            "model.asymflow.lpips_enabled=false",
            "model.asymflow.vae_path=null",
            "model.asymflow.lpips_loss_weight=0.0",
        ),
    ),
    ModelSpec(
        name="pixel_asym_aall_vr_lpips",
        root=PIXEL_ROOT,
        checkpoint=PIXEL_ROOT
        / "runs/asym_libero_joint_2cam224_1e-4/2026-08-19_084450_all_LN_VR_LPIPS_10ep/checkpoints/weights/step_021700.pt",
        dataset_stats=PIXEL_ROOT
        / "runs/asym_libero_joint_2cam224_1e-4/2026-08-19_084450_all_LN_VR_LPIPS_10ep/dataset_stats.json",
        overrides=(
            f"model.projection_artifact_path={PROJECTION}",
            "model.asymflow.enabled=true",
            "model.asymflow.vr_enabled=true",
            "model.asymflow.lpips_enabled=true",
        ),
    ),
    ModelSpec(
        name="latent_fastwam_joint_local",
        root=LATENT_ROOT,
        checkpoint=PIXEL_ROOT
        / "runs/libero_joint_2cam224_1e-4/2026-07-27_2017/checkpoints/weights/step_021700.pt",
        dataset_stats=PIXEL_ROOT
        / "runs/libero_joint_2cam224_1e-4/2026-07-27_2017/dataset_stats.json",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PIXEL_ROOT
        / "evaluate_results/sampler_solver_ablation_20260826_task6_trials0-4_seed42",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[model.name for model in MODELS],
        help="Evaluate only selected built-in models (default: all three).",
    )
    parser.add_argument(
        "--vr-only-run-dir",
        type=Path,
        help="Evaluate one completed A_all VR-only run instead of built-in models.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_models(args: argparse.Namespace) -> tuple[ModelSpec, ...]:
    if args.vr_only_run_dir is not None:
        run_dir = args.vr_only_run_dir.resolve()
        return (
            ModelSpec(
                name=f"pixel_asym_aall_vr_only_{run_dir.name}",
                root=PIXEL_ROOT,
                checkpoint=run_dir / "checkpoints/weights/step_021700.pt",
                dataset_stats=run_dir / "dataset_stats.json",
                overrides=(
                    f"model.projection_artifact_path={PROJECTION}",
                    "model.asymflow.enabled=true",
                    "model.asymflow.vr_enabled=true",
                    "model.asymflow.lpips_enabled=false",
                    "model.asymflow.lpips_loss_weight=0.0",
                ),
            ),
        )
    if args.models:
        selected = set(args.models)
        return tuple(model for model in MODELS if model.name in selected)
    return MODELS


def build_jobs(
    output_root: Path,
    models: tuple[ModelSpec, ...],
) -> list[Job]:
    jobs = []
    # Interleave models so every GPU receives a mixture of model variants.
    for schedule in ("uniform", "logit_normal"):
        for solver in ("euler", "heun", "midpoint"):
            for steps in (10, 20):
                for model in models:
                    job_id = f"{model.name}__{schedule}__{solver}__n{steps}"
                    jobs.append(
                        Job(
                            model=model,
                            schedule=schedule,
                            solver=solver,
                            steps=steps,
                            output_dir=output_root / job_id,
                            job_id=job_id,
                        )
                    )
    return jobs


def result_path(job: Job) -> Path | None:
    files = sorted((job.output_dir / "libero_10").glob("gpu*_task6_results.json"))
    return files[0] if len(files) == 1 else None


def expected_model_evaluations(job: Job) -> int:
    if job.solver == "euler":
        return job.steps
    if job.solver == "heun":
        return 2 * job.steps - 1
    return 2 * job.steps


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nonempty_files(paths: Any, *, expected_count: int) -> bool:
    if not isinstance(paths, list) or len(paths) != expected_count:
        return False
    try:
        return all(Path(value).is_file() and Path(value).stat().st_size > 0 for value in paths)
    except (OSError, TypeError, ValueError):
        return False


def result_is_complete(job: Job) -> bool:
    path = result_path(job)
    if path is None:
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    inference = result.get("inference", {})
    metrics = result.get("future_video_metrics_mean", {})
    metric_protocol = result.get("metric_protocol", {})
    gfvd_protocol = metric_protocol.get("gfvd", {})
    code = result.get("code", {})
    expected_shift = 17.0 if job.schedule == "logit_normal" else None
    try:
        recorded_config = Path(str(result.get("resolved_config"))).resolve()
        recorded_config.relative_to(job.output_dir.resolve())
    except (OSError, TypeError):
        return False
    core_protocol_matches = (
        Path(str(result.get("ckpt"))).resolve() == job.model.checkpoint.resolve()
        and result.get("checkpoint_type") == "raw"
        and result.get("seed") == 42
        and result.get("task_suite") == "libero_10"
        and result.get("task_id") == 6
        and result.get("total_episodes") == 5
        and result.get("trial_indices") == [0, 1, 2, 3, 4]
        and inference.get("schedule") == job.schedule
        and inference.get("schedule_shift") == expected_shift
        and inference.get("ode_solver") == job.solver
        and inference.get("endpoint_policy")
        == ("terminal_euler" if job.solver == "heun" else None)
        and inference.get("num_steps") == job.steps
        and inference.get("num_model_evaluations")
        == expected_model_evaluations(job)
    )
    metrics_are_complete = (
        isinstance(metrics, dict)
        and all(_is_finite_number(metrics.get(name)) for name in CORE_VIDEO_METRICS)
        and isinstance(result.get("episode_future_video_metrics"), list)
        and len(result["episode_future_video_metrics"]) == 5
        and sorted(
            row.get("trial_index") for row in result["episode_future_video_metrics"]
        )
        == [0, 1, 2, 3, 4]
        and isinstance(result.get("future_video_clip_metrics"), list)
        and len(result["future_video_clip_metrics"]) > 0
    )
    provenance_is_complete = (
        recorded_config.is_file()
        and recorded_config.stat().st_size > 0
        and isinstance(code, dict)
        and bool(re.fullmatch(r"[0-9a-f]{40}", str(code.get("commit", ""))))
        and isinstance(code.get("dirty"), bool)
        and (
            not code["dirty"]
            or bool(
                re.fullmatch(
                    r"[0-9a-f]{64}", str(code.get("diff_sha256", ""))
                )
            )
        )
        and isinstance(gfvd_protocol, dict)
        and gfvd_protocol.get("feature_detector")
        == "kinetics_400_i3d_torchscript"
        and bool(
            re.fullmatch(
                r"[0-9a-f]{64}", str(gfvd_protocol.get("detector_sha256", ""))
            )
        )
        and _is_finite_number(gfvd_protocol.get("num_clips"))
        and int(gfvd_protocol["num_clips"]) > 0
        and gfvd_protocol.get("num_clips") == result.get("gfvd_num_clips")
    )
    videos_are_complete = _nonempty_files(
        result.get("rollout_video_files"), expected_count=5
    ) and _nonempty_files(result.get("prediction_video_files"), expected_count=5)
    return (
        core_protocol_matches
        and metrics_are_complete
        and provenance_is_complete
        and videos_are_complete
    )


def summary_is_complete(job: Job) -> bool:
    summary_path = job.output_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    overall = summary.get("overall", {})
    metrics = overall.get("future_video_metrics_mean", {})
    inference = summary.get("inference", {})
    expected_shift = 17.0 if job.schedule == "logit_normal" else None
    return (
        summary.get("run_id") == job.job_id
        and Path(str(summary.get("ckpt"))).resolve()
        == job.model.checkpoint.resolve()
        and summary.get("checkpoint_type") == "raw"
        and summary.get("seed") == 42
        and inference.get("schedule") == job.schedule
        and inference.get("schedule_shift") == expected_shift
        and inference.get("ode_solver") == job.solver
        and inference.get("num_steps") == job.steps
        and inference.get("num_model_evaluations")
        == expected_model_evaluations(job)
        and overall.get("total_trials") == 5
        and isinstance(metrics, dict)
        and all(_is_finite_number(metrics.get(name)) for name in CORE_VIDEO_METRICS)
    )


def build_command(job: Job, gpu: int) -> list[str]:
    schedule_shift = "17.0" if job.schedule == "logit_normal" else "null"
    return [
        str(PYTHON),
        "experiments/libero/eval_libero_single.py",
        "task=libero_joint_2cam224_1e-4",
        f"ckpt={job.model.checkpoint}",
        f"EVALUATION.dataset_stats_path={job.model.dataset_stats}",
        f"EVALUATION.output_dir={job.output_dir}",
        "EVALUATION.task_suite_name=libero_10",
        "EVALUATION.task_id=6",
        "EVALUATION.num_trials=5",
        "EVALUATION.trial_indices=[0,1,2,3,4]",
        "EVALUATION.visualize_future_video=true",
        "EVALUATION.save_prediction_clip_videos=false",
        "EVALUATION.compute_lpips=true",
        "EVALUATION.compute_gfvd=true",
        f"EVALUATION.inference_schedule={job.schedule}",
        f"EVALUATION.inference_schedule_shift={schedule_shift}",
        f"EVALUATION.ode_solver={job.solver}",
        f"EVALUATION.num_inference_steps={job.steps}",
        "seed=42",
        # CUDA_VISIBLE_DEVICES selects the physical device.  Keep this logical
        # id stable so retries cannot leave two differently named result files.
        "gpu_id=0",
        *job.model.overrides,
    ]


def job_record(job: Job) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "model": job.model.name,
        "repository": str(job.model.root),
        "checkpoint": str(job.model.checkpoint),
        "dataset_stats": str(job.model.dataset_stats),
        "schedule": job.schedule,
        "schedule_shift": 17.0 if job.schedule == "logit_normal" else None,
        "solver": job.solver,
        "steps": job.steps,
        "model_evaluations": expected_model_evaluations(job),
        "output_dir": str(job.output_dir),
        "status": job.status,
        "attempts": job.attempts,
        "gpu": job.gpu,
        "returncode": job.returncode,
        "duration": job.duration,
        "command": job.command,
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_manifest(output_root: Path, jobs: list[Job], workers: dict[int, int]) -> None:
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "task_suite": "libero_10",
            "task_id": 6,
            "trial_indices": [0, 1, 2, 3, 4],
            "seed": 42,
            "models": sorted({job.model.name for job in jobs}),
            "schedules": ["uniform", "logit_normal"],
            "schedule_scope": (
                "The selected schedule is applied jointly to video and action. "
                "Logit-normal shift 17 matches pixel-video training only; "
                "action training and the latent baseline use their native "
                "shift-5 scheduler sampling."
            ),
            "solvers": ["euler", "heun", "midpoint"],
            "steps": [10, 20],
            "total_jobs": len(jobs),
            "workers_per_gpu": workers,
        },
        "counts": {
            status: sum(job.status == status for job in jobs)
            for status in ("pending", "running", "completed", "failed")
        },
        "jobs": [job_record(job) for job in jobs],
    }
    write_json_atomic(output_root / "manifest.json", payload)


def summarize_job(job: Job, log_handle) -> bool:
    result = subprocess.run(
        [
            str(PYTHON),
            "experiments/libero/summarize_results.py",
            "--output_dir",
            str(job.output_dir),
        ],
        cwd=job.model.root,
        env={
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "src",
        },
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.returncode == 0 and (job.output_dir / "summary.json").is_file()


def write_matrix_summary(output_root: Path, jobs: list[Job]) -> None:
    rows = []
    for job in jobs:
        path = result_path(job)
        if path is None or not result_is_complete(job):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "job_id": job.job_id,
                "model": job.model.name,
                "checkpoint": str(job.model.checkpoint),
                "schedule": job.schedule,
                "solver": job.solver,
                "steps": job.steps,
                "model_evaluations": result["inference"]["num_model_evaluations"],
                "successes": result["successes"],
                "trials": result["total_episodes"],
                "success_rate": result["successes"] / result["total_episodes"],
                **result.get("future_video_metrics_mean", {}),
                "result_file": str(path),
                "summary_file": str(job.output_dir / "summary.json"),
                "resolved_config": result.get("resolved_config"),
            }
        )
    write_json_atomic(
        output_root / "matrix_summary.json",
        {
            "protocol": {
                "task_suite": "libero_10",
                "task_id": 6,
                "trial_indices": [0, 1, 2, 3, 4],
                "seed": 42,
                "metric_scope": "closed_loop_replan_clips",
                "schedule_scope": (
                    "The selected schedule is applied jointly to video and action. "
                    "Logit-normal shift 17 matches pixel-video training only; "
                    "action training and the latent baseline use their native "
                    "shift-5 scheduler sampling."
                ),
                "mfe_note": (
                    "Euler uses N MFE; endpoint-safe Heun uses 2N-1 MFE; "
                    "midpoint uses 2N MFE."
                ),
            },
            "completed_jobs": len(rows),
            "expected_jobs": len(jobs),
            "rows": rows,
        },
    )
    metric_keys = sorted(
        {
            key
            for row in rows
            for key in row
            if key
            not in {
                "job_id",
                "model",
                "checkpoint",
                "schedule",
                "solver",
                "steps",
                "model_evaluations",
                "successes",
                "trials",
                "result_file",
                "summary_file",
                "resolved_config",
            }
        }
    )
    columns = [
        "job_id",
        "model",
        "schedule",
        "solver",
        "steps",
        "model_evaluations",
        "successes",
        "trials",
        *metric_keys,
        "checkpoint",
        "result_file",
        "summary_file",
        "resolved_config",
    ]
    with (output_root / "matrix_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must contain unique GPU indices")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be positive")
    models = resolve_models(args)
    for model in models:
        for path in (model.root, model.checkpoint, model.dataset_stats):
            if not path.exists():
                raise FileNotFoundError(path)
    if not PROJECTION.is_file():
        raise FileNotFoundError(PROJECTION)

    jobs = build_jobs(output_root, models)
    for job in jobs:
        if result_is_complete(job):
            if not summary_is_complete(job):
                log_path = output_root / "logs" / f"{job.job_id}.log"
                with log_path.open("a", encoding="utf-8") as log_handle:
                    summarize_job(job, log_handle)
            if summary_is_complete(job):
                job.status = "completed"
    capacity = {gpu: args.workers_per_gpu for gpu in gpus}
    write_manifest(output_root, jobs, capacity)
    if args.dry_run:
        for index, job in enumerate(jobs):
            gpu = gpus[index % len(gpus)]
            print(job.job_id, "\n ", " ".join(build_command(job, gpu)))
        return 0

    pending = deque(job for job in jobs if job.status != "completed")
    running: dict[int, dict[str, Any]] = {}
    gpu_load = defaultdict(int)
    try:
        while pending or running:
            launched = True
            while pending and launched:
                launched = False
                for gpu in gpus:
                    if not pending or gpu_load[gpu] >= capacity[gpu]:
                        continue
                    job = pending.popleft()
                    job.attempts += 1
                    job.status = "running"
                    job.gpu = gpu
                    job.command = build_command(job, gpu)
                    job.output_dir.mkdir(parents=True, exist_ok=True)
                    log_path = output_root / "logs" / f"{job.job_id}.log"
                    log_handle = log_path.open("a", encoding="utf-8")
                    log_handle.write(
                        f"\n[{time.strftime('%F %T')}] attempt={job.attempts} gpu={gpu}\n"
                    )
                    log_handle.write("command=" + " ".join(job.command) + "\n")
                    log_handle.flush()
                    environment = {
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "PYTHONNOUSERSITE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONPATH": "src",
                    }
                    process = subprocess.Popen(
                        job.command,
                        cwd=job.model.root,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                    running[process.pid] = {
                        "process": process,
                        "job": job,
                        "gpu": gpu,
                        "start": time.time(),
                        "log": log_handle,
                        "log_path": log_path,
                    }
                    gpu_load[gpu] += 1
                    launched = True
                    print(
                        f"[{time.strftime('%F %T')}] launch {job.job_id} "
                        f"gpu={gpu} slot={gpu_load[gpu]}/{capacity[gpu]}"
                    )
                write_manifest(output_root, jobs, capacity)

            completed_pids = []
            for pid, state in running.items():
                returncode = state["process"].poll()
                if returncode is None:
                    continue
                completed_pids.append(pid)
                job = state["job"]
                job.returncode = returncode
                job.duration = time.time() - state["start"]
                gpu = state["gpu"]
                gpu_load[gpu] -= 1
                log_handle = state["log"]
                valid = returncode == 0 and result_is_complete(job)
                if valid:
                    valid = summarize_job(job, log_handle)
                if valid:
                    valid = summary_is_complete(job)
                log_handle.flush()
                log_handle.close()
                if valid:
                    job.status = "completed"
                    print(
                        f"[{time.strftime('%F %T')}] complete {job.job_id} "
                        f"gpu={gpu} duration={job.duration:.1f}s"
                    )
                elif job.attempts < args.max_attempts:
                    log_text = state["log_path"].read_text(
                        encoding="utf-8", errors="replace"
                    )
                    if "CUDA out of memory" in log_text and capacity[gpu] > 1:
                        capacity[gpu] -= 1
                    job.status = "pending"
                    pending.append(job)
                    print(
                        f"[{time.strftime('%F %T')}] retry {job.job_id} "
                        f"gpu={gpu} rc={returncode} capacity={capacity[gpu]}"
                    )
                else:
                    job.status = "failed"
                    print(
                        f"[{time.strftime('%F %T')}] failed {job.job_id} "
                        f"gpu={gpu} rc={returncode} log={state['log_path']}"
                    )
            for pid in completed_pids:
                del running[pid]
            if completed_pids:
                write_manifest(output_root, jobs, capacity)
                write_matrix_summary(output_root, jobs)
            if running and not completed_pids:
                time.sleep(5)
    except KeyboardInterrupt:
        print("Interrupt received; terminating matrix children.")
        for state in running.values():
            state["process"].terminate()
        for state in running.values():
            state["process"].wait()
            state["log"].close()
        raise

    write_manifest(output_root, jobs, capacity)
    write_matrix_summary(output_root, jobs)
    failed = [job for job in jobs if job.status == "failed"]
    print(
        f"Matrix finished: completed={sum(job.status == 'completed' for job in jobs)} "
        f"failed={len(failed)} output={output_root}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
