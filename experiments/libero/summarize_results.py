import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg


SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)


def format_time(seconds: float) -> str:
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds:02d}s"
    if seconds < 3600:
        return f"{seconds // 60:02d}m{seconds % 60:02d}s"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:02d}h{minutes:02d}m{seconds % 60:02d}s"


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: sum(float(row[key]) for row in rows if row.get(key) is not None)
        / sum(row.get(key) is not None for row in rows)
        for key in keys
        if any(row.get(key) is not None for row in rows)
    }


def _frechet_feature_distance(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.ndim != 2 or target.ndim != 2 or pred.shape[1] != target.shape[1]:
        raise ValueError(
            f"Invalid FVD feature shapes: pred={pred.shape} target={target.shape}"
        )
    if pred.shape[0] < 2 or target.shape[0] < 2:
        raise ValueError("FVD requires at least two clips per distribution")
    pred_mean = pred.mean(axis=0)
    target_mean = target.mean(axis=0)
    pred_cov = np.cov(pred, rowvar=False)
    target_cov = np.cov(target, rowvar=False)
    covariance_mean = scipy.linalg.sqrtm(pred_cov @ target_cov)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(pred_cov.shape[0]) * 1e-6
        covariance_mean = scipy.linalg.sqrtm(
            (pred_cov + offset) @ (target_cov + offset)
        )
    if np.iscomplexobj(covariance_mean):
        imaginary_max = float(np.max(np.abs(covariance_mean.imag)))
        if imaginary_max > 1e-3:
            raise ValueError(
                "FVD covariance product has a large imaginary component: "
                f"{imaginary_max}"
            )
        covariance_mean = covariance_mean.real
    distance = (
        np.square(pred_mean - target_mean).sum()
        + np.trace(pred_cov + target_cov - 2.0 * covariance_mean)
    )
    return max(float(np.real(distance)), 0.0)


def _load_fvd_features(result: dict) -> tuple[np.ndarray, np.ndarray] | None:
    configured = result.get("fvd_feature_file")
    if not configured:
        return None
    feature_file = Path(str(configured))
    if not feature_file.is_file():
        raise FileNotFoundError(f"Missing task FVD feature file: {feature_file}")
    with np.load(feature_file) as features:
        gt = np.asarray(features["gt"], dtype=np.float32)
        pred = np.asarray(features["pred"], dtype=np.float32)
    if gt.shape != pred.shape:
        raise ValueError(
            f"Task FVD feature shape mismatch: gt={gt.shape} pred={pred.shape}"
        )
    return gt, pred


def _result_files(output_dir: Path):
    for suite in SUITES:
        suite_dir = output_dir / suite
        if not suite_dir.is_dir():
            continue
        yield from sorted(suite_dir.glob("gpu*_task*_results.json"))


def summarize_results(output_dir: str) -> None:
    output_path = Path(output_dir).resolve()
    suite_stats = defaultdict(
        lambda: {
            "total_tasks": 0,
            "total_trials": 0,
            "total_successes": 0,
            "total_time": 0.0,
            "max_time": 0.0,
            "metric_rows": [],
            "gt_fvd_features": [],
            "pred_fvd_features": [],
        }
    )
    task_results: dict[str, dict] = {}
    run_metadata: dict = {}
    all_gt_fvd_features = []
    all_pred_fvd_features = []

    for result_file in _result_files(output_path):
        with result_file.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        suite = str(result.get("task_suite", result_file.parent.name))
        task_id = int(result["task_id"])
        task_key = f"{suite}_{task_id}"
        stats = suite_stats[suite]
        stats["total_tasks"] += 1
        stats["total_trials"] += int(result["total_episodes"])
        stats["total_successes"] += int(result["successes"])
        stats["total_time"] += float(result["duration"])
        stats["max_time"] = max(stats["max_time"], float(result["duration"]))

        metrics = {
            key: float(value)
            for key, value in result.get("future_video_metrics_mean", {}).items()
            if value is not None
        }
        if not metrics and result.get("future_video_psnr_mean") is not None:
            metrics["psnr"] = float(result["future_video_psnr_mean"])
        if metrics:
            stats["metric_rows"].append(
                {key: value for key, value in metrics.items() if key != "gfvd"}
            )

        fvd_features = _load_fvd_features(result)
        if fvd_features is not None:
            gt_features, pred_features = fvd_features
            stats["gt_fvd_features"].append(gt_features)
            stats["pred_fvd_features"].append(pred_features)
            all_gt_fvd_features.append(gt_features)
            all_pred_fvd_features.append(pred_features)

        task_results[task_key] = {
            "success_rate": 100.0 * result["successes"] / result["total_episodes"],
            "duration": float(result["duration"]),
            "total_episodes": int(result["total_episodes"]),
            "successes": int(result["successes"]),
            "trial_indices": result.get("trial_indices"),
            "task_description": result.get("task_description", ""),
            "future_video_metrics_mean": metrics,
            "fvd_feature_file": result.get("fvd_feature_file"),
            "result_file": str(result_file),
        }
        if not run_metadata:
            run_metadata = {
                key: result.get(key)
                for key in (
                    "ckpt",
                    "checkpoint_type",
                    "seed",
                    "inference",
                    "evaluation_protocol",
                    "metric_protocol",
                    "code",
                    "resolved_config",
                )
            }

    suite_stats_output = {}
    all_metric_rows = []
    total_trials = 0
    total_successes = 0
    total_time = 0.0
    max_time = 0.0
    for suite, stats in suite_stats.items():
        metric_rows = stats.pop("metric_rows")
        gt_fvd_features = stats.pop("gt_fvd_features")
        pred_fvd_features = stats.pop("pred_fvd_features")
        metric_mean = _mean_metrics(metric_rows)
        suite_gfvd_num_clips = 0
        if gt_fvd_features:
            suite_gt = np.concatenate(gt_fvd_features, axis=0)
            suite_pred = np.concatenate(pred_fvd_features, axis=0)
            metric_mean["gfvd"] = _frechet_feature_distance(
                suite_pred, suite_gt
            )
            suite_gfvd_num_clips = int(suite_gt.shape[0])
        all_metric_rows.extend(metric_rows)
        suite_stats_output[suite] = {
            **stats,
            "success_rate": (
                100.0 * stats["total_successes"] / stats["total_trials"]
                if stats["total_trials"]
                else 0.0
            ),
            "future_video_metrics_mean": metric_mean,
            "gfvd_num_clips": suite_gfvd_num_clips,
        }
        total_trials += stats["total_trials"]
        total_successes += stats["total_successes"]
        total_time += stats["total_time"]
        max_time = max(max_time, stats["max_time"])

    total_tasks = sum(stats["total_tasks"] for stats in suite_stats_output.values())
    overall_metrics = _mean_metrics(all_metric_rows)
    overall_gfvd_num_clips = 0
    if all_gt_fvd_features:
        overall_gt = np.concatenate(all_gt_fvd_features, axis=0)
        overall_pred = np.concatenate(all_pred_fvd_features, axis=0)
        overall_metrics["gfvd"] = _frechet_feature_distance(
            overall_pred, overall_gt
        )
        overall_gfvd_num_clips = int(overall_gt.shape[0])
    overall = {
        "success_rate": 100.0 * total_successes / total_trials if total_trials else 0.0,
        "total_trials": total_trials,
        "total_successes": total_successes,
        "total_time": total_time,
        "average_task_time": total_time / total_tasks if total_tasks else 0.0,
        "max_task_time": max_time,
        "future_video_metrics_mean": overall_metrics,
        "gfvd_num_clips": overall_gfvd_num_clips,
    }
    metric_protocol = run_metadata.get("metric_protocol")
    if isinstance(metric_protocol, dict) and isinstance(
        metric_protocol.get("gfvd"), dict
    ):
        metric_protocol["gfvd"]["num_clips"] = overall_gfvd_num_clips
    summary = {
        "run_id": output_path.name,
        **run_metadata,
        "config": run_metadata.get("resolved_config", os.environ.get("CONFIG", "")),
        "suite_stats": suite_stats_output,
        "task_results": task_results,
        "overall": overall,
    }
    with (output_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)

    table_rows = []
    for suite, stats in suite_stats_output.items():
        row = {
            "Task Suite": suite,
            "Success Rate (%)": stats["success_rate"],
            "Average Time (s)": stats["total_time"] / stats["total_tasks"],
            "Max Time (s)": stats["max_time"],
        }
        row.update(
            {f"Future {key}": value for key, value in stats["future_video_metrics_mean"].items()}
        )
        table_rows.append(row)
    if suite_stats_output:
        overall_row = {
            "Task Suite": "Overall",
            "Success Rate (%)": overall["success_rate"],
            "Average Time (s)": overall["average_task_time"],
            "Max Time (s)": overall["max_task_time"],
        }
        overall_row.update(
            {f"Future {key}": value for key, value in overall["future_video_metrics_mean"].items()}
        )
        table_rows.append(overall_row)
    table = pd.DataFrame(table_rows)
    title = Path(str(run_metadata.get("ckpt") or os.environ.get("CKPT", "Results"))).name
    with (output_path / "summary.csv").open("w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        table.to_csv(handle, index=False)

    task_table = pd.DataFrame(
        [
            {
                "Task": task,
                "Description": result["task_description"],
                "Success Rate (%)": result["success_rate"],
                **{
                    f"Future {key}": value
                    for key, value in result["future_video_metrics_mean"].items()
                },
            }
            for task, result in sorted(task_results.items())
        ]
    )
    task_table.to_csv(output_path / "task_success_rates.csv", index=False)

    print("\n=== Evaluation Results Summary ===")
    print(f"Results directory: {output_path}")
    print(f"Checkpoint: {run_metadata.get('ckpt', '')}")
    print(f"Success: {total_successes}/{total_trials} ({overall['success_rate']:.2f}%)")
    print(f"Total time: {format_time(total_time)}")
    if overall["future_video_metrics_mean"]:
        print("Future-video metrics:")
        for name, value in overall["future_video_metrics_mean"].items():
            print(f"- {name}: {value:.6f}")
    print(f"Summary file: {output_path / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory containing evaluation results",
    )
    args = parser.parse_args()
    summarize_results(args.output_dir)


if __name__ == "__main__":
    main()
