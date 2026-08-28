import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd


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
        }
    )
    task_results: dict[str, dict] = {}
    run_metadata: dict = {}

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
            stats["metric_rows"].append(metrics)

        task_results[task_key] = {
            "success_rate": 100.0 * result["successes"] / result["total_episodes"],
            "duration": float(result["duration"]),
            "total_episodes": int(result["total_episodes"]),
            "successes": int(result["successes"]),
            "trial_indices": result.get("trial_indices"),
            "task_description": result.get("task_description", ""),
            "future_video_metrics_mean": metrics,
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
        metric_mean = _mean_metrics(metric_rows)
        all_metric_rows.extend(metric_rows)
        suite_stats_output[suite] = {
            **stats,
            "success_rate": (
                100.0 * stats["total_successes"] / stats["total_trials"]
                if stats["total_trials"]
                else 0.0
            ),
            "future_video_metrics_mean": metric_mean,
        }
        total_trials += stats["total_trials"]
        total_successes += stats["total_successes"]
        total_time += stats["total_time"]
        max_time = max(max_time, stats["max_time"])

    total_tasks = sum(stats["total_tasks"] for stats in suite_stats_output.values())
    overall = {
        "success_rate": 100.0 * total_successes / total_trials if total_trials else 0.0,
        "total_trials": total_trials,
        "total_successes": total_successes,
        "total_time": total_time,
        "average_task_time": total_time / total_tasks if total_tasks else 0.0,
        "max_task_time": max_time,
        "future_video_metrics_mean": _mean_metrics(all_metric_rows),
    }
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
