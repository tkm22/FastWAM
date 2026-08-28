#!/usr/bin/env python3
"""Create a compact comparison report from a completed sampler matrix."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


METRICS = (
    "success_rate",
    "gfvd",
    "psnr",
    "ssim",
    "lpips",
    "wavelet_high_mse",
    "wavelet_temporal_high_mse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-summary", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def keyed(rows, keys):
    return {tuple(row[key] for key in keys): row for row in rows}


def metric_delta(left: dict, right: dict) -> dict[str, float]:
    return {
        metric: float(right[metric]) - float(left[metric])
        for metric in METRICS
        if left.get(metric) is not None and right.get(metric) is not None
    }


def average_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: mean(float(row[key]) for row in rows if key in row)
        for key in keys
    }


def paired_effect(
    rows: list[dict],
    *,
    keys: tuple[str, ...],
    dimension: str,
    left_value,
    right_value,
) -> dict:
    table = keyed(rows, (*keys, dimension))
    deltas = []
    pairs = []
    base_keys = sorted({tuple(row[key] for key in keys) for row in rows})
    for base in base_keys:
        left = table.get((*base, left_value))
        right = table.get((*base, right_value))
        if left is None or right is None:
            continue
        delta = metric_delta(left, right)
        deltas.append(delta)
        pairs.append(
            {
                **dict(zip(keys, base)),
                "left": left_value,
                "right": right_value,
                "delta_right_minus_left": delta,
            }
        )
    return {
        "comparison": f"{right_value} minus {left_value}",
        "num_pairs": len(pairs),
        "mean_delta": average_dicts(deltas),
        "pairs": pairs,
    }


def format_value(value, digits=4):
    return "-" if value is None else f"{float(value):.{digits}f}"


def model_label(name: str) -> str:
    if name.startswith("pixel_asym_aall_vr_only_"):
        return "Pixel A_all + VR-only"
    labels = {
        "pixel_asym_aall_no_vr_lpips": "Pixel A_all",
        "pixel_asym_aall_vr_lpips": "Pixel A_all + VR/LPIPS",
        "latent_fastwam_joint_local": "Latent FastWAM-Joint",
    }
    return labels.get(name, name)


def main() -> int:
    args = parse_args()
    summary_path = args.matrix_summary.resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    if (
        not args.allow_incomplete
        and payload["completed_jobs"] != payload["expected_jobs"]
    ):
        raise RuntimeError(
            f"Matrix incomplete: {payload['completed_jobs']}/{payload['expected_jobs']}"
        )

    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    best_by_model = {}
    for model, model_rows in by_model.items():
        best_by_model[model] = min(
            model_rows,
            key=lambda row: (
                -float(row["success_rate"]),
                float(row.get("gfvd", float("inf"))),
                float(row.get("lpips", float("inf"))),
            ),
        )

    schedule_effect = paired_effect(
        rows,
        keys=("model", "solver", "steps"),
        dimension="schedule",
        left_value="uniform",
        right_value="logit_normal",
    )
    step_effect = paired_effect(
        rows,
        keys=("model", "schedule", "solver"),
        dimension="steps",
        left_value=10,
        right_value=20,
    )
    heun_at_fixed_n = paired_effect(
        rows,
        keys=("model", "schedule", "steps"),
        dimension="solver",
        left_value="euler",
        right_value="heun",
    )
    midpoint_at_fixed_n = paired_effect(
        rows,
        keys=("model", "schedule", "steps"),
        dimension="solver",
        left_value="euler",
        right_value="midpoint",
    )

    near_matched_mfe20 = [
        row
        for row in rows
        if (row["solver"] == "euler" and row["steps"] == 20)
        or (row["solver"] in {"heun", "midpoint"} and row["steps"] == 10)
    ]
    near_matched_mfe20.sort(
        key=lambda row: (row["model"], row["schedule"], row["solver"])
    )
    report = {
        "protocol": payload["protocol"],
        "schedule_scope_note": (
            "The selected inference schedule is applied jointly to video and action. "
            "Logit-normal shift 17 matches pixel-video training only; action training "
            "and the latent baseline use their native shift-5 scheduler sampling."
        ),
        "completed_jobs": payload["completed_jobs"],
        "best_by_model": best_by_model,
        "near_matched_mfe19_20": near_matched_mfe20,
        "paired_effects": {
            "logit_normal_minus_uniform": schedule_effect,
            "n20_minus_n10": step_effect,
            "heun_minus_euler_at_fixed_n_unmatched_mfe": heun_at_fixed_n,
            "midpoint_minus_euler_at_fixed_n_unmatched_mfe": midpoint_at_fixed_n,
        },
    }
    report_json = summary_path.parent / "matrix_report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Pixel WAM sampler/solver matrix",
        "",
        f"Completed jobs: {payload['completed_jobs']}/{payload['expected_jobs']}.",
        "Protocol: LIBERO-10 task 6, trials 0--4, seed 42, raw checkpoints.",
        "The selected inference schedule is applied jointly to video and action. "
        "Logit-normal shift 17 matches pixel-video training only; action training "
        "and the latent baseline use their native shift-5 scheduler sampling.",
        "SR has 20 percentage-point resolution because each cell has five trials.",
        "Video metrics use closed-loop future clips; gFVD is a short-horizon within-matrix diagnostic.",
        "",
        "## Best cell per model",
        "",
        "| Model | Schedule | Solver | N | MFE | SR | gFVD | PSNR | SSIM | LPIPS | Wavelet-high |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, row in sorted(best_by_model.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    model_label(model),
                    row["schedule"],
                    row["solver"],
                    str(row["steps"]),
                    str(row["model_evaluations"]),
                    f"{100.0 * row['success_rate']:.1f}%",
                    format_value(row.get("gfvd"), 2),
                    format_value(row.get("psnr"), 3),
                    format_value(row.get("ssim"), 4),
                    format_value(row.get("lpips"), 4),
                    format_value(row.get("wavelet_high_mse"), 6),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Near-matched 19--20-MFE comparison",
            "",
            "Euler N=20 and Midpoint N=10 use 20 joint model evaluations; "
            "endpoint-safe Heun N=10 uses 19 because its terminal interval falls back to Euler.",
            "",
            "| Model | Schedule | Solver | N | SR | gFVD | PSNR | SSIM | LPIPS |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in near_matched_mfe20:
        lines.append(
            "| "
            + " | ".join(
                [
                    model_label(row["model"]),
                    row["schedule"],
                    row["solver"],
                    str(row["steps"]),
                    f"{100.0 * row['success_rate']:.1f}%",
                    format_value(row.get("gfvd"), 2),
                    format_value(row.get("psnr"), 3),
                    format_value(row.get("ssim"), 4),
                    format_value(row.get("lpips"), 4),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Mean paired deltas",
            "",
            "Positive SR/PSNR/SSIM is favorable; negative gFVD/LPIPS/wavelet error is favorable.",
            "Fixed-N solver deltas are not compute-matched: midpoint uses 2N MFE and endpoint-safe Heun uses 2N-1.",
            "",
            "| Comparison | Pairs | dSR | dgFVD | dPSNR | dSSIM | dLPIPS | dWavelet-high |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, effect in report["paired_effects"].items():
        delta = effect["mean_delta"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(effect["num_pairs"]),
                    format_value(100.0 * delta.get("success_rate", 0.0), 2),
                    format_value(delta.get("gfvd"), 2),
                    format_value(delta.get("psnr"), 3),
                    format_value(delta.get("ssim"), 4),
                    format_value(delta.get("lpips"), 4),
                    format_value(delta.get("wavelet_high_mse"), 6),
                ]
            )
            + " |"
        )

    report_md = summary_path.parent / "matrix_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_md)
    print(report_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
