from pathlib import Path

from scripts.run_sampler_solver_400_video import (
    ModelSpec,
    build_cells,
    build_command,
    expected_model_evaluations,
)


def _model(tmp_path: Path) -> ModelSpec:
    return ModelSpec(
        name="test_model",
        root=tmp_path,
        kind="pixel",
        run_dir=tmp_path / "run",
        checkpoint=tmp_path / "run/checkpoints/weights/step_021700.pt",
        dataset_stats=tmp_path / "run/dataset_stats.json",
        projection_artifact=tmp_path / "projection.pt",
        prediction_type="x0",
        vr_enabled=True,
        lpips_enabled=True,
    )


def test_matrix_has_six_400_episode_cells(tmp_path):
    cells = build_cells(tmp_path / "output", _model(tmp_path))
    assert len(cells) == 6
    assert {(cell.schedule, cell.solver) for cell in cells} == {
        (schedule, solver)
        for schedule in ("uniform", "logit_normal")
        for solver in ("euler", "heun", "midpoint")
    }
    assert expected_model_evaluations("euler") == 10
    assert expected_model_evaluations("heun") == 19
    assert expected_model_evaluations("midpoint") == 20


def test_command_enforces_video_trials_and_one_worker_per_gpu(tmp_path):
    cells = build_cells(tmp_path / "output", _model(tmp_path))
    uniform = next(
        cell for cell in cells if cell.schedule == "uniform" and cell.solver == "euler"
    )
    logit = next(
        cell
        for cell in cells
        if cell.schedule == "logit_normal" and cell.solver == "midpoint"
    )
    uniform_command = build_command(uniform, 8)
    logit_command = build_command(logit, 8)
    required = {
        "EVALUATION.num_trials=10",
        "EVALUATION.trial_indices=[0,1,2,3,4,5,6,7,8,9]",
        "EVALUATION.visualize_future_video=true",
        "EVALUATION.save_rollout_videos=true",
        "EVALUATION.save_prediction_clip_videos=false",
        "EVALUATION.compute_lpips=true",
        "EVALUATION.compute_gfvd=true",
        "EVALUATION.num_inference_steps=10",
        "MULTIRUN.num_gpus=8",
        "MULTIRUN.max_tasks_per_gpu=1",
        "model.asymflow.prediction_type=x0",
    }
    assert required.issubset(uniform_command)
    assert "EVALUATION.inference_schedule_shift=null" in uniform_command
    assert "EVALUATION.inference_schedule_shift=17.0" in logit_command
    assert "EVALUATION.ode_solver=midpoint" in logit_command


def test_latent_command_has_no_pixel_asymflow_overrides(tmp_path):
    run_dir = tmp_path / "latent_run"
    model = ModelSpec(
        name="latent",
        root=tmp_path,
        kind="latent",
        run_dir=run_dir,
        checkpoint=run_dir / "checkpoints/weights/step_021700.pt",
        dataset_stats=run_dir / "dataset_stats.json",
    )
    command = build_command(build_cells(tmp_path / "output", model)[0], 8)
    assert not any(value.startswith("model.asymflow.") for value in command)
    assert not any(value.startswith("model.projection_artifact_path=") for value in command)
