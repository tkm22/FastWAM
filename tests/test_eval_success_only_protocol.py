import sys
from pathlib import Path

from omegaconf import OmegaConf


LIBERO_EXPERIMENT_DIR = Path(__file__).parents[1] / "experiments" / "libero"
sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))
from experiments.libero import eval_libero_single  # noqa: E402


def test_success_only_protocol_does_not_encode_rollout_videos(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "EVALUATION": {
                "task_suite_name": "libero_10",
                "task_id": 6,
                "num_trials": 2,
                "trial_indices": None,
                "visualize_future_video": False,
                "save_rollout_videos": False,
            },
        }
    )
    monkeypatch.setattr(
        eval_libero_single,
        "get_libero_env",
        lambda task, resolution, seed: (object(), "test task"),
    )
    monkeypatch.setattr(
        eval_libero_single,
        "run_single_episode",
        lambda **kwargs: (True, [object()], [], {}),
    )

    def unexpected_video_write(*args, **kwargs):
        raise AssertionError("success-only evaluation must not encode rollout video")

    monkeypatch.setattr(
        eval_libero_single, "save_rollout_video", unexpected_video_write
    )
    result = eval_libero_single.run_single_task(
        task=object(),
        initial_states=[object(), object()],
        model=None,
        processor=None,
        cfg=cfg,
        video_dir=tmp_path / "videos",
        predicted_video_dir=tmp_path / "predicted_videos",
        action_horizon=8,
        input_w=224,
        input_h=224,
        model_device="cpu",
        lpips_model=None,
        fvd_detector=None,
    )
    assert result["successes"] == 2
    assert result["trial_indices"] == [0, 1]
    assert result["rollout_video_files"] == []
    assert "prediction_video_files" not in result
