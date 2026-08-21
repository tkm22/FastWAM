from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from fastwam import runtime
from fastwam.trainer import Wan22Trainer
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.helpers.jit_pixel_loader import build_jit_pixel_state_dict
from fastwam.models.wan22.jit_pixel_fastwam_joint import FastWAMJointJiTPixel
from fastwam.models.wan22.jit_pixel_objective import (
    add_jit_noise,
    jit_velocity_from_x0,
    sample_jit_sigma,
)
from fastwam.models.wan22.jit_pixel_packing import (
    patchify_first_frame,
    patchify_future_tubes,
    unpatchify_first_frame,
    unpatchify_future_tubes,
)
from fastwam.models.wan22.jit_pixel_video_dit import JiTPixelWanVideoDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


def video_config(*, patch: int = 16, layers: int = 1) -> dict:
    return {
        "hidden_dim": 48,
        "in_dim": 48,
        "ffn_dim": 96,
        "out_dim": 48,
        "text_dim": 24,
        "freq_dim": 16,
        "eps": 1.0e-6,
        "patch_size": (1, 2, 2),
        "num_heads": 2,
        "attn_head_dim": 24,
        "num_layers": layers,
        "has_image_input": False,
        "seperated_timestep": True,
        "require_vae_embedding": False,
        "require_clip_embedding": False,
        "fuse_vae_embedding_in_latents": True,
        "action_conditioned": False,
        "video_attention_mask_mode": "first_frame_causal",
        "pixel_patch_size": patch,
        "future_tube_size": 4,
        "bottleneck_dim": 192,
    }


@pytest.mark.parametrize(
    ("patch", "first_dim", "future_dim"),
    [(8, 192, 768), (16, 768, 3072), (32, 3072, 12288)],
)
def test_pixel_packing_round_trip(patch: int, first_dim: int, future_dim: int):
    first = torch.randn(2, 3, 64, 96)
    future = torch.randn(2, 3, 8, 64, 96)
    first_tokens = patchify_first_frame(first, patch)
    future_tokens = patchify_future_tubes(future, patch, 4)

    assert first_tokens.shape[-1] == first_dim
    assert future_tokens.shape[-1] == future_dim
    torch.testing.assert_close(
        unpatchify_first_frame(
            first_tokens, height=64, width=96, patch_size=patch
        ),
        first,
    )
    torch.testing.assert_close(
        unpatchify_future_tubes(
            future_tokens,
            frames=8,
            height=64,
            width=96,
            patch_size=patch,
            tube_size=4,
        ),
        future,
    )


def test_p16_matches_wan_1176_token_grid():
    model = JiTPixelWanVideoDiT(**video_config())
    state = model.pre_dit(
        first_frame=torch.randn(1, 3, 224, 448),
        future_x=torch.randn(1, 3, 8, 224, 448),
        timestep=torch.tensor([500.0]),
        context=torch.randn(1, 5, 24),
        context_mask=torch.ones(1, 5, dtype=torch.bool),
    )
    assert state["tokens"].shape == (1, 1176, 48)
    assert state["meta"]["grid_size"] == (3, 14, 28)
    assert state["meta"]["tokens_per_frame"] == 392
    assert model.post_dit(state["tokens"], state).shape == (1, 3, 8, 224, 448)


def test_dynamic_shapes_and_invalid_sizes():
    model = JiTPixelWanVideoDiT(**video_config(patch=8))
    state = model.pre_dit(
        first_frame=torch.randn(2, 3, 32, 48),
        future_x=torch.randn(2, 3, 12, 32, 48),
        timestep=torch.tensor([100.0, 200.0]),
        context=torch.randn(2, 3, 24),
    )
    assert state["tokens"].shape == (2, 96, 48)

    with pytest.raises(ValueError, match="divisible"):
        patchify_future_tubes(torch.randn(1, 3, 6, 32, 32), 8, 4)
    with pytest.raises(ValueError, match="divisible"):
        model.pre_dit(
            first_frame=torch.randn(1, 3, 30, 32),
            future_x=torch.randn(1, 3, 8, 30, 32),
            timestep=torch.tensor([1.0]),
            context=torch.randn(1, 3, 24),
        )
    with pytest.raises(ValueError, match="one of"):
        JiTPixelWanVideoDiT(**video_config(patch=12))


def test_wan_body_and_192_projection_are_copied_exactly():
    target_config = video_config()
    source_config = {
        key: value
        for key, value in target_config.items()
        if key not in {"pixel_patch_size", "future_tube_size", "bottleneck_dim"}
    }
    source = WanVideoDiT(**source_config)
    target = JiTPixelWanVideoDiT(**target_config)
    adapted = build_jit_pixel_state_dict(source.state_dict(), target)
    target.load_state_dict(adapted, strict=True)

    torch.testing.assert_close(
        target.wan_patch_projection.weight,
        source.patch_embedding.weight.flatten(1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        target.wan_patch_projection.bias,
        source.patch_embedding.bias,
        rtol=0,
        atol=0,
    )
    for key, value in source.state_dict().items():
        if key.startswith("patch_embedding") or key.startswith("head"):
            continue
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
    assert torch.count_nonzero(target.head.head.weight) == 0
    assert torch.count_nonzero(target.head.head.bias) == 0
    assert torch.count_nonzero(target.head.modulation) == 0


def test_jit_velocity_loss_matches_official_sign_convention_and_clamp():
    clean = torch.randn(3, 3, 2, 4, 4)
    noise = torch.randn_like(clean)
    sigma = torch.tensor([0.01, 0.05, 0.7])
    noisy = add_jit_noise(clean, noise, sigma)
    pred_x0 = torch.randn_like(clean)

    ours_target = jit_velocity_from_x0(noisy, clean, sigma, t_eps=0.05)
    ours_pred = jit_velocity_from_x0(noisy, pred_x0, sigma, t_eps=0.05)
    denominator = sigma.view(-1, 1, 1, 1, 1).clamp_min(0.05)
    official_target = (clean - noisy) / denominator
    official_pred = (pred_x0 - noisy) / denominator

    torch.testing.assert_close(ours_target, -official_target)
    torch.testing.assert_close(ours_pred, -official_pred)
    torch.testing.assert_close(
        (ours_target - ours_pred).square().mean(),
        (official_target - official_pred).square().mean(),
    )
    torch.testing.assert_close(
        ours_target[0], (noisy[0] - clean[0]) / 0.05
    )


def test_jit_sigma_sampling_matches_official_data_time(monkeypatch):
    standard_normal = torch.tensor([-1.0, 0.0, 1.0])
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: standard_normal.clone())

    sigma = sample_jit_sigma(3, device="cpu", p_mean=-0.8, p_std=0.8)
    official_data_t = torch.sigmoid(standard_normal * 0.8 - 0.8)
    torch.testing.assert_close(sigma, 1.0 - official_data_t)


def test_training_seeds_before_model_initialization(monkeypatch, tmp_path):
    events = []

    def fake_seed(seed, *, get_worker_init_fn):
        events.append(("seed", seed, get_worker_init_fn))

    def fake_instantiate(config, **kwargs):
        kind = "model" if "model_dtype" in kwargs else "dataset"
        events.append(("instantiate", kind))
        return object()

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            events.append(("trainer",))

        def train(self):
            events.append(("train",))

    cfg = OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "seed": 42,
            "mixed_precision": "no",
            "model": {"name": "model"},
            "data": {"train": {"name": "dataset"}, "val": None},
        }
    )
    monkeypatch.setattr(runtime, "set_global_seed", fake_seed)
    monkeypatch.setattr(runtime, "instantiate", fake_instantiate)
    monkeypatch.setattr(runtime, "Wan22Trainer", FakeTrainer)
    monkeypatch.setattr(runtime, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(runtime.misc, "register_work_dir", lambda path: None)
    monkeypatch.setattr(runtime, "_resolve_train_device", lambda: "cpu")

    runtime.run_training(cfg)

    assert events[:2] == [("seed", 42, False), ("instantiate", "model")]


def test_resume_lr_reset_preserves_configured_constant_scheduler():
    parameter = torch.nn.Parameter(torch.ones(()))
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.learning_rate = 1.0e-6
    trainer.optimizer = torch.optim.AdamW([parameter], lr=trainer.learning_rate)
    trainer.scheduler = torch.optim.lr_scheduler.ConstantLR(
        trainer.optimizer, factor=1.0, total_iters=100
    )
    trainer._configured_resume_scheduler_state = trainer.scheduler.state_dict()

    old_parameter = torch.nn.Parameter(torch.ones(()))
    old_optimizer = torch.optim.AdamW([old_parameter], lr=1.0e-4)
    old_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        old_optimizer, T_max=10
    )
    for _ in range(5):
        old_optimizer.step()
        old_scheduler.step()
    trainer.optimizer.param_groups[0]["lr"] = old_optimizer.param_groups[0]["lr"]
    trainer.optimizer.param_groups[0]["initial_lr"] = 1.0e-4
    trainer.scheduler.load_state_dict(old_scheduler.state_dict())

    trainer._use_configured_scheduler_after_resume()

    for _ in range(3):
        trainer.optimizer.step()
        trainer.scheduler.step()
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-6)


def test_resume_position_advances_at_epoch_boundary():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.train_loader = range(2170)

    assert trainer._normalize_resume_position(epoch=9, batch_in_epoch=2169) == (
        9,
        2169,
    )
    assert trainer._normalize_resume_position(epoch=9, batch_in_epoch=2170) == (
        10,
        0,
    )


def test_trainer_saves_paired_weights_and_state(tmp_path):
    class FakeModel:
        def save_checkpoint(self, path, optimizer=None, step=None):
            assert optimizer is None
            torch.save({"step": step}, path)

    class FakeAccelerator:
        is_main_process = True

        def wait_for_everyone(self):
            pass

        def unwrap_model(self, model):
            return model

        def save_state(self, output_dir):
            Path(output_dir, "rank_state.pt").write_bytes(b"state")

    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = FakeModel()
    trainer.accelerator = FakeAccelerator()
    trainer.weights_dir = str(tmp_path / "checkpoints" / "weights")
    trainer.state_dir = str(tmp_path / "checkpoints" / "state")
    Path(trainer.weights_dir).mkdir(parents=True)
    Path(trainer.state_dir).mkdir(parents=True)
    trainer.global_step = 2000
    trainer.epoch = 0
    trainer.batch_in_epoch = 2000

    paths = trainer.save_checkpoint()

    assert Path(paths["weights_path"]).name == "step_002000.pt"
    assert Path(paths["weights_path"]).is_file()
    state_path = Path(paths["state_path"])
    assert state_path.name == "step_002000"
    assert (state_path / "rank_state.pt").is_file()
    assert (state_path / "trainer_state.json").is_file()


def build_small_joint_model() -> FastWAMJointJiTPixel:
    video = JiTPixelWanVideoDiT(**video_config())
    action = ActionDiT(
        action_dim=7,
        hidden_dim=48,
        ffn_dim=96,
        num_heads=2,
        attn_head_dim=24,
        num_layers=1,
        text_dim=24,
        freq_dim=16,
        eps=1.0e-6,
    )
    mot = MoT(
        mixtures={"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    )
    return FastWAMJointJiTPixel(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=None,
        text_dim=24,
        proprio_dim=None,
        device="cpu",
        torch_dtype=torch.float32,
    )


def test_single_batch_backward_and_future_output_shape():
    model = build_small_joint_model()
    sample = {
        "video": torch.randn(2, 3, 9, 32, 32).clamp(-1, 1),
        "action": torch.randn(2, 32, 7),
        "context": torch.randn(2, 4, 24),
        "context_mask": torch.ones(2, 4, dtype=torch.bool),
        "image_is_pad": torch.tensor(
            [[False] * 9, [False, False, False, False, False, True, True, True, True]]
        ),
        "action_is_pad": torch.tensor(
            [[False] * 32, [False] * 16 + [True] * 16]
        ),
    }
    loss, metrics = model.training_loss(sample)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss_video", "loss_action", "pred_x0_mse", "sigma_mean"}

    parameters = {
        "first_down": model.video_expert.first_patch_down.weight,
        "future_down": model.video_expert.future_patch_down.weight,
        "bridge": model.video_expert.wan_patch_projection.weight,
        "head": model.video_expert.head.head.weight,
        "wan_body": model.video_expert.blocks[0].self_attn.q.weight,
        "action": model.action_expert.action_encoder.weight,
    }
    for name, parameter in parameters.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

    video_pre = model.video_expert.pre_dit(
        first_frame=sample["video"][:, :, 0],
        future_x=sample["video"][:, :, 1:],
        timestep=torch.ones(2),
        context=sample["context"],
        context_mask=sample["context_mask"],
    )
    output = model.video_expert.post_dit(video_pre["tokens"], video_pre)
    assert output.shape == (2, 3, 8, 32, 32)


def test_video_loss_ignores_padded_future_frames():
    pred = torch.zeros(2, 3, 8, 4, 4)
    target = torch.zeros_like(pred)
    target[0, :, 4:] = 100.0
    future_is_pad = torch.tensor(
        [[False] * 4 + [True] * 4, [False] * 8]
    )

    loss = FastWAMJointJiTPixel._masked_temporal_mse(
        pred, target, future_is_pad
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss))


def test_action_and_joint_euler_inference_paths_match():
    model = build_small_joint_model().eval()
    common = {
        "prompt": None,
        "input_image": torch.randn(3, 32, 32).clamp(-1, 1),
        "action_horizon": 32,
        "num_video_frames": 9,
        "context": torch.randn(1, 4, 24),
        "context_mask": torch.ones(1, 4, dtype=torch.bool),
        "num_inference_steps": 2,
        "seed": 7,
        "rand_device": "cpu",
    }
    action_only = model.infer_action(**common)
    assert action_only["action"].shape == (32, 7)
    assert torch.isfinite(action_only["action"]).all()

    joint = model.infer_joint(**common)
    assert len(joint["video"]) == 9
    assert joint["action"].shape == (32, 7)
    assert torch.isfinite(joint["action"]).all()
    torch.testing.assert_close(action_only["action"], joint["action"], rtol=0, atol=0)


def test_joint_attention_preserves_fastwam_joint_directionality():
    model = build_small_joint_model()
    spatial_tokens = (32 // 16) * (32 // 16)
    video_tokens = 3 * spatial_tokens
    action_tokens = 32
    mask = model._build_mot_attention_mask(
        video_tokens, action_tokens, spatial_tokens, torch.device("cpu")
    )

    assert not mask[:spatial_tokens, spatial_tokens:video_tokens].any()
    assert mask[video_tokens:, :video_tokens].all()
    assert mask[video_tokens:, video_tokens:].all()
    assert not mask[:video_tokens, video_tokens:].any()


def test_checkpoint_metadata_is_strict():
    model = build_small_joint_model()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "model.pt"
        model.save_checkpoint(checkpoint, step=3)
        payload = model.load_checkpoint(checkpoint)
        assert payload["jit_pixel"] == model._checkpoint_metadata()

        payload["jit_pixel"]["pixel_patch_size"] = 8
        torch.save(payload, checkpoint)
        with pytest.raises(ValueError, match="metadata mismatch"):
            model.load_checkpoint(checkpoint)


def test_full_state_resume_validates_matching_raw_checkpoint():
    model = build_small_joint_model()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_root = Path(tmp) / "checkpoints"
        state_dir = checkpoint_root / "state" / "step_000003"
        weights_path = checkpoint_root / "weights" / "step_000003.pt"
        state_dir.mkdir(parents=True)
        weights_path.parent.mkdir(parents=True)
        model.save_checkpoint(weights_path, step=3)
        model.validate_training_state(str(state_dir))

        payload = torch.load(weights_path, map_location="cpu")
        payload["jit_pixel"]["future_tube_size"] = 2
        torch.save(payload, weights_path)
        with pytest.raises(ValueError, match="metadata mismatch"):
            model.validate_training_state(str(state_dir))

        payload["jit_pixel"] = model._checkpoint_metadata()
        payload["step"] = 2
        torch.save(payload, weights_path)
        with pytest.raises(ValueError, match="raw/state step mismatch"):
            model.validate_training_state(str(state_dir))


def test_pixel_model_source_has_no_asym_or_vae_runtime_imports():
    source_paths = [
        Path("src/fastwam/models/wan22/jit_pixel_fastwam_joint.py"),
        Path("src/fastwam/models/wan22/helpers/jit_pixel_loader.py"),
    ]
    forbidden = ("wan_video_vae", "oklab", "A_first", "A_future", "lpips", "asymflow")
    joined = "\n".join(path.read_text(encoding="utf-8").lower() for path in source_paths)
    for term in forbidden:
        assert term.lower() not in joined


def test_importing_pixel_model_does_not_import_vae_or_asym_modules():
    code = """
import sys
import fastwam.models.wan22.jit_pixel_fastwam_joint
forbidden = ('wan_video_vae', 'asymflow', 'oklab', 'lpips')
loaded = [name for name in sys.modules if any(term in name.lower() for term in forbidden)]
assert loaded == [], loaded
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
