import pytest
import torch

import fastwam.models.wan22.fastwam as fastwam_module

from asymflow.color import (
    DEFAULT_LIBERO_OKLAB_MEAN,
    DEFAULT_LIBERO_OKLAB_STD,
    OklabColorEncoder,
)
from asymflow.projection import (
    ProjectionArtifact,
    fit_orthogonal_procrustes,
    scale_from_projected_energy,
)
from asymflow.velocity import asymflow_velocity, x0_prediction_velocity
from asymflow.training import (
    build_vr_lpips_gate,
    build_vr_target,
    calc_shifted_signal_ratio,
    compute_vr_coefficient,
    sample_logit_normal_sigma,
)
from asymflow.video_packing import (
    patchify_first_frame,
    patchify_future_tubes,
    unpatchify_future_tubes,
)
from fastwam.models.wan22.helpers.loader import build_asym_pixel_state_dict
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.fastwam_idm import FastWAMIDM
from fastwam.models.wan22.fastwam_joint import FastWAMJoint
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


def test_oklab_transform_supports_images_and_videos():
    color = OklabColorEncoder(
        mean=(0.5, 0.01, -0.02),
        std=(0.2, 0.03, 0.04),
    )
    image = torch.rand(2, 3, 8, 8) * 2 - 1
    image_lab = color.encode(image)
    video_lab = color.encode(image.unsqueeze(2))
    assert image_lab.shape == image.shape
    assert video_lab.shape == image.unsqueeze(2).shape
    torch.testing.assert_close(image_lab, video_lab[:, :, 0])
    torch.testing.assert_close(
        color.decode(image_lab),
        image,
        atol=2e-5,
        rtol=2e-5,
    )


def test_oklab_default_statistics_match_configured_libero_artifact():
    color = OklabColorEncoder()
    torch.testing.assert_close(
        color.affine_mean, torch.tensor(DEFAULT_LIBERO_OKLAB_MEAN)
    )
    torch.testing.assert_close(
        color.affine_std, torch.tensor(DEFAULT_LIBERO_OKLAB_STD)
    )


def test_oklab_decode_clamps_out_of_gamut_linear_rgb():
    color = OklabColorEncoder(mean=(0.5, 0.0, 0.0), std=(0.2, 0.2, 0.2))
    decoded = color.decode(torch.randn(2, 3, 4, 8, 8) * 100)
    assert torch.isfinite(decoded).all()
    assert decoded.min() >= -1.0
    assert decoded.max() <= 1.0


def test_oklab_decode_has_finite_gradient_at_black():
    """The inactive power branch of torch.where must stay differentiable."""
    linear_rgb = torch.zeros(1, 3, 2, 2, requires_grad=True)
    OklabColorEncoder.lrgb_to_srgb(linear_rgb).sum().backward()
    assert linear_rgb.grad is not None
    assert torch.isfinite(linear_rgb.grad).all()


def test_oklab_legacy_checkpoint_keeps_corrected_color_buffers():
    """Old state keys must not undo the Oklab/projection-artifact correction."""
    color = OklabColorEncoder()
    expected = {name: value.detach().clone() for name, value in color.state_dict().items()}
    legacy = {
        "mean": torch.tensor((0.5, 0.0, 0.0)),
        "std": torch.tensor((0.2, 0.1, 0.1)),
        "rgb_to_lms": torch.eye(3),
        "lms_to_oklab": torch.eye(3),
        "oklab_to_lms": torch.eye(3),
        "lms_to_rgb": torch.eye(3),
    }

    with pytest.warns(UserWarning, match="pre-Oklab-fix checkpoint"):
        result = color.load_state_dict(legacy, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
    for name, value in expected.items():
        torch.testing.assert_close(getattr(color, name), value)


def test_asymflow_logit_normal_and_patch_vr_helpers():
    sigma = sample_logit_normal_sigma(
        32, device=torch.device("cpu"), dtype=torch.float32, shift=17.0
    )
    assert torch.all((sigma > 0) & (sigma < 1))
    signal = calc_shifted_signal_ratio(sigma, 0.3)
    assert torch.all((signal >= 0) & (signal <= 1))

    full = torch.randn(2, 3, 8, 32, 32)
    pred = torch.randn_like(full, requires_grad=True)
    low = torch.randn_like(full)
    ref_low = torch.randn_like(full)
    coefficient, low_diff = compute_vr_coefficient(full, pred, low, ref_low)
    assert coefficient.shape == (2, 3, 1, 2, 1, 1)
    assert low_diff.shape == (2, 3, 4096, 2, 1, 1)
    assert torch.all((coefficient >= 0) & (coefficient <= 1))
    gate = build_vr_lpips_gate(coefficient, frames=8, height=32, width=32)
    assert gate.shape == (2, 1, 8, 32, 32)
    # The adaptive coefficient is deliberately detached from the main output.
    assert not coefficient.requires_grad


def test_inference_logit_normal_schedule_uses_training_quantiles():
    scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000, shift=5.0
    )
    uniform_t, uniform_delta = scheduler.build_inference_schedule(
        4, torch.device("cpu"), torch.float32
    )
    lognormal_t, lognormal_delta = scheduler.build_inference_schedule(
        4,
        torch.device("cpu"),
        torch.float32,
        schedule="logit_normal",
        logit_normal_shift=17.0,
    )

    assert uniform_t[0] == 1000
    assert lognormal_t[0] == 1000
    assert uniform_delta.sum() == pytest.approx(-1.0)
    assert lognormal_delta.sum() == pytest.approx(-1.0)
    assert torch.all(uniform_delta < 0)
    assert torch.all(lognormal_delta < 0)
    assert not torch.allclose(uniform_t, lognormal_t)


@pytest.mark.parametrize(
    ("ode_solver", "expected_scale"),
    [("euler", 0.25), ("heun", 0.3125), ("midpoint", 0.390625)],
)
def test_joint_ode_solvers_advance_video_and_action_together(
    ode_solver, expected_scale
):
    class IdentityColor:
        @staticmethod
        def encode(x):
            return x

    class TinyVideoExpert:
        future_tube_size = 4
        action_conditioned = False

    class TinyActionExpert:
        action_dim = 1

    model = object.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.color = IdentityColor()
    model.video_expert = TinyVideoExpert()
    model.action_expert = TinyActionExpert()
    model.infer_video_scheduler = WanContinuousFlowMatchScheduler(shift=1.0)
    model.infer_action_scheduler = WanContinuousFlowMatchScheduler(shift=1.0)
    model.asymflow = {"clamp_denoised": False}
    model._pixel_context_for_infer = lambda *args: (
        torch.zeros(1, 1, 1),
        torch.ones(1, 1, dtype=torch.bool),
    )
    model._predict_joint_noise = lambda **kwargs: (
        kwargs["future_x"],
        kwargs["latents_action"],
    )

    seed = 7
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.randn((1, 3, 4, 32, 32), generator=generator)
    initial_action = torch.randn((1, 1, 1), generator=generator)
    output = model.infer_joint(
        prompt=None,
        input_image=torch.zeros(1, 3, 32, 32),
        num_video_frames=5,
        action_horizon=1,
        context=torch.zeros(1, 1, 1),
        context_mask=torch.ones(1, 1, dtype=torch.bool),
        num_inference_steps=2,
        sigma_shift=1.0,
        ode_solver=ode_solver,
        seed=seed,
        return_video=False,
    )

    torch.testing.assert_close(
        output["action"], initial_action[0] * expected_scale
    )
    expected_evals = {"euler": 2, "heun": 3, "midpoint": 4}[ode_solver]
    assert output["inference"]["num_model_evaluations"] == expected_evals
    expected_policy = "terminal_euler" if ode_solver == "heun" else None
    assert output["inference"]["endpoint_policy"] == expected_policy


def test_video_vr_matches_upstream_channelwise_patch_formula():
    """Video VR must preserve the channel dimension as upstream VR does."""
    full = torch.randn(2, 3, 8, 64, 96)
    pred = torch.randn_like(full, requires_grad=True)
    low = torch.randn_like(full)
    ref_low = torch.randn_like(full)
    signal_ratio = torch.tensor([0.2, 0.8])

    def source_patchify_video(x):
        batch, channels, frames, height, width = x.shape
        return x.reshape(batch, channels, frames // 4, 4, height // 32, 32, width // 32, 32).permute(
            0, 1, 3, 5, 7, 2, 4, 6
        ).reshape(batch, channels, 4 * 32 * 32, frames // 4, height // 32, width // 32)

    def source_unpatchify_video(x):
        batch, channels, _, tube_groups, height_groups, width_groups = x.shape
        return x.reshape(batch, channels, 4, 32, 32, tube_groups, height_groups, width_groups).permute(
            0, 1, 5, 2, 6, 3, 7, 4
        ).reshape(batch, channels, tube_groups * 4, height_groups * 32, width_groups * 32)

    ref_low_diff = source_patchify_video(low - ref_low)
    ref_full_diff = source_patchify_video(full - pred.detach())
    ref_coefficient = (
        (ref_full_diff * ref_low_diff).mean(dim=2, keepdim=True)
        / ref_low_diff.square().mean(dim=2, keepdim=True).clamp_min(1e-4)
    ).clamp_(0, 1)
    ref_target = full - (1 - signal_ratio[:, None, None, None, None]) * source_unpatchify_video(
        ref_coefficient * ref_low_diff
    )
    ref_gate = source_unpatchify_video(
        ref_coefficient.expand_as(ref_low_diff).square().mean(dim=1, keepdim=True).sqrt()
    )

    coefficient, low_diff = compute_vr_coefficient(full, pred, low, ref_low)
    target = build_vr_target(full, coefficient, low_diff, signal_ratio)
    gate = build_vr_lpips_gate(coefficient, frames=8, height=64, width=96)
    torch.testing.assert_close(coefficient, ref_coefficient)
    torch.testing.assert_close(low_diff, ref_low_diff)
    torch.testing.assert_close(target, ref_target)
    torch.testing.assert_close(gate, ref_gate)


def test_asymflow_vr_loss_uses_shared_noise_and_backpropagates():
    class ZeroTeacher(torch.nn.Module):
        def forward(self, x, **kwargs):
            return torch.zeros_like(x[:, :, 1:])

    model = object.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.train_video_scheduler = WanContinuousFlowMatchScheduler()
    model.asymflow = {
        "sigma_min": 5e-2,
        "loss_shift": 0.3,
        "lpips_enabled": False,
    }
    model._asymflow_teacher = ZeroTeacher()
    model.color = OklabColorEncoder()

    full_x0 = torch.randn(2, 3, 8, 32, 32)
    low_x0 = torch.randn_like(full_x0)
    noise = torch.randn_like(full_x0)
    timestep = torch.tensor([100.0, 700.0])
    noisy = model.train_video_scheduler.add_noise(full_x0, noise, timestep)
    prediction = torch.randn_like(full_x0, requires_grad=True)
    loss, logs = model._asymflow_video_loss(
        full_x0=full_x0,
        low_x0=low_x0,
        noisy_video=noisy,
        noise=noise,
        timestep=timestep,
        pred_velocity=prediction,
        first_frame=torch.randn(2, 3, 32, 32),
        context=torch.randn(2, 1, 4),
        context_mask=torch.ones(2, 1, dtype=torch.bool),
        image_is_pad=None,
    )
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert logs["loss_video_lpips"] == 0.0


def test_asymflow_vr_mse_has_unit_weight():
    class ZeroTeacher(torch.nn.Module):
        def forward(self, x, **kwargs):
            return torch.zeros_like(x[:, :, 1:])

    model = object.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.train_video_scheduler = WanContinuousFlowMatchScheduler()
    model.asymflow = {
        "sigma_min": 5e-2,
        "loss_shift": 0.3,
        "lpips_enabled": False,
    }
    model._asymflow_teacher = ZeroTeacher()

    full_x0 = torch.zeros(1, 3, 8, 32, 32)
    low_x0 = torch.zeros_like(full_x0)
    noise = torch.zeros_like(full_x0)
    timestep = torch.tensor([500.0])
    noisy = model.train_video_scheduler.add_noise(full_x0, noise, timestep)
    pred_velocity = torch.ones_like(full_x0)
    loss, logs = model._asymflow_video_loss(
        full_x0=full_x0,
        low_x0=low_x0,
        noisy_video=noisy,
        noise=noise,
        timestep=timestep,
        pred_velocity=pred_velocity,
        first_frame=torch.zeros(1, 3, 32, 32),
        context=torch.zeros(1, 1, 4),
        context_mask=torch.ones(1, 1, dtype=torch.bool),
        image_is_pad=None,
    )

    assert float(loss) == pytest.approx(1.0)
    assert logs["loss_video_mse"] == pytest.approx(1.0)
    assert logs["loss_video_lpips"] == 0.0


def test_asymflow_no_vr_uses_unscaled_standard_velocity_loss():
    model = object.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.train_video_scheduler = WanContinuousFlowMatchScheduler()

    full_x0 = torch.randn(2, 3, 8, 32, 32)
    noise = torch.randn_like(full_x0)
    timestep = torch.tensor([20.0, 700.0])
    prediction = torch.randn_like(full_x0, requires_grad=True)
    target_velocity = model.train_video_scheduler.training_target(
        full_x0, noise, timestep
    )
    loss = model._pixel_video_loss(prediction, target_velocity, None).mean()
    expected = (prediction.float() - target_velocity.float()).square().mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_asymflow_legacy_timestep_mode_uses_scheduler_sampler():
    class RecordingScheduler:
        num_train_timesteps = 1000

        def __init__(self):
            self.call = None

        def sample_training_t(self, batch_size, device, dtype):
            self.call = (batch_size, device, dtype)
            return torch.full((batch_size,), 123.0, device=device, dtype=dtype)

    model = object.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.asymflow_enabled = True
    model.asymflow = {"timestep_sampling": "scheduler_uniform"}
    model.train_video_scheduler = RecordingScheduler()
    model.device = torch.device("cpu")

    batch_size = 3
    dtype = torch.float32
    got = model._sample_video_timestep(batch_size, dtype)

    assert model.train_video_scheduler.call == (batch_size, model.device, dtype)
    torch.testing.assert_close(got, torch.full((batch_size,), 123.0))


def test_asymflow_default_timestep_mode_remains_shifted_logit_normal():
    class NoSchedulerSampling:
        num_train_timesteps = 1000

        def sample_training_t(self, *args, **kwargs):
            raise AssertionError("logit-normal mode must not call the scheduler sampler")

    model = object.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.asymflow_enabled = True
    model.asymflow = {"timestep_shift": 17.0}
    model.train_video_scheduler = NoSchedulerSampling()
    model.device = torch.device("cpu")

    torch.manual_seed(7)
    got = model._sample_video_timestep(4, torch.float32)
    torch.manual_seed(7)
    expected = sample_logit_normal_sigma(
        4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        shift=17.0,
    ) * 1000.0
    torch.testing.assert_close(got, expected)


def test_asymflow_rejects_unknown_timestep_mode():
    model = object.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.asymflow_enabled = True
    model.asymflow = {"timestep_sampling": "unknown"}
    model.train_video_scheduler = WanContinuousFlowMatchScheduler()
    model.device = torch.device("cpu")

    with pytest.raises(ValueError, match="timestep_sampling"):
        model._sample_video_timestep(1, torch.float32)


def test_idm_training_uses_shared_asymflow_video_loss():
    class Scheduler:
        num_train_timesteps = 1000

        @staticmethod
        def sample_training_t(batch_size, device, dtype):
            return torch.full((batch_size,), 500.0, device=device, dtype=dtype)

        @staticmethod
        def add_noise(clean, noise, timestep):
            return clean

        @staticmethod
        def training_target(clean, noise, timestep):
            return torch.zeros_like(clean)

        @staticmethod
        def training_weight(timestep):
            return torch.ones_like(timestep)

    class VideoExpert(torch.nn.Module):
        @staticmethod
        def build_video_to_video_mask(seq_len, tokens_per_group, device):
            return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

        def pre_dit(self, *, future_x, context, context_mask, **kwargs):
            batch = future_x.shape[0]
            return {
                "tokens": torch.zeros(batch, 1, 4),
                "freqs": torch.zeros(1, 1),
                "context": context,
                "context_mask": context_mask,
                "t_mod": torch.zeros(batch, 1, 1),
                "meta": {"tokens_per_group": 1},
                "future_x": future_x,
            }

        @staticmethod
        def post_dit(tokens, pre):
            return torch.zeros_like(pre["future_x"])

    class ActionExpert(torch.nn.Module):
        def pre_dit(self, *, action_tokens, context, context_mask, **kwargs):
            batch, horizon, _ = action_tokens.shape
            return {
                "tokens": torch.zeros(batch, horizon, 4),
                "freqs": torch.zeros(horizon, 1),
                "context": context,
                "context_mask": context_mask,
                "t_mod": torch.zeros(batch, horizon, 1),
            }

        @staticmethod
        def post_dit(tokens, pre):
            return torch.zeros(tokens.shape[0], tokens.shape[1], 2)

    class IdentityMoT(torch.nn.Module):
        def forward(self, *, embeds_all, **kwargs):
            return embeds_all

    model = object.__new__(FastWAMIDM)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.loss_lambda_video = 1.0
    model.loss_lambda_action = 1.0
    model.train_video_scheduler = Scheduler()
    model.train_action_scheduler = Scheduler()
    model.video_expert = VideoExpert()
    model.action_expert = ActionExpert()
    model.mot = IdentityMoT()
    inputs = {
        "first_frame": torch.zeros(1, 3, 2, 2),
        "future_pixels": torch.zeros(1, 3, 8, 2, 2),
        "low_rank_future": torch.zeros(1, 3, 8, 2, 2),
        "context": torch.zeros(1, 1, 4),
        "context_mask": torch.ones(1, 1, dtype=torch.bool),
        "action": torch.zeros(1, 2, 2),
        "action_is_pad": None,
        "image_is_pad": None,
    }
    model.build_inputs = lambda sample, tiled: inputs
    model._sample_video_timestep = lambda batch_size, dtype: torch.full(
        (batch_size,), 500.0, dtype=dtype
    )
    called = {}

    def shared_video_loss(**kwargs):
        called.update(kwargs)
        return kwargs["pred_video"].sum() * 0 + 2.0, {"shared_vr": 1.0}

    model._video_training_loss = shared_video_loss
    loss, logs = model.training_loss({})

    assert called["inputs"] is inputs
    assert float(loss) == pytest.approx(2.0)
    assert logs["shared_vr"] == 1.0


def test_vr_auxiliaries_load_only_for_vr_training(monkeypatch):
    class TinyExpert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.flow_num_train_timesteps = None

    artifact = ProjectionArtifact(
        A_first=torch.zeros(1, 1),
        A_future=torch.zeros(1, 1),
        scale_first=torch.ones(()),
        scale_future=torch.ones(()),
        oklab_mean=torch.zeros(3),
        oklab_std=torch.ones(3),
        metadata={},
    )
    vae_loads = []
    monkeypatch.setattr(fastwam_module.ProjectionArtifact, "load", lambda path: artifact)
    monkeypatch.setattr(fastwam_module, "validate_projection_artifact", lambda value: None)
    monkeypatch.setattr(
        fastwam_module,
        "_load_frozen_wan_vae",
        lambda path, device, dtype: vae_loads.append(path) or torch.nn.Identity(),
    )

    def build_model(*, vr_enabled, load_training_auxiliaries):
        return FastWAM(
            video_expert=TinyExpert(),
            action_expert=TinyExpert(),
            mot=TinyExpert(),
            text_dim=4,
            projection_artifact_path="artifact.pt",
            asymflow={
                "enabled": True,
                "vr_enabled": vr_enabled,
                "lpips_enabled": vr_enabled,
                "vae_path": "vae.pt",
            },
            load_training_auxiliaries=load_training_auxiliaries,
        )

    inference_model = build_model(vr_enabled=True, load_training_auxiliaries=False)
    no_vr_training_model = build_model(
        vr_enabled=False, load_training_auxiliaries=True
    )
    assert inference_model._asymflow_teacher is None
    assert inference_model._asymflow_vae is None
    assert no_vr_training_model._asymflow_teacher is None
    assert no_vr_training_model._asymflow_vae is None
    assert not vae_loads

    vr_training_model = build_model(vr_enabled=True, load_training_auxiliaries=True)
    assert vr_training_model._asymflow_teacher is not None
    assert vr_training_model._asymflow_vae is not None
    assert vae_loads == ["vae.pt"]


def test_asymflow_vr_lpips_keeps_absolute_gate_and_timestep_weighting():
    class ZeroTeacher(torch.nn.Module):
        def forward(self, x, **kwargs):
            return torch.zeros_like(x[:, :, 1:])

    class UnitSpatialLPIPS(torch.nn.Module):
        def forward(self, pred, target):
            return torch.ones_like(pred[:, :1])

    model = object.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.train_video_scheduler = WanContinuousFlowMatchScheduler()
    model.torch_dtype = torch.float32
    model.asymflow = {
        "sigma_min": 5e-2,
        "loss_shift": 0.3,
        "lpips_enabled": True,
    }
    model._asymflow_teacher = ZeroTeacher()
    object.__setattr__(model, "_asymflow_lpips", UnitSpatialLPIPS())
    model.color = OklabColorEncoder()

    full_x0 = torch.randn(1, 3, 8, 32, 32)
    low_x0 = torch.randn_like(full_x0)
    noise = torch.randn_like(full_x0)
    timestep = torch.tensor([500.0])
    noisy = model.train_video_scheduler.add_noise(full_x0, noise, timestep)
    pred_velocity = torch.randn_like(full_x0)
    _, logs = model._asymflow_video_loss(
        full_x0=full_x0,
        low_x0=low_x0,
        noisy_video=noisy,
        noise=noise,
        timestep=timestep,
        pred_velocity=pred_velocity,
        first_frame=torch.randn(1, 3, 32, 32),
        context=torch.randn(1, 1, 4),
        context_mask=torch.ones(1, 1, dtype=torch.bool),
        image_is_pad=None,
    )

    sigma = timestep / model.train_video_scheduler.num_train_timesteps
    sigma_view = sigma.view(-1, 1, 1, 1, 1)
    low_noisy = (1 - sigma_view) * low_x0 + sigma_view * noise
    pred_x0 = noisy - sigma_view * pred_velocity
    coefficient, _ = compute_vr_coefficient(full_x0, pred_x0, low_x0, low_noisy)
    gate = build_vr_lpips_gate(coefficient, frames=8, height=32, width=32)
    expected = (
        gate
        * (calc_shifted_signal_ratio(sigma, 0.3) / sigma.clamp_min(5e-2).square()).view(1, 1, 1, 1, 1)
    ).mean() * 0.2
    assert logs["loss_video_lpips"] == pytest.approx(float(expected), rel=1e-6)


def test_fastwam_checkpoint_keeps_original_minimal_payload(tmp_path):
    model = object.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.mot = torch.nn.Linear(2, 2)
    model.proprio_encoder = None
    model.torch_dtype = torch.float32
    checkpoint = tmp_path / "pixel.pt"
    model.save_checkpoint(checkpoint)

    payload = torch.load(checkpoint, weights_only=False)
    assert set(payload) == {"mot", "step", "torch_dtype"}
    model.load_checkpoint(checkpoint)


def test_denoised_callback_reencodes_clamped_rgb_each_step():
    model = object.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.asymflow = {"clamp_denoised": True}
    model.infer_video_scheduler = WanContinuousFlowMatchScheduler()
    model.color = OklabColorEncoder()
    future = torch.randn(1, 3, 8, 32, 32)
    velocity = torch.randn_like(future)
    timestep = torch.tensor([500.0])
    got = model._clamped_video_velocity(future, velocity, timestep)
    denoised = future - 0.5 * got
    rgb = model.color.decode(denoised)
    assert torch.isfinite(got).all()
    assert rgb.min() >= -1.0
    assert rgb.max() <= 1.0


def test_procrustes_and_scale_recover_known_pixel_lift():
    latent = torch.randn(4096, 16, dtype=torch.float64)
    basis = torch.linalg.qr(
        torch.randn(64, 16, dtype=torch.float64), mode="reduced"
    ).Q
    scale = torch.tensor(2.75, dtype=torch.float64)
    pixel = scale * latent @ basis.T

    fitted = fit_orthogonal_procrustes(pixel.T @ latent)
    fitted_scale = scale_from_projected_energy(
        (pixel @ fitted).square().sum(),
        latent.square().sum(),
    )

    torch.testing.assert_close(fitted, basis, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(fitted.T @ fitted, torch.eye(16, dtype=torch.float64))
    torch.testing.assert_close(fitted_scale, scale)


def test_asymmetric_reconstruction_s_equals_one():
    # At s=1, calibrated u_A=P eps-x0 reduces to the uncalibrated target.
    A = torch.linalg.qr(torch.randn(256, 16), mode="reduced").Q
    x0, eps = torch.randn(2, 5, 256), torch.randn(2, 5, 256)
    sigma = torch.tensor([0.2, 0.8])
    x_sigma = (1 - sigma[:, None, None]) * x0 + sigma[:, None, None] * eps
    u_asym = (eps @ A) @ A.T - x0
    got = asymflow_velocity(u_asym, x_sigma, sigma, torch.tensor(1.0), A)
    torch.testing.assert_close(got, eps - x0, atol=2e-5, rtol=2e-5)


def test_scale_calibrated_asymmetric_reconstruction():
    A = torch.linalg.qr(torch.randn(256, 16), mode="reduced").Q
    x0, eps = torch.randn(2, 5, 256), torch.randn(2, 5, 256)
    sigma = torch.tensor([0.2, 0.8])
    scale = torch.tensor(0.7)
    x_sigma = (1 - sigma[:, None, None]) * x0 + sigma[:, None, None] * eps
    u_asym_cal = (eps @ A) @ A.T - x0 / scale
    got = asymflow_velocity(u_asym_cal, x_sigma, sigma, scale, A)
    torch.testing.assert_close(got, eps - x0, atol=2e-5, rtol=2e-5)


def test_asymmetric_reconstruction_has_no_endpoint_nan():
    A = torch.linalg.qr(torch.randn(64, 8), mode="reduced").Q
    x = torch.randn(2, 64)
    u = torch.randn_like(x)
    for sigma in (
        torch.tensor([1.0, 1.0]),
        torch.tensor([1e-8, 1e-6]),
        torch.tensor([0.0, 0.0]),
    ):
        assert torch.isfinite(asymflow_velocity(u, x, sigma, torch.tensor(0.7), A)).all()


def test_x0_prediction_velocity_matches_jit_boundary_and_backpropagates():
    sigma = torch.tensor([0.2, 0.8])
    x_sigma = torch.randn(2, 3, 64)
    pred_x0 = torch.randn_like(x_sigma, requires_grad=True)

    got = x0_prediction_velocity(x_sigma, pred_x0, sigma, sigma_min=0.05)
    expected = (x_sigma - pred_x0) / sigma[:, None, None]
    torch.testing.assert_close(got, expected)

    got.square().mean().backward()
    assert pred_x0.grad is not None
    assert torch.isfinite(pred_x0.grad).all()


def test_x0_prediction_velocity_uses_endpoint_floor():
    x_sigma = torch.randn(2, 3, 64, dtype=torch.bfloat16)
    pred_x0 = torch.randn_like(x_sigma)
    got = x0_prediction_velocity(
        x_sigma, pred_x0, torch.tensor([0.0, 1e-4]), sigma_min=0.05
    )
    expected = (x_sigma.float() - pred_x0.float()) / 0.05
    assert got.dtype == torch.bfloat16
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got.float(), expected.bfloat16().float())


def test_wan_x0_head_returns_scheduler_velocity_without_asym_recovery():
    class FixedHead(torch.nn.Module):
        def __init__(self, output):
            super().__init__()
            self.register_buffer("output", output)

        def forward(self, tokens, timestep):
            del tokens, timestep
            return self.output

    model = object.__new__(WanVideoDiT)
    torch.nn.Module.__init__(model)
    model.set_prediction_type("x_prediction", x_prediction_eps=0.05)

    future_x = torch.randn(2, 3, 4, 32, 32)
    pred_x0_packed = torch.randn(2, 1, 3 * 4 * 32 * 32)
    model.head = FixedHead(pred_x0_packed)
    pre_state = {
        "t": torch.zeros(2, 2, 6, 1),
        "meta": {
            "future_x": future_x,
            "tokens_per_group": 1,
            "sigma": torch.tensor([0.2, 0.8]),
            "height": 32,
            "width": 32,
        },
    }

    got = model.post_dit(torch.zeros(2, 2, 4), pre_state)
    pred_x0 = unpatchify_future_tubes(pred_x0_packed, 4, 32, 32)
    expected = (future_x - pred_x0) / torch.tensor([0.2, 0.8]).view(
        2, 1, 1, 1, 1
    )
    torch.testing.assert_close(got, expected)
    assert model.prediction_type == "x0"

    with pytest.raises(ValueError, match="Unsupported pixel prediction type"):
        model.set_prediction_type("epsilon")


def test_asymmetric_reconstruction_uses_explicit_inference_sigma_floor():
    A = torch.linalg.qr(torch.randn(64, 8), mode="reduced").Q
    u = torch.randn(1, 64)
    x = torch.randn_like(u)
    sigma = torch.tensor([1e-6])
    scale = torch.tensor(0.7)
    got = asymflow_velocity(u, x, sigma, scale, A, sigma_min=1e-4)
    p_u = (u @ A) @ A.T
    p_x = (x @ A) @ A.T
    k = 1 / (scale + (1 - scale) * sigma)
    expected = (
        scale * k[:, None] * p_u
        + (1 - scale * k[:, None]) * p_x / 1e-4
        + (x - p_x + scale * (u - p_u)) / 1e-4
    )
    torch.testing.assert_close(got, expected)


def test_bfloat16_reconstruction_computes_formula_in_float32():
    A = torch.linalg.qr(torch.randn(64, 8), mode="reduced").Q
    x0 = torch.randn(2, 3, 64)
    eps = torch.randn_like(x0)
    sigma = torch.tensor([0.02, 0.8])
    scale = torch.tensor(0.7)
    x_sigma = ((1 - sigma[:, None, None]) * x0 + sigma[:, None, None] * eps)
    u_asym = (eps @ A) @ A.T - x0 / scale
    got = asymflow_velocity(
        u_asym.bfloat16(),
        x_sigma.bfloat16(),
        sigma.bfloat16(),
        scale,
        A,
    )
    assert got.dtype == torch.bfloat16
    u32 = u_asym.bfloat16().float()
    x32 = x_sigma.bfloat16().float()
    sigma32 = sigma.bfloat16().float()
    p_u = (u32 @ A) @ A.T
    p_x = (x32 @ A) @ A.T
    k = 1 / (scale + (1 - scale) * sigma32)
    view = (2, 1, 1)
    reference = (
        scale * k.reshape(view) * p_u
        + (1 - scale * k.reshape(view)) * p_x / sigma32.reshape(view)
        + (x32 - p_x + scale * (u32 - p_u)) / sigma32.reshape(view)
    )
    torch.testing.assert_close(
        got.float(),
        reference.bfloat16().float(),
        atol=0,
        rtol=0,
    )


def test_asymmetric_reconstruction_backpropagates_to_head_output():
    A = torch.linalg.qr(torch.randn(64, 8), mode="reduced").Q
    u_asym = torch.randn(2, 3, 64, requires_grad=True)
    velocity = asymflow_velocity(
        u_asym,
        torch.randn_like(u_asym),
        torch.tensor([0.2, 0.8]),
        torch.tensor(0.7),
        A,
    )
    velocity.square().mean().backward()
    assert u_asym.grad is not None
    assert torch.isfinite(u_asym.grad).all()


def test_wan_boundary_rewrite_is_strict_and_equivalent():
    config = {
        "hidden_dim": 64,
        "in_dim": 48,
        "ffn_dim": 128,
        "out_dim": 48,
        "text_dim": 32,
        "freq_dim": 32,
        "eps": 1e-6,
        "patch_size": (1, 2, 2),
        "num_heads": 4,
        "attn_head_dim": 16,
        "num_layers": 1,
        "has_image_input": False,
        "seperated_timestep": True,
    }
    a_first = torch.linalg.qr(torch.randn(3072, 192), mode="reduced").Q
    a_future = torch.linalg.qr(torch.randn(12288, 192), mode="reduced").Q
    artifact = ProjectionArtifact(
        A_first=a_first,
        A_future=a_future,
        scale_first=torch.tensor(0.7),
        scale_future=torch.tensor(0.8),
        oklab_mean=torch.zeros(3),
        oklab_std=torch.ones(3),
        metadata={},
    )
    target = WanVideoDiT(**config, projection_artifact=artifact.state_dict())
    target_state = target.state_dict()
    pixel_only = {
        "first_patch_embedding.weight",
        "first_patch_embedding.bias",
        "future_patch_embedding.weight",
        "future_patch_embedding.bias",
        "A_first",
        "A_future",
        "scale_first",
        "scale_future",
    }
    source_state = {
        key: value.clone()
        for key, value in target_state.items()
        if key not in pixel_only
    }
    source_state["patch_embedding.weight"] = torch.randn(64, 48, 1, 2, 2)
    source_state["patch_embedding.bias"] = torch.randn(64)
    source_state["head.head.weight"] = torch.randn(192, 64)
    source_state["head.head.bias"] = torch.randn(192)
    adapted = build_asym_pixel_state_dict(source_state, target, artifact)
    target.load_state_dict(adapted, strict=True)

    source_in = source_state["patch_embedding.weight"].flatten(1)
    torch.testing.assert_close(
        target.first_patch_embedding.weight.flatten(1),
        source_in @ a_first.T,
    )
    torch.testing.assert_close(
        target.future_patch_embedding.weight,
        source_in @ a_future.T,
    )
    torch.testing.assert_close(
        target.head.head.weight,
        a_future @ source_state["head.head.weight"],
    )

    # At sigma=0, k=1/s.  A pixel token x=sAz must therefore enter the
    # unchanged Transformer body exactly like its source latent token z.
    z = torch.randn(5, 192)
    first_x = artifact.scale_first * (z @ a_first.T)
    future_x = artifact.scale_future * (z @ a_future.T)
    source_hidden = z @ source_in.T + source_state["patch_embedding.bias"]
    first_hidden = (
        first_x / artifact.scale_first
    ) @ target.first_patch_embedding.weight.flatten(1).T
    first_hidden = first_hidden + target.first_patch_embedding.bias
    future_hidden = (
        future_x / artifact.scale_future
    ) @ target.future_patch_embedding.weight.T
    future_hidden = future_hidden + target.future_patch_embedding.bias
    torch.testing.assert_close(first_hidden, source_hidden, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(future_hidden, source_hidden, atol=2e-4, rtol=2e-4)

    # The adapted output is the source latent output lifted by A_future.
    hidden = torch.randn(5, 64)
    source_output = (
        hidden @ source_state["head.head.weight"].T
        + source_state["head.head.bias"]
    )
    pixel_output = hidden @ target.head.head.weight.T + target.head.head.bias
    torch.testing.assert_close(
        pixel_output,
        source_output @ a_future.T,
        atol=2e-4,
        rtol=2e-4,
    )


def test_libero_stride4_pixel_token_dimensions():
    assert list(range(0, 33, 4)) == [0, 4, 8, 12, 16, 20, 24, 28, 32]
    first = torch.randn(1, 3, 224, 448)
    future = torch.randn(1, 3, 8, 224, 448)
    first_tokens = patchify_first_frame(first)
    future_tokens = patchify_future_tubes(future)
    assert first_tokens.shape == (1, 98, 3072)
    assert future_tokens.shape == (1, 196, 12288)
    torch.testing.assert_close(
        unpatchify_future_tubes(future_tokens, 8, 224, 448),
        future,
    )


def test_pixel_wan_pre_and_post_dit_use_real_pixel_boundaries():
    a_first = torch.zeros(3072, 192)
    a_future = torch.zeros(12288, 192)
    a_first[:192] = torch.eye(192)
    a_future[:192] = torch.eye(192)
    artifact = ProjectionArtifact(
        A_first=a_first,
        A_future=a_future,
        scale_first=torch.tensor(0.7),
        scale_future=torch.tensor(0.8),
        oklab_mean=torch.zeros(3),
        oklab_std=torch.ones(3),
        metadata={},
    )
    model = WanVideoDiT(
        hidden_dim=64,
        in_dim=48,
        ffn_dim=128,
        out_dim=48,
        text_dim=32,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        attn_head_dim=16,
        num_layers=0,
        has_image_input=False,
        seperated_timestep=True,
        projection_artifact=artifact.state_dict(),
    )
    model.to(dtype=torch.bfloat16)
    assert model.first_patch_embedding.weight.dtype == torch.bfloat16
    assert model.A_first.dtype == torch.float32
    assert model.A_future.dtype == torch.float32
    assert model.scale_first.dtype == torch.float32
    assert model.scale_future.dtype == torch.float32
    first = torch.randn(1, 3, 32, 32)
    # The same pixel DiT supports the corrected 9-frame horizon and the
    # previous ratio-1 checkpoint's 33-frame horizon.  Here 32 future RGB
    # frames become eight future temporal groups.
    future = torch.randn(1, 3, 32, 32, 32)
    pre = model.pre_dit(
        first_frame=first,
        future_x=future,
        timestep=torch.tensor([500.0]),
        context=torch.randn(1, 2, 32, dtype=torch.bfloat16),
        context_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert pre["tokens"].shape == (1, 9, 64)
    assert pre["meta"]["grid_size"] == (9, 1, 1)
    velocity = model.post_dit(pre["tokens"], pre)
    assert velocity.shape == future.shape
    assert torch.isfinite(velocity).all()


def test_joint_action_inference_keeps_joint_video_denoising():
    class TinyJoint(FastWAMJoint):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.called = None

        def infer_joint(self, **kwargs):
            self.called = kwargs
            return {
                "video": [],
                "action": torch.zeros(kwargs["action_horizon"], 7),
            }

    model = TinyJoint()
    result = model.infer_action(
        prompt="test",
        input_image=torch.zeros(1, 3, 32, 32),
        action_horizon=32,
        num_video_frames=33,
    )
    assert model.called["num_video_frames"] == 33
    assert model.called["action_horizon"] == 32
    assert model.called["return_video"] is False
    assert result["action"].shape == (32, 7)
