from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from asymflow.color import OklabColorEncoder
from asymflow.projection import (
    ProjectionArtifact,
    validate_projection_artifact,
)
from asymflow.training import (
    build_vr_lpips_gate,
    build_vr_target,
    calc_shifted_signal_ratio,
    compute_vr_coefficient,
    lift_future_latents,
    sample_logit_normal_sigma,
    wan_future_latent_patches,
)
from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .helpers.io import load_state_dict
from .helpers.state_dict_converters import wan_video_vae_state_dict_converter
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_vae import WanVideoVAE38

logger = get_logger(__name__)


def _load_frozen_wan_vae(path: str, device: torch.device, dtype: torch.dtype) -> WanVideoVAE38:
    """Build the training-only VAE used to produce AsymFlow's low-rank state."""
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "AsymFlow VR requires the Wan VAE used for the projection fit; "
            f"checkpoint not found: {checkpoint}"
        )
    vae = WanVideoVAE38()
    state_dict = load_state_dict(str(checkpoint), torch_dtype=dtype)
    vae.load_state_dict(wan_video_vae_state_dict_converter(state_dict), strict=True)
    return vae.to(device=device, dtype=dtype).eval().requires_grad_(False)


class FastWAM(torch.nn.Module):
    """Asymmetric pixel FastWAM with no Wan VAE in its runtime graph.

    LIBERO contributes 33 observation steps and 32 actions. The dataset keeps
    RGB observations at offsets ``0,4,...,32``, so this model consumes
    ``[B,3,9,224,448]``: one clean first-frame condition and eight diffused
    future pixel frames. The future frames are packed into two four-frame
    tubes before entering the unchanged pretrained Wan Transformer body.
    """

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        projection_artifact_path: Optional[str] = None,
        asymflow: Optional[dict[str, Any]] = None,
        load_training_auxiliaries: bool = False,
    ):
        super().__init__()
        if projection_artifact_path is None:
            raise ValueError("Asym pixel FastWAM requires projection_artifact_path")
        artifact = ProjectionArtifact.load(projection_artifact_path)
        validate_projection_artifact(artifact)

        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if text_encoder is None:
                raise ValueError("text_dim is required when the text encoder is not loaded")
            text_dim = int(text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler
        self.video_expert.flow_num_train_timesteps = (
            self.train_video_scheduler.num_train_timesteps
        )

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.asymflow = dict(asymflow or {})
        self.asymflow_enabled = bool(self.asymflow.get("enabled", False))
        self.asymflow_vr_enabled = bool(self.asymflow.get("vr_enabled", True))
        self.load_training_auxiliaries = bool(load_training_auxiliaries)
        if (
            self.asymflow_enabled
            and bool(self.asymflow.get("lpips_enabled", True))
            and not self.asymflow_vr_enabled
        ):
            raise ValueError("AsymFlow LPIPS requires vr_enabled=true for its patch-wise gate")
        self.color = OklabColorEncoder(
            mean=artifact.oklab_mean.flatten().tolist(),
            std=artifact.oklab_std.flatten().tolist(),
        )
        self.to(device=self.device, dtype=self.torch_dtype)
        # Match the upstream AsymFlow color encoder: color conversion and
        # affine normalization stay in float32, then pixel states enter the
        # Transformer in its configured dtype.
        self.color.float()

        # These helpers deliberately bypass nn.Module registration.  They are
        # frozen training inputs, not model weights: DDP/ZeRO, the optimizer,
        # and main checkpoints contain only the finetuned MoT.
        object.__setattr__(self, "_asymflow_teacher", None)
        object.__setattr__(self, "_asymflow_vae", None)
        object.__setattr__(self, "_asymflow_lpips", None)
        if (
            self.load_training_auxiliaries
            and self.asymflow_enabled
            and self.asymflow_vr_enabled
        ):
            vae_path = self.asymflow.get("vae_path")
            if not vae_path:
                raise ValueError("asymflow.enabled requires asymflow.vae_path")
            teacher = deepcopy(self.video_expert).eval().requires_grad_(False)
            teacher.to(device=self.device, dtype=self.torch_dtype)
            object.__setattr__(self, "_asymflow_teacher", teacher)
            object.__setattr__(
                self,
                "_asymflow_vae",
                _load_frozen_wan_vae(str(vae_path), self.device, self.torch_dtype),
            )
            logger.info(
                "AsymFlow training auxiliaries: artifact=%s vae=%s",
                projection_artifact_path,
                vae_path,
            )

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        projection_artifact_path: Optional[str] = None,
        asymflow: Optional[dict[str, Any]] = None,
        load_training_auxiliaries: bool = False,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")
        if projection_artifact_path is None:
            raise ValueError("Pass projection_artifact_path=<fit artifact .pt>")
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
            projection_artifact_path=projection_artifact_path,
        )
        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            projection_artifact_path=projection_artifact_path,
            asymflow=asymflow,
            load_training_auxiliaries=load_training_auxiliaries,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(
            prompt, return_mask=True, add_special_tokens=True
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def build_inputs(self, sample, tiled: bool = False):
        """Convert one sampled LIBERO batch into pixel-flow model inputs.

        ``sample['video']`` must already have FastWAM's stride-4 observation
        sampling and two-camera preprocessing: ``[B,3,9,H,W]``.  This method
        does not use ``tiled`` or a VAE.  It converts RGB ``[-1,1]`` to the
        artifact-standardized Oklab space, keeps frame 0 as the clean
        ``first_frame`` condition, and returns frames 1..8 as
        ``future_pixels`` to be diffused.  When enabled, only the first
        proprio state is appended as one text-context token.
        """
        del tiled
        video = sample["video"]
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[2] != 9:
            raise ValueError(
                "pixel FastWAM expects sampled video [B,3,9,H,W]"
            )
        if video.shape[-2] % 32 or video.shape[-1] % 32:
            raise ValueError("pixel video H/W must be divisible by 32")
        if (
            "context" not in sample
            or "context_mask" not in sample
            or "action" not in sample
        ):
            raise ValueError("sample needs context, context_mask and action")

        video_rgb = video.to(self.device, torch.float32, non_blocking=True)
        pixel = self.color.encode(video_rgb)
        context = sample["context"].to(
            self.device, self.torch_dtype, non_blocking=True
        )
        context_mask = sample["context_mask"].to(
            self.device, torch.bool, non_blocking=True
        )
        if self.proprio_encoder is not None:
            proprio = sample.get("proprio")
            if proprio is None:
                raise ValueError("proprio is required when proprio_dim is enabled")
            context, context_mask = self._append_proprio_to_context(
                context,
                context_mask,
                proprio[:, 0].to(self.device, self.torch_dtype),
            )
        inputs = {
            "first_frame": pixel[:, :, 0],
            "future_pixels": pixel[:, :, 1:],
            "context": context,
            "context_mask": context_mask,
            "action": sample["action"].to(
                self.device, self.torch_dtype, non_blocking=True
            ),
            "action_is_pad": (
                None
                if sample.get("action_is_pad") is None
                else sample["action_is_pad"].to(self.device, torch.bool)
            ),
            "image_is_pad": (
                None
                if sample.get("image_is_pad") is None
                else sample["image_is_pad"].to(self.device, torch.bool)
            ),
        }
        if self.asymflow_enabled and self.asymflow_vr_enabled:
            inputs["low_rank_future"] = self._asymflow_low_rank_future(video_rgb)
        return inputs

    def _asymflow_low_rank_future(self, video_rgb: torch.Tensor) -> torch.Tensor:
        """Encode the sampled RGB clip with the frozen VAE and lift future tokens."""
        vae = self._asymflow_vae
        if vae is None:
            raise RuntimeError("AsymFlow VAE is not initialized")
        with torch.no_grad():
            z = vae.model.encode(video_rgb.to(dtype=self.torch_dtype), vae.scale)
            tokens = wan_future_latent_patches(z)
            low_rank = lift_future_latents(
                tokens,
                self.video_expert.A_future,
                self.video_expert.scale_future,
                frames=video_rgb.shape[2] - 1,
                height=video_rgb.shape[-2],
                width=video_rgb.shape[-1],
            ).to(dtype=video_rgb.dtype)
        return low_rank

    def _get_asymflow_lpips(self):
        model = self._asymflow_lpips
        if model is not None:
            return model
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "AsymFlow LPIPS is enabled but `lpips` is not installed. "
                "Install the project dependencies before training."
            ) from exc
        model = lpips.LPIPS(net="vgg", spatial=True, eval_mode=True, pnet_tune=False)
        model = model.to(
            device=self.device, dtype=self.torch_dtype
        ).eval().requires_grad_(False)
        object.__setattr__(self, "_asymflow_lpips", model)
        return model

    def _asymflow_video_loss(
        self,
        *,
        full_x0: torch.Tensor,
        low_x0: Optional[torch.Tensor],
        noisy_video: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        pred_velocity: torch.Tensor,
        first_frame: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Official AsymFLUX-style VR objective with optional LPIPS correction."""
        sigma = (timestep.float() / float(self.train_video_scheduler.num_train_timesteps))
        sigma_view = sigma.view(-1, 1, 1, 1, 1)
        sigma_clamped = sigma.clamp_min(float(self.asymflow.get("sigma_min", 5e-2)))
        pred_x0 = noisy_video.float() - sigma_view * pred_velocity.float()
        vr_enabled = bool(self.asymflow.get("vr_enabled", True))
        lpips_enabled = bool(self.asymflow.get("lpips_enabled", True))
        if not vr_enabled:
            raise RuntimeError("_asymflow_video_loss requires vr_enabled=true")

        if low_x0 is None:
            raise RuntimeError("AsymFlow VR requires the paired low-rank target")
        low_noisy = (1.0 - sigma_view) * low_x0 + sigma_view * noise
        teacher = self._asymflow_teacher
        if teacher is None:
            raise RuntimeError("AsymFlow teacher is not initialized")
        with torch.no_grad():
            ref_velocity = teacher(
                x=torch.cat([first_frame.unsqueeze(2), low_noisy], dim=2),
                timestep=timestep.to(dtype=low_noisy.dtype),
                context=context,
                context_mask=context_mask,
                action=None,
            )
            ref_low_x0 = low_noisy - sigma_view * ref_velocity.float()

        coefficient, low_diff = compute_vr_coefficient(
            full_x0.float(), pred_x0, low_x0.float(), ref_low_x0.float()
        )
        signal_ratio = calc_shifted_signal_ratio(
            sigma, float(self.asymflow.get("loss_shift", 0.3))
        )
        target_x0 = build_vr_target(
            full_x0.float(), coefficient, low_diff, signal_ratio
        )

        squared_error = (pred_x0 - target_x0).square().mean(dim=(1, 3, 4))
        if image_is_pad is not None:
            valid = (~image_is_pad[:, 1:]).to(squared_error)
            squared_error = (squared_error * valid).sum(1) / valid.sum(1).clamp_min(1)
        else:
            squared_error = squared_error.mean(1)
        mse_weight = float(self.asymflow.get("mse_loss_weight", 10.0))
        mse_loss = 0.5 * mse_weight * (squared_error / sigma_clamped.square()).mean()

        lpips_loss = pred_x0.new_zeros(())
        if lpips_enabled:
            lpips_model = self._get_asymflow_lpips()
            pred_rgb = self.color.decode(pred_x0)
            target_rgb = self.color.decode(full_x0.float())
            batch, _, frames, height, width = pred_rgb.shape
            pred_images = pred_rgb.permute(0, 2, 1, 3, 4).reshape(
                batch * frames, 3, height, width
            )
            target_images = target_rgb.permute(0, 2, 1, 3, 4).reshape(
                batch * frames, 3, height, width
            )
            spatial_loss = lpips_model(
                pred_images.to(self.torch_dtype),
                target_images.to(self.torch_dtype),
            )
            gate = build_vr_lpips_gate(
                coefficient.detach(), frames=frames, height=height, width=width
            )
            gate = gate.permute(0, 2, 1, 3, 4).reshape(batch * frames, 1, height, width)
            gate = F.interpolate(gate, size=spatial_loss.shape[-2:], mode="area")
            time_weight = (signal_ratio.reshape(batch) / sigma_clamped.square()).repeat_interleave(frames)
            weight = gate * time_weight[:, None, None, None]
            if image_is_pad is not None:
                valid = (~image_is_pad[:, 1:]).reshape(batch * frames, 1, 1, 1).to(weight)
                weight = weight * valid
            # Match AsymFlow's weighted-loss reduction: apply the VR/time gate
            # elementwise, then take the ordinary mean.  Dividing by
            # ``weight.sum()`` would cancel the absolute timestep/VR weighting.
            lpips_loss = (spatial_loss * weight).mean()
            lpips_loss = lpips_loss * float(self.asymflow.get("lpips_loss_weight", 1.0))

        total = mse_loss + lpips_loss
        return total, {
            "loss_video_mse": float(mse_loss.detach()),
            "loss_video_lpips": float(lpips_loss.detach()),
            "vr_coef": float(coefficient.detach().mean()),
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_group: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the mixed video/action self-attention visibility matrix.

        Video-to-video visibility is delegated to ``WanVideoDiT`` and is
        defined over temporal *groups*: one clean first-frame group and one
        group per four-frame future pixel tube.  Standard FastWAM action
        tokens can attend to action tokens and only the clean first-frame
        video group; ``FastWAMJoint`` overrides this last rule.
        """
        total_seq_len = video_seq_len + action_seq_len
        attention_mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        attention_mask[:video_seq_len, :video_seq_len] = (
            self.video_expert.build_video_to_video_mask(
                video_seq_len, video_tokens_per_group, device
            )
        )
        attention_mask[video_seq_len:, video_seq_len:] = True
        # Standard FastWAM actions read only the 98 clean first-frame tokens.
        attention_mask[
            video_seq_len:, : min(video_tokens_per_group, video_seq_len)
        ] = True
        return attention_mask

    @staticmethod
    def _pixel_video_loss(pred, target, image_is_pad):
        """Return per-sample MSE over future RGB/Oklab pixel frames.

        ``pred`` and ``target`` are reconstructed full flow velocities with
        shape ``[B,3,8,H,W]``.  Unlike the latent implementation, each future
        sampled frame has one loss entry, so ``image_is_pad[:, 1:]`` can be
        applied directly without VAE temporal-downsample mask conversion.
        """
        per_frame = F.mse_loss(
            pred.float(), target.float(), reduction="none"
        ).mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return per_frame.mean(1)
        valid = (~image_is_pad[:, 1:]).to(per_frame)
        return (per_frame * valid).sum(1) / valid.sum(1).clamp_min(1)

    def _sample_video_timestep(
        self, batch_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        timestep_sampling = str(
            self.asymflow.get("timestep_sampling", "logit_normal")
        )
        if self.asymflow_enabled and timestep_sampling == "logit_normal":
            sigma = sample_logit_normal_sigma(
                batch_size,
                device=self.device,
                dtype=dtype,
                shift=float(self.asymflow.get("timestep_shift", 17.0)),
            )
            return sigma * float(self.train_video_scheduler.num_train_timesteps)
        if not self.asymflow_enabled or timestep_sampling == "scheduler_uniform":
            return self.train_video_scheduler.sample_training_t(
                batch_size, self.device, dtype
            )
        raise ValueError(
            "Unsupported asymflow.timestep_sampling="
            f"{timestep_sampling!r}; expected 'logit_normal' or "
            "'scheduler_uniform'"
        )

    def _video_training_loss(
        self,
        *,
        inputs: dict[str, Any],
        noisy_video: torch.Tensor,
        noise_video: torch.Tensor,
        timestep_video: torch.Tensor,
        target_video: torch.Tensor,
        pred_video: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Apply the shared FastWAM, Joint, or IDM pixel-video objective."""
        if self.asymflow_enabled and self.asymflow_vr_enabled:
            return self._asymflow_video_loss(
                full_x0=inputs["future_pixels"],
                low_x0=inputs.get("low_rank_future"),
                noisy_video=noisy_video,
                noise=noise_video,
                timestep=timestep_video,
                pred_velocity=pred_video,
                first_frame=inputs["first_frame"],
                context=inputs["context"],
                context_mask=inputs["context_mask"],
                image_is_pad=inputs["image_is_pad"],
            )
        if self.asymflow_enabled:
            # Clean no-VR AsymFlow baseline: keep the asymmetric head and the
            # matched timestep sampler, but use ordinary, unscaled velocity MSE.
            loss_video = self._pixel_video_loss(
                pred_video, target_video, inputs["image_is_pad"]
            ).mean()
            return loss_video, {
                "loss_video_mse": float(loss_video.detach()),
                "loss_video_lpips": 0.0,
                "vr_coef": 0.0,
            }

        loss_video = self._pixel_video_loss(
            pred_video, target_video, inputs["image_is_pad"]
        )
        loss_video = (
            loss_video
            * self.train_video_scheduler.training_weight(timestep_video).to(
                loss_video
            )
        ).mean()
        return loss_video, {}

    def training_loss(self, sample, tiled: bool = False):
        """Compute weighted joint pixel-video and action flow-matching loss.

        The video scheduler noises only ``future_pixels``.  ``first_frame``
        stays clean and enters the video expert as a condition.  The video
        head predicts an asymmetric velocity internally, but ``post_dit``
        reconstructs the ordinary full velocity ``epsilon - x0`` before this
        method compares it with the scheduler target.  Action flow matching,
        padding masks, scheduler timestep weights, and the two lambda weights
        retain FastWAM's original objective structure.
        """
        inputs = self.build_inputs(sample, tiled)
        batch_size = inputs["future_pixels"].shape[0]

        timestep_video = self._sample_video_timestep(
            batch_size, inputs["future_pixels"].dtype
        )
        noise_video = torch.randn_like(inputs["future_pixels"])
        noisy_video = self.train_video_scheduler.add_noise(
            inputs["future_pixels"], noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            inputs["future_pixels"], noise_video, timestep_video
        )

        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size, self.device, inputs["action"].dtype
        )
        noise_action = torch.randn_like(inputs["action"])
        noisy_action = self.train_action_scheduler.add_noise(
            inputs["action"], noise_action, timestep_action
        )
        target_action = self.train_action_scheduler.training_target(
            inputs["action"], noise_action, timestep_action
        )

        video_pre = self.video_expert.pre_dit(
            first_frame=inputs["first_frame"],
            future_x=noisy_video,
            timestep=timestep_video,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=inputs["action"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
        )
        attention_mask = self._build_mot_attention_mask(
            video_pre["tokens"].shape[1],
            action_pre["tokens"].shape[1],
            video_pre["meta"]["tokens_per_group"],
            video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        loss_video, video_logs = self._video_training_loss(
            inputs=inputs,
            noisy_video=noisy_video,
            noise_video=noise_video,
            timestep_video=timestep_video,
            target_video=target_video,
            pred_video=pred_video,
        )
        loss_action = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(2)
        if inputs["action_is_pad"] is not None:
            valid = (~inputs["action_is_pad"]).to(loss_action)
            loss_action = (loss_action * valid).sum(1) / valid.sum(1).clamp_min(1)
        else:
            loss_action = loss_action.mean(1)
        loss_action = (
            loss_action
            * self.train_action_scheduler.training_weight(timestep_action).to(
                loss_action
            )
        ).mean()
        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
        )
        logs = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach()),
        }
        logs.update(video_logs)
        return loss_total, logs

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        first_frame: torch.Tensor,
        future_x: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the unchanged MoT body for one pixel/video-action flow step.

        This restores the original FastWAM inference boundary.  Only the
        video payload differs: ``first_frame`` is a clean Oklab frame and
        ``future_x`` is an 8-frame pixel state, rather than Wan VAE latents.
        """
        video_pre = self.video_expert.pre_dit(
            first_frame=first_frame,
            future_x=future_x,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self._build_mot_attention_mask(
            video_pre["tokens"].shape[1],
            action_pre["tokens"].shape[1],
            video_pre["meta"]["tokens_per_group"],
            video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        return (
            self.video_expert.post_dit(tokens_out["video"], video_pre),
            self.action_expert.post_dit(tokens_out["action"], action_pre),
        )

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Uncached action prediction; kept for parity with FastWAM."""
        video_pre = self.video_expert.pre_dit(
            first_frame=first_frame,
            timestep=torch.zeros_like(timestep_action, dtype=first_frame.dtype),
            context=context,
            context_mask=context_mask,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self._build_mot_attention_mask(
            video_pre["tokens"].shape[1],
            action_pre["tokens"].shape[1],
            video_pre["meta"]["tokens_per_group"],
            video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        return self.action_expert.post_dit(tokens_out["action"], action_pre)

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    def _pixel_context_for_infer(
        self, prompt, context, context_mask, proprio
    ):
        """Resolve text/cached context and append the optional proprio token.

        Inference accepts either a prompt or cached text embeddings, never
        both.  The returned context is ``[B,L,3072]`` before the video/text
        projection; proprio is represented as one additional context token,
        not added into video pixels or action coordinates.
        """
        if prompt is not None:
            if context is not None or context_mask is not None:
                raise ValueError(
                    "prompt and cached context are mutually exclusive"
                )
            context, context_mask = self.encode_prompt(prompt)
        if context is None or context_mask is None:
            raise ValueError(
                "pixel inference needs prompt or cached context/context_mask"
            )
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        context = context.to(self.device, self.torch_dtype)
        context_mask = context_mask.to(self.device, torch.bool)
        if proprio is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            context, context_mask = self._append_proprio_to_context(
                context,
                context_mask,
                proprio.to(self.device, self.torch_dtype),
            )
        return context, context_mask

    @torch.no_grad()
    def _clamped_video_velocity(
        self,
        future_x: torch.Tensor,
        velocity: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Apply AsymFlow's decode-clamp-encode denoised-state callback."""
        if not bool(self.asymflow.get("clamp_denoised", True)):
            return velocity
        sigma = timestep.float() / float(self.infer_video_scheduler.num_train_timesteps)
        sigma_view = sigma.view(-1, 1, 1, 1, 1)
        denoised = future_x.float() - sigma_view * velocity.float()
        rgb = self.color.decode(denoised).clamp(-1.0, 1.0)
        clamped_denoised = self.color.encode(rgb)
        return ((future_x.float() - clamped_denoised) / sigma_view.clamp_min(1e-4)).to(
            dtype=velocity.dtype
        )

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = False,
        return_video: bool = True,
    ) -> dict[str, Any]:
        """Jointly denoise future pixel video and actions from one clean RGB frame.

        ``input_image`` is converted to Oklab and remains fixed.  The method
        initializes an eight-frame future pixel state and an action state,
        then advances their independent flow schedulers together.  At every
        step ``_predict_joint_noise`` runs the shared MoT body on all video
        and action tokens.  ``return_video`` controls only RGB decoding of
        the final Oklab state; it does not change the denoising computation.
        """
        del negative_prompt, text_cfg_scale, tiled, test_action_with_infer_action
        if (
            num_video_frames <= 1
            or (num_video_frames - 1) % self.video_expert.future_tube_size
        ):
            raise ValueError(
                "pixel FastWAM requires 1+4n RGB frames: one clean first "
                "frame followed by complete four-frame tubes"
            )
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if (
            input_image.shape[:2] != (1, 3)
            or input_image.shape[-2] % 32
            or input_image.shape[-1] % 32
        ):
            raise ValueError(
                "input_image must be [1,3,H,W] with H/W divisible by 32"
            )
        context, context_mask = self._pixel_context_for_infer(
            prompt, context, context_mask, proprio
        )
        first = self.color.encode(
            input_image.to(self.device, torch.float32).unsqueeze(2)
        )[:, :, 0]
        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        future = torch.randn(
            (1, 3, num_video_frames - 1, *input_image.shape[-2:]),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(self.device)
        action_state = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(self.device, self.torch_dtype)
        timesteps_video, deltas_video = (
            self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps, self.device, future.dtype, sigma_shift
            )
        )
        timesteps_action, deltas_action = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps,
                self.device,
                action_state.dtype,
                sigma_shift,
            )
        )
        for tv, dv, ta, da in zip(
            timesteps_video,
            deltas_video,
            timesteps_action,
            deltas_action,
        ):
            timestep_video = tv[None].to(self.device, future.dtype)
            timestep_action = ta[None].to(self.device, action_state.dtype)
            video_action = (
                action
                if action is not None
                else (
                    action_state
                    if self.video_expert.action_conditioned
                    else None
                )
            )
            pred_video, pred_action = self._predict_joint_noise(
                first_frame=first,
                future_x=future,
                latents_action=action_state,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                gt_action=video_action,
            )
            pred_video = self._clamped_video_velocity(
                future, pred_video, timestep_video
            )
            future = self.infer_video_scheduler.step(
                pred_video,
                dv,
                future,
            )
            action_state = self.infer_action_scheduler.step(
                pred_action,
                da,
                action_state,
            )
        result = {"action": action_state[0].float().cpu()}
        if return_video:
            rgb = self.color.decode(
                torch.cat([first.unsqueeze(2), future], dim=2).float()
            )[0].float().clamp(-1, 1)
            result["video"] = [
                Image.fromarray(
                    ((rgb[:, t].permute(1, 2, 0) + 1) * 127.5)
                    .byte()
                    .cpu()
                    .numpy()
                )
                for t in range(num_video_frames)
            ]
        return result

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        """Denoise actions conditioned only on the clean first pixel frame.

        This is the standard FastWAM action-only path.  It precomputes the
        first-frame video K/V cache once, then each action timestep executes
        only ``forward_action_with_video_cache``.  ``num_video_frames`` is
        accepted for the shared evaluation interface but is intentionally not
        used: no future-video state is generated on this path.
        """
        del num_video_frames, negative_prompt, text_cfg_scale, tiled
        self.eval()
        if self.video_expert.video_attention_mask_mode != "first_frame_causal":
            raise ValueError(
                "infer_action requires video_attention_mask_mode='first_frame_causal'"
            )
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        context, context_mask = self._pixel_context_for_infer(
            prompt, context, context_mask, proprio
        )
        first = self.color.encode(
            input_image.to(self.device, torch.float32).unsqueeze(2)
        )[:, :, 0]
        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(self.device, self.torch_dtype)
        video_pre = self.video_expert.pre_dit(
            first_frame=first,
            timestep=torch.zeros(
                first.shape[0], device=self.device, dtype=first.dtype
            ),
            context=context,
            context_mask=context_mask,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len,
            latents_action.shape[1],
            video_pre["meta"]["tokens_per_group"],
            video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        timesteps, deltas = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps, self.device, latents_action.dtype, sigma_shift
            )
        )
        for timestep, delta in zip(timesteps, deltas):
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep[None].to(self.device, latents_action.dtype),
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action,
                delta,
                latents_action,
            )
        return {"action": latents_action[0].float().cpu()}

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        """Compatibility wrapper around :meth:`infer_joint`.

        Pixel FastWAM requires ``action_horizon`` because this wrapper always
        invokes joint video/action generation; action CFG remains unsupported.
        """
        if action_horizon is None:
            raise ValueError("Pixel FastWAM infer requires action_horizon")
        if action_cfg_scale != 1.0:
            raise ValueError("Pixel FastWAM does not implement action CFG")
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
