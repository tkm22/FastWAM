from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from PIL import Image

from .action_dit import ActionDiT
from .fastwam_joint import FastWAMJoint
from .helpers.jit_pixel_loader import load_jit_pixel_wan22_components
from .jit_pixel_objective import add_jit_noise, jit_velocity_from_x0, sample_jit_sigma
from .mot import MoT


class FastWAMJointJiTPixel(FastWAMJoint):
    """VAE-free RGB pixel-space FastWAM-Joint with the JiT x-prediction loss."""

    REPRESENTATION = "rgb_jit_pixel_xpred_v1"

    def __init__(
        self,
        *args,
        p_mean: float = -0.8,
        p_std: float = 0.8,
        noise_scale: float = 1.0,
        t_eps: float = 0.05,
        adapter_only_finetune: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.vae is not None:
            raise ValueError("FastWAMJointJiTPixel must be constructed without a VAE")
        if p_std < 0 or noise_scale <= 0 or t_eps <= 0:
            raise ValueError(
                "`p_std` must be non-negative and `noise_scale`/`t_eps` positive, got "
                f"{p_std}, {noise_scale}, {t_eps}"
            )
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.noise_scale = float(noise_scale)
        self.t_eps = float(t_eps)
        self.adapter_only_finetune = bool(adapter_only_finetune)

    @property
    def pixel_patch_size(self) -> int:
        return int(self.video_expert.pixel_patch_size)

    @property
    def future_tube_size(self) -> int:
        return int(self.video_expert.future_tube_size)

    @property
    def bottleneck_dim(self) -> int:
        return int(self.video_expert.bottleneck_dim)

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
        p_mean: float = -0.8,
        p_std: float = 0.8,
        noise_scale: float = 1.0,
        t_eps: float = 0.05,
        adapter_only_finetune: bool = False,
    ) -> "FastWAMJointJiTPixel":
        if not isinstance(video_dit_config, dict):
            raise ValueError("`video_dit_config` is required for JiT pixel FastWAM-Joint")
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError("JiT pixel FastWAM-Joint requires `action_conditioned=false`")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required")

        components = load_jit_pixel_wan22_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
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
            raise ValueError("ActionDiT `num_heads` must match the video expert")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match the video expert")
        if len(action_expert.blocks) != len(video_expert.blocks):
            raise ValueError("ActionDiT `num_layers` must match the video expert")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=None,
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
            p_mean=p_mean,
            p_std=p_std,
            noise_scale=noise_scale,
            t_eps=t_eps,
            adapter_only_finetune=adapter_only_finetune,
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

    def configure_training_mode(self) -> bool:
        """Freeze pretrained backbones and train only the pixel/action boundaries."""
        if not self.adapter_only_finetune:
            return False

        self.eval()
        self.requires_grad_(False)

        # Keep the MoT wrapper marked as training so trainer eval can restore this mode.
        self.dit.train()
        self.video_expert.eval()
        self.action_expert.eval()

        trainable_modules = (
            self.video_expert.first_patch_down,
            self.video_expert.future_patch_down,
            self.video_expert.wan_patch_projection,
            self.video_expert.head,
            self.action_expert.action_encoder,
            self.action_expert.head,
        )
        for module in trainable_modules:
            module.train()
            module.requires_grad_(True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.train()
            self.proprio_encoder.requires_grad_(True)
        return True

    def _validate_video_shape(self, video: torch.Tensor) -> tuple[int, int, int, int]:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Video must be [B,3,T,H,W], got {tuple(video.shape)}")
        batch_size, _, num_frames, height, width = video.shape
        if height % self.pixel_patch_size or width % self.pixel_patch_size:
            raise ValueError(
                f"Video H/W must be divisible by patch={self.pixel_patch_size}, got {height}x{width}"
            )
        if num_frames <= 1 or (num_frames - 1) % self.future_tube_size:
            raise ValueError(
                "Video must have one clean frame plus a future length divisible by "
                f"tube={self.future_tube_size}, got T={num_frames}"
            )
        return batch_size, num_frames, height, width

    def build_inputs(self, sample, tiled: bool = False):
        if tiled:
            raise ValueError("JiT pixel mode has no VAE tiling path")
        if "video" not in sample or "action" not in sample:
            raise ValueError("JiT pixel training requires `video` and `action`")
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError("JiT pixel training requires cached `context` and `context_mask`")

        video = sample["video"]
        batch_size, num_frames, _, _ = self._validate_video_shape(video)
        action = sample["action"]
        if action.ndim != 3 or action.shape[0] != batch_size:
            raise ValueError(f"Action must be [B,T,D] matching video batch, got {tuple(action.shape)}")
        if action.shape[1] % (num_frames - 1):
            raise ValueError(
                f"Action horizon {action.shape[1]} must be divisible by {num_frames - 1} future frames"
            )

        context = sample["context"]
        context_mask = sample["context_mask"]
        if context.ndim != 3 or context.shape[0] != batch_size:
            raise ValueError(f"Context must be [B,L,D], got {tuple(context.shape)}")
        if context_mask.shape != context.shape[:2]:
            raise ValueError(
                f"Context mask must be {tuple(context.shape[:2])}, got {tuple(context_mask.shape)}"
            )

        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None and action_is_pad.shape != action.shape[:2]:
            raise ValueError(
                f"Action padding mask must be {tuple(action.shape[:2])}, got {tuple(action_is_pad.shape)}"
            )
        image_is_pad = sample.get("image_is_pad")
        if image_is_pad is not None and image_is_pad.shape != (batch_size, num_frames):
            raise ValueError(
                f"Image padding mask must be {(batch_size, num_frames)}, got {tuple(image_is_pad.shape)}"
            )

        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        proprio = sample.get("proprio")
        if self.proprio_encoder is not None:
            if proprio is None or proprio.ndim != 3 or proprio.shape[2] != self.proprio_dim:
                shape = None if proprio is None else tuple(proprio.shape)
                raise ValueError(
                    f"Proprio must be [B,T,{self.proprio_dim}] when enabled, got {shape}"
                )
            context, context_mask = self._append_proprio_to_context(
                context,
                context_mask,
                proprio[:, 0].to(device=self.device, dtype=self.torch_dtype),
            )

        video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        return {
            "first_frame": video[:, :, 0],
            "future_clean": video[:, :, 1:],
            "context": context,
            "context_mask": context_mask,
            "action": action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            "action_is_pad": (
                None
                if action_is_pad is None
                else action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
            ),
            "future_is_pad": (
                None
                if image_is_pad is None
                else image_is_pad[:, 1:].to(device=self.device, dtype=torch.bool, non_blocking=True)
            ),
        }

    def _run_joint_experts(
        self,
        *,
        first_frame: torch.Tensor,
        future_x: torch.Tensor,
        timestep_video: torch.Tensor,
        action_x: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            first_frame=first_frame,
            future_x=future_x,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=action_x,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        return (
            self.video_expert.post_dit(tokens_out["video"], video_pre),
            self.action_expert.post_dit(tokens_out["action"], action_pre),
        )

    @staticmethod
    def _masked_temporal_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        per_frame = F.mse_loss(pred.float(), target.float(), reduction="none").mean(
            dim=(1, 3, 4)
        )
        if is_pad is None:
            return per_frame.mean(dim=1)
        if is_pad.shape != per_frame.shape:
            raise ValueError(
                f"Future padding mask must be {tuple(per_frame.shape)}, got {tuple(is_pad.shape)}"
            )
        valid = (~is_pad).to(device=per_frame.device, dtype=per_frame.dtype)
        return (per_frame * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        clean = inputs["future_clean"]
        batch_size = clean.shape[0]
        sigma = sample_jit_sigma(
            batch_size,
            device=self.device,
            p_mean=self.p_mean,
            p_std=self.p_std,
        )
        noise = torch.randn_like(clean) * self.noise_scale
        noisy = add_jit_noise(clean.float(), noise.float(), sigma).to(self.torch_dtype)
        timestep_video = (
            sigma * float(self.train_video_scheduler.num_train_timesteps)
        ).to(dtype=self.torch_dtype)

        action = inputs["action"]
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action, noise_action, timestep_action
        )
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )

        pred_x0, pred_action = self._run_joint_experts(
            first_frame=inputs["first_frame"],
            future_x=noisy,
            timestep_video=timestep_video,
            action_x=noisy_action,
            timestep_action=timestep_action,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
        )
        pred_velocity = jit_velocity_from_x0(
            noisy.float(), pred_x0.float(), sigma, t_eps=self.t_eps
        )
        target_velocity = jit_velocity_from_x0(
            noisy.float(), clean.float(), sigma, t_eps=self.t_eps
        )
        loss_video = self._masked_temporal_mse(
            pred_velocity, target_velocity, inputs["future_is_pad"]
        ).mean()
        pred_x0_mse = self._masked_temporal_mse(
            pred_x0, clean, inputs["future_is_pad"]
        ).mean()

        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if inputs["action_is_pad"] is None:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        else:
            valid = (~inputs["action_is_pad"]).to(action_loss_token.dtype)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid.sum(
                dim=1
            ).clamp_min(1.0)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            device=action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach()),
            "pred_x0_mse": float(pred_x0_mse.detach()),
            "sigma_mean": float(sigma.mean().detach()),
        }

    def _prepare_inference_context(
        self,
        *,
        prompt: Optional[Union[str, Sequence[str]]],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt == use_context:
            raise ValueError("Provide exactly one of prompt or context/context_mask")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be provided together")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.shape != context.shape[:2]:
                raise ValueError(
                    f"Invalid context shapes: {tuple(context.shape)}, {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided while proprio encoding is disabled")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.shape != (context.shape[0], self.proprio_dim):
                raise ValueError(
                    f"Proprio must be [B,{self.proprio_dim}], got {tuple(proprio.shape)}"
                )
            context, context_mask = self._append_proprio_to_context(
                context,
                context_mask,
                proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        return context, context_mask

    def _prepare_input_image(self, input_image: torch.Tensor) -> torch.Tensor:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must be [3,H,W] or [1,3,H,W], got {tuple(input_image.shape)}"
            )
        if input_image.shape[-2] % self.pixel_patch_size or input_image.shape[-1] % self.pixel_patch_size:
            raise ValueError(
                f"Input H/W must be divisible by patch={self.pixel_patch_size}, got {input_image.shape[-2:]}"
            )
        return input_image.to(device=self.device, dtype=self.torch_dtype)

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
        result = self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=None,
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
            test_action_with_infer_action=False,
        )
        return {"action": result["action"]}

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
    ) -> dict[str, Any]:
        del action, negative_prompt, text_cfg_scale, test_action_with_infer_action
        if tiled:
            raise ValueError("JiT pixel mode has no VAE tiling path")
        if num_video_frames <= 1 or (num_video_frames - 1) % self.future_tube_size:
            raise ValueError(
                f"`num_video_frames-1` must be divisible by tube={self.future_tube_size}"
            )
        self.eval()
        first_frame = self._prepare_input_image(input_image)
        context, context_mask = self._prepare_inference_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        future_x = (
            torch.randn(
                (1, 3, num_video_frames - 1, first_frame.shape[-2], first_frame.shape[-1]),
                generator=video_generator,
                device=rand_device,
                dtype=torch.float32,
            )
            * self.noise_scale
        ).to(device=self.device, dtype=self.torch_dtype)
        action_x = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        video_ts, video_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=future_x.dtype,
            shift_override=sigma_shift,
        )
        action_ts, action_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=action_x.dtype,
            shift_override=sigma_shift,
        )
        for video_t, video_delta, action_t, action_delta in zip(
            video_ts, video_deltas, action_ts, action_deltas
        ):
            pred_x0, pred_action = self._run_joint_experts(
                first_frame=first_frame,
                future_x=future_x,
                timestep_video=video_t.unsqueeze(0),
                action_x=action_x,
                timestep_action=action_t.unsqueeze(0),
                context=context,
                context_mask=context_mask,
            )
            sigma = video_t.float().reshape(1) / float(
                self.infer_video_scheduler.num_train_timesteps
            )
            pred_velocity = jit_velocity_from_x0(
                future_x.float(), pred_x0.float(), sigma, t_eps=self.t_eps
            ).to(future_x.dtype)
            future_x = self.infer_video_scheduler.step(pred_velocity, video_delta, future_x)
            action_x = self.infer_action_scheduler.step(pred_action, action_delta, action_x)

        video = torch.cat([first_frame.unsqueeze(2), future_x], dim=2)
        video = ((video[0].float().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = [
            Image.fromarray(video[:, index].permute(1, 2, 0).numpy())
            for index in range(video.shape[1])
        ]
        return {"video": frames, "action": action_x[0].detach().cpu().float()}

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "representation": self.REPRESENTATION,
            "pixel_patch_size": self.pixel_patch_size,
            "future_tube_size": self.future_tube_size,
            "bottleneck_dim": self.bottleneck_dim,
            "t_eps": self.t_eps,
        }

    def _validate_checkpoint_metadata(self, payload: dict[str, Any], path: Path) -> None:
        actual = payload.get("jit_pixel")
        expected = self._checkpoint_metadata()
        if not isinstance(actual, dict):
            raise ValueError(f"Checkpoint is missing JiT pixel metadata: {path}")
        mismatches = {
            key: (expected[key], actual.get(key))
            for key in expected
            if actual.get(key) != expected[key]
        }
        if mismatches:
            raise ValueError(f"JiT pixel checkpoint metadata mismatch: {mismatches}")

    def validate_training_state(self, state_dir: str) -> None:
        state_path = Path(state_dir)
        step_match = state_path.name.removeprefix("step_")
        if not step_match.isdigit():
            raise ValueError(f"Invalid JiT pixel state directory name: {state_path.name}")
        weights_path = state_path.parent.parent / "weights" / f"{state_path.name}.pt"
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Full-state resume requires its matching raw checkpoint: {weights_path}"
            )
        payload = torch.load(weights_path, map_location="cpu", mmap=True)
        self._validate_checkpoint_metadata(payload, weights_path)
        expected_step = int(step_match)
        if payload.get("step") != expected_step:
            raise ValueError(
                "JiT pixel raw/state step mismatch: "
                f"checkpoint={payload.get('step')}, state={expected_step}"
            )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "jit_pixel": self._checkpoint_metadata(),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        path = Path(path)
        payload = torch.load(path, map_location="cpu")
        self._validate_checkpoint_metadata(payload, path)
        if "mot" not in payload:
            raise ValueError(f"JiT pixel checkpoint is missing `mot`: {path}")
        self.mot.load_state_dict(payload["mot"], strict=True)
        if self.proprio_encoder is not None:
            if "proprio_encoder" not in payload:
                raise ValueError(f"Checkpoint is missing `proprio_encoder`: {path}")
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        elif "proprio_encoder" in payload:
            raise ValueError("Checkpoint has proprio weights but this model disables proprio")
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
