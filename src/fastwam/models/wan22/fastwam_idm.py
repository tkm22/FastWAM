from typing import Any, Optional

import torch
import torch.nn.functional as F

from PIL import Image

from .fastwam_joint import FastWAMJoint
from fastwam.utils.logging_config import get_logger


logger = get_logger(__name__)


class FastWAMIDM(FastWAMJoint):
    """Pixel IDM with a separate teacher-forcing video stream."""

    # Hardcoded probability: during training, cond-video is noised with this chance.
    video_cond_noise_prob = 0.5

    @torch.no_grad()
    def _build_teacher_forcing_attention_mask(
        self,
        noisy_video_seq_len: int,
        cond_video_seq_len: int,
        action_seq_len: int,
        tokens_per_group: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Mask IDM's noisy and teacher-forcing pixel-video streams.

        The concatenated video sequence is ``[noisy_video, cond_video]``.
        Each stream has normal internal video visibility, action tokens attend
        to each other and only the condition stream, and they cannot read the
        noisy stream used for the video-denoising target.
        """
        noisy_end = noisy_video_seq_len
        cond_end = noisy_end + cond_video_seq_len
        mask = torch.zeros(
            (cond_end + action_seq_len, cond_end + action_seq_len),
            dtype=torch.bool,
            device=device,
        )
        mask[:noisy_end, :noisy_end] = (
            self.video_expert.build_video_to_video_mask(
                noisy_video_seq_len, tokens_per_group, device
            )
        )
        mask[noisy_end:cond_end, noisy_end:cond_end] = (
            self.video_expert.build_video_to_video_mask(
                cond_video_seq_len, tokens_per_group, device
            )
        )
        mask[cond_end:, cond_end:] = True
        mask[cond_end:, noisy_end:cond_end] = True
        return mask

    def training_loss(self, sample, tiled: bool = False):
        """Compute IDM pixel-video/action loss with teacher-forcing video tokens.

        The method builds a normal noisy future-video stream for video loss
        plus a second future condition stream.  Each sample's condition stream
        is independently noised with probability ``video_cond_noise_prob``;
        action queries can see that stream but not the target-noisy stream.
        Only the target-noisy video half contributes to reconstructed pixel
        velocity loss, while the action half uses the original action loss.
        """
        inputs = self.build_inputs(sample, tiled)
        batch_size = inputs["future_pixels"].shape[0]

        noise_video = torch.randn_like(inputs["future_pixels"])
        timestep_video = self._sample_video_timestep(
            batch_size, inputs["future_pixels"].dtype
        )
        noisy_video = self.train_video_scheduler.add_noise(
            inputs["future_pixels"], noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            inputs["future_pixels"], noise_video, timestep_video
        )

        noise_action = torch.randn_like(inputs["action"])
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size, self.device, inputs["action"].dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(
            inputs["action"], noise_action, timestep_action
        )
        target_action = self.train_action_scheduler.training_target(
            inputs["action"], noise_action, timestep_action
        )

        # Half of the teacher-forcing streams are independently noised.
        cond_noise_mask = (
            torch.rand(batch_size, device=self.device)
            < self.video_cond_noise_prob
        )
        timestep_video_cond = torch.zeros_like(timestep_video)
        future_cond = inputs["future_pixels"]
        if bool(cond_noise_mask.any()):
            timestep_video_cond_sampled = self.train_video_scheduler.sample_training_t(
                batch_size, self.device, inputs["future_pixels"].dtype
            )
            timestep_video_cond = torch.where(
                cond_noise_mask, timestep_video_cond_sampled, timestep_video_cond
            )
            future_cond_noisy = self.train_video_scheduler.add_noise(
                inputs["future_pixels"],
                torch.randn_like(inputs["future_pixels"]),
                timestep_video_cond_sampled,
            )
            future_cond = torch.where(
                cond_noise_mask[:, None, None, None, None],
                future_cond_noisy,
                future_cond,
            )

        video_pre_noisy = self.video_expert.pre_dit(
            first_frame=inputs["first_frame"],
            future_x=noisy_video,
            timestep=timestep_video,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=None,
        )
        video_pre_cond = self.video_expert.pre_dit(
            first_frame=inputs["first_frame"],
            future_x=future_cond,
            timestep=timestep_video_cond,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=None,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
        )
        noisy_video_seq_len = int(video_pre_noisy["tokens"].shape[1])
        cond_video_seq_len = int(video_pre_cond["tokens"].shape[1])
        tokens_per_group = int(video_pre_noisy["meta"]["tokens_per_group"])
        if tokens_per_group != int(video_pre_cond["meta"]["tokens_per_group"]):
            raise ValueError(
                "IDM video streams must have equal spatial token counts"
            )
        attention_mask = self._build_teacher_forcing_attention_mask(
            noisy_video_seq_len,
            cond_video_seq_len,
            action_pre["tokens"].shape[1],
            tokens_per_group,
            video_pre_noisy["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": torch.cat(
                    [video_pre_noisy["tokens"], video_pre_cond["tokens"]], dim=1
                ),
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": torch.cat(
                    [video_pre_noisy["freqs"], video_pre_cond["freqs"]], dim=0
                ),
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre_noisy["context"],
                    "mask": torch.cat(
                        [
                            video_pre_noisy["context_mask"],
                            video_pre_cond["context_mask"],
                        ],
                        dim=1,
                    ),
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": torch.cat(
                    [video_pre_noisy["t_mod"], video_pre_cond["t_mod"]], dim=1
                ),
                "action": action_pre["t_mod"],
            },
        )
        pred_video = self.video_expert.post_dit(
            tokens_out["video"][:, :noisy_video_seq_len], video_pre_noisy
        )
        pred_action = self.action_expert.post_dit(
            tokens_out["action"], action_pre
        )

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
        ).mean(dim=2)
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
        """Generate future pixel video first, then condition action denoising on it.

        Stage 1 runs the video expert alone from the clean first frame and
        future pixel noise.  Stage 2 tokenizes the resulting video with
        timestep zero, prefills its MoT K/V cache, and denoises actions against
        that fixed generated-video condition.  User-supplied ``action`` is
        intentionally ignored because IDM has no action-conditioned video
        generation stage.
        """
        if action is not None:
            logger.warning(
                "FastWAMIDM.infer_joint ignores `action`; it first generates "
                "the video stream."
            )
        del action, negative_prompt, text_cfg_scale, tiled
        del test_action_with_infer_action
        self.eval()
        if (
            num_video_frames <= 1
            or (num_video_frames - 1) % self.video_expert.future_tube_size
        ):
            raise ValueError(
                "pixel IDM requires one clean frame followed by complete "
                "four-frame tubes"
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
        timesteps_video, deltas_video = (
            self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps, self.device, future.dtype, sigma_shift
            )
        )
        for timestep, delta in zip(timesteps_video, deltas_video):
            velocity = self.video_expert(
                x=torch.cat([first.unsqueeze(2), future], dim=2),
                timestep=timestep[None].to(self.device, future.dtype),
                context=context,
                context_mask=context_mask,
                action=None,
            )
            velocity = self._clamped_video_velocity(
                future,
                velocity,
                timestep[None].to(self.device, future.dtype),
            )
            future = self.infer_video_scheduler.step(
                velocity, delta, future
            )

        action_state = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(self.device, self.torch_dtype)
        video_pre = self.video_expert.pre_dit(
            first_frame=first,
            future_x=future,
            timestep=torch.zeros(
                1, device=self.device, dtype=future.dtype
            ),
            context=context,
            context_mask=context_mask,
            action=None,
        )
        mask = self._build_mot_attention_mask(
            video_pre["tokens"].shape[1],
            action_horizon,
            video_pre["meta"]["tokens_per_group"],
            self.device,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=mask[:video_seq_len, :video_seq_len],
        )
        timesteps_action, deltas_action = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps,
                self.device,
                action_state.dtype,
                sigma_shift,
            )
        )
        for timestep, delta in zip(timesteps_action, deltas_action):
            pred_action = self._predict_action_noise_with_cache(
                latents_action=action_state,
                timestep_action=timestep[None].to(
                    self.device, action_state.dtype
                ),
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=mask,
                video_seq_len=video_seq_len,
            )
            action_state = self.infer_action_scheduler.step(
                pred_action,
                delta,
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
        """Run IDM's two-stage generation and return only its action result."""
        return {
            "action": self.infer_joint(
                prompt=prompt,
                input_image=input_image,
                num_video_frames=num_video_frames,
                action_horizon=action_horizon,
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
                return_video=False,
            )["action"]
        }
