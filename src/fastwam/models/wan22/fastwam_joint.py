from typing import Any, Optional

import torch

from fastwam.utils.logging_config import get_logger

from .fastwam import FastWAM

logger = get_logger(__name__)


class FastWAMJoint(FastWAM):
    """Joint pixel model in which action tokens attend to all video tokens."""

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        video_dit_config = kwargs.get("video_dit_config", None)
        if not isinstance(video_dit_config, dict):
            raise ValueError(
                "`video_dit_config` must be provided as dict for FastWAMJoint."
            )
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError(
                "FastWAMJoint requires `video_dit_config['action_conditioned']=false`."
            )
        return super().from_wan22_pretrained(**kwargs)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_group: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Use FastWAM's video mask but expose all video tokens to actions.

        Unlike standard FastWAM, an action query may attend to the clean
        first-frame group and all noised/generated future pixel-tube groups.
        Video-to-video and action-to-action visibility are unchanged.
        """
        total = video_seq_len + action_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = (
            self.video_expert.build_video_to_video_mask(
                video_seq_len, video_tokens_per_group, device
            )
        )
        mask[video_seq_len:, video_seq_len:] = True
        mask[video_seq_len:, :video_seq_len] = True
        return mask

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
        test_action_with_infer_action: bool = True,
        return_video: bool = True,
    ) -> dict[str, Any]:
        """Run the base pixel joint denoising loop with Joint visibility.

        This override only changes the MoT attention mask.  It disables the
        legacy comparison against ``infer_action`` because a Joint action is
        defined to read the evolving future video state.  ``return_video``
        controls output decoding only, not future-video denoising.
        """
        if test_action_with_infer_action:
            logger.warning(
                "`FastWAMJoint.infer_joint` ignores "
                "`test_action_with_infer_action=True` and always runs with "
                "`test_action_with_infer_action=False`."
            )
        return super().infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
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
            test_action_with_infer_action=False,
            return_video=return_video,
        )

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
        """Generate the Joint video/action trajectory but return only actions.

        Calling base ``infer_action`` would condition actions on the first
        frame only.  This method instead invokes ``infer_joint`` with
        ``return_video=False`` so future video tokens still participate in
        every action denoising step.
        """
        output = self.infer_joint(
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
            return_video=False,
        )
        return {"action": output["action"]}
