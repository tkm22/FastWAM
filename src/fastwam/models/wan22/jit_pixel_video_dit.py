from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from einops import rearrange

from .jit_pixel_packing import unpatchify_future_tubes
from .wan_video_dit import Head, WanVideoDiT, sinusoidal_embedding_1d


class JiTPixelWanVideoDiT(WanVideoDiT):
    """Wan Transformer body with trainable JiT-style RGB pixel boundaries."""

    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = False,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        action_conditioned: bool = False,
        action_dim: int = 7,
        action_group_causal_mask_mode: str = "causal",
        video_attention_mask_mode: str = "bidirectional",
        use_gradient_checkpointing: bool = False,
        pixel_patch_size: int = 16,
        future_tube_size: int = 4,
        bottleneck_dim: int = 192,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            in_dim=in_dim,
            ffn_dim=ffn_dim,
            out_dim=out_dim,
            text_dim=text_dim,
            freq_dim=freq_dim,
            eps=eps,
            patch_size=patch_size,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            num_layers=num_layers,
            has_image_input=has_image_input,
            has_image_pos_emb=has_image_pos_emb,
            has_ref_conv=has_ref_conv,
            add_control_adapter=add_control_adapter,
            in_dim_control_adapter=in_dim_control_adapter,
            seperated_timestep=seperated_timestep,
            require_vae_embedding=require_vae_embedding,
            require_clip_embedding=require_clip_embedding,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            action_conditioned=action_conditioned,
            action_dim=action_dim,
            action_group_causal_mask_mode=action_group_causal_mask_mode,
            video_attention_mask_mode=video_attention_mask_mode,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        source_patch_dim = self.in_dim * math.prod(self.patch_size)
        source_output_dim = int(out_dim) * math.prod(self.patch_size)
        if source_patch_dim != 192 or source_output_dim != 192:
            raise ValueError(
                "JiT pixel Wan requires the Wan2.2 192-D latent token boundary; "
                f"got input={source_patch_dim}, output={source_output_dim}"
            )
        if pixel_patch_size not in {8, 16, 32}:
            raise ValueError(
                f"`pixel_patch_size` must be one of [8, 16, 32], got {pixel_patch_size}"
            )
        if future_tube_size <= 0 or bottleneck_dim <= 0:
            raise ValueError(
                "`future_tube_size` and `bottleneck_dim` must be positive, got "
                f"{future_tube_size}, {bottleneck_dim}"
            )

        self.pixel_patch_size = int(pixel_patch_size)
        self.future_tube_size = int(future_tube_size)
        self.bottleneck_dim = int(bottleneck_dim)
        self.source_patch_size = tuple(self.patch_size)
        self.source_patch_dim = int(source_patch_dim)

        del self.patch_embedding
        self.first_patch_down = nn.Conv2d(
            3,
            self.bottleneck_dim,
            kernel_size=self.pixel_patch_size,
            stride=self.pixel_patch_size,
            bias=False,
        )
        self.future_patch_down = nn.Conv3d(
            3,
            self.bottleneck_dim,
            kernel_size=(self.future_tube_size, self.pixel_patch_size, self.pixel_patch_size),
            stride=(self.future_tube_size, self.pixel_patch_size, self.pixel_patch_size),
            bias=False,
        )
        self.wan_patch_projection = nn.Linear(
            self.bottleneck_dim,
            self.hidden_dim,
            bias=True,
        )
        self.head = Head(
            self.hidden_dim,
            3,
            (self.future_tube_size, self.pixel_patch_size, self.pixel_patch_size),
            eps,
        )
        self._initialize_pixel_boundaries()

    def _initialize_pixel_boundaries(self) -> None:
        nn.init.xavier_uniform_(self.first_patch_down.weight.flatten(1))
        nn.init.xavier_uniform_(self.future_patch_down.weight.flatten(1))
        nn.init.xavier_uniform_(self.wan_patch_projection.weight)
        nn.init.zeros_(self.wan_patch_projection.bias)
        nn.init.zeros_(self.head.head.weight)
        if self.head.head.bias is not None:
            nn.init.zeros_(self.head.head.bias)
        nn.init.zeros_(self.head.modulation)

    def _validate_pixel_inputs(
        self,
        first_frame: torch.Tensor,
        future_x: Optional[torch.Tensor],
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        if first_frame.ndim != 4 or first_frame.shape[1] != 3:
            raise ValueError(f"`first_frame` must be [B,3,H,W], got {tuple(first_frame.shape)}")
        batch_size, _, height, width = first_frame.shape
        if height % self.pixel_patch_size or width % self.pixel_patch_size:
            raise ValueError(
                f"Pixel H/W must be divisible by patch={self.pixel_patch_size}, got {height}x{width}"
            )
        if future_x is not None:
            if (
                future_x.ndim != 5
                or future_x.shape[:2] != (batch_size, 3)
                or future_x.shape[3:] != (height, width)
                or future_x.shape[2] <= 0
                or future_x.shape[2] % self.future_tube_size
            ):
                raise ValueError(
                    "`future_x` must be [B,3,T,H,W] with positive T divisible by "
                    f"tube_size={self.future_tube_size}, got {tuple(future_x.shape)}"
                )
        if context.ndim != 3 or context.shape[0] != batch_size:
            raise ValueError(
                f"`context` must be [B,L,D] matching batch={batch_size}, got {tuple(context.shape)}"
            )
        if timestep.ndim != 1 or timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` must be [1] or [B={batch_size}], got {tuple(timestep.shape)}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("Training timestep length must match batch size")
            timestep = timestep.expand(batch_size)
        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]),
                dtype=torch.bool,
                device=context.device,
            )
        elif context_mask.shape != context.shape[:2]:
            raise ValueError(
                f"`context_mask` must be {tuple(context.shape[:2])}, got {tuple(context_mask.shape)}"
            )
        return first_frame, future_x, timestep, context_mask

    def pre_dit(
        self,
        *,
        first_frame: torch.Tensor,
        future_x: Optional[torch.Tensor],
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if self.action_conditioned and action is not None:
            raise NotImplementedError("JiT pixel FastWAM-Joint does not use action-conditioned video cross-attention")
        first_frame, future_x, timestep, context_mask = self._validate_pixel_inputs(
            first_frame, future_x, timestep, context, context_mask
        )
        batch_size, _, height, width = first_frame.shape
        grid_h = height // self.pixel_patch_size
        grid_w = width // self.pixel_patch_size
        tokens_per_group = grid_h * grid_w

        first_low = rearrange(
            self.first_patch_down(first_frame.to(self.first_patch_down.weight.dtype)),
            "b c h w -> b (h w) c",
        )
        first_tokens = self.wan_patch_projection(first_low)
        if future_x is None:
            future_tokens = None
            num_groups = 1
            all_tokens = first_tokens
        else:
            future_low = rearrange(
                self.future_patch_down(future_x.to(self.future_patch_down.weight.dtype)),
                "b c f h w -> b (f h w) c",
            )
            future_tokens = self.wan_patch_projection(future_low)
            num_groups = 1 + future_x.shape[2] // self.future_tube_size
            all_tokens = torch.cat([first_tokens, future_tokens], dim=1)

        token_timesteps = torch.zeros(
            (batch_size, all_tokens.shape[1]),
            dtype=timestep.dtype,
            device=timestep.device,
        )
        if future_tokens is not None:
            token_timesteps[:, tokens_per_group:] = timestep[:, None]
        time_input = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        t = self.time_embedding(time_input).reshape(batch_size, -1, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))

        context = self.text_embedding(context)
        context_mask = context_mask[:, None, :].expand(-1, all_tokens.shape[1], -1)
        freqs = torch.cat(
            [
                self.freqs[0][:num_groups]
                .view(num_groups, 1, 1, -1)
                .expand(num_groups, grid_h, grid_w, -1),
                self.freqs[1][:grid_h]
                .view(1, grid_h, 1, -1)
                .expand(num_groups, grid_h, grid_w, -1),
                self.freqs[2][:grid_w]
                .view(1, 1, grid_w, -1)
                .expand(num_groups, grid_h, grid_w, -1),
            ],
            dim=-1,
        ).reshape(num_groups * grid_h * grid_w, 1, -1).to(all_tokens.device)

        return {
            "tokens": all_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": context_mask,
            "meta": {
                "grid_size": (num_groups, grid_h, grid_w),
                "tokens_per_frame": tokens_per_group,
                "batch_size": batch_size,
                "height": height,
                "width": width,
                "future_frames": 0 if future_x is None else future_x.shape[2],
                "has_future": future_x is not None,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        if not pre_state["meta"]["has_future"]:
            raise ValueError("JiT pixel video head requires future pixel tokens")
        spatial = int(pre_state["meta"]["tokens_per_frame"])
        clean_patches = self.head(x_tokens[:, spatial:], pre_state["t"][:, spatial:])
        return unpatchify_future_tubes(
            clean_patches,
            frames=int(pre_state["meta"]["future_frames"]),
            height=int(pre_state["meta"]["height"]),
            width=int(pre_state["meta"]["width"]),
            patch_size=self.pixel_patch_size,
            tube_size=self.future_tube_size,
        )

    def forward(
        self,
        *,
        first_frame: torch.Tensor,
        future_x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            first_frame=first_frame,
            future_x=future_x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
        )
        tokens = pre_state["tokens"]
        self_mask = self.build_video_to_video_mask(
            video_seq_len=tokens.shape[1],
            video_tokens_per_frame=int(pre_state["meta"]["tokens_per_frame"]),
            device=tokens.device,
        )
        for block in self.blocks:
            tokens = block(
                tokens,
                pre_state["context"],
                pre_state["t_mod"],
                pre_state["freqs"],
                context_mask=pre_state["context_mask"],
                self_attn_mask=self_mask,
            )
        return self.post_dit(tokens, pre_state)
