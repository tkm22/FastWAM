import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, Tuple, Optional
from einops import rearrange

from asymflow.velocity import asymflow_calibration, asymflow_velocity
from asymflow.video_packing import (
    patchify_future_tubes,
    unpatchify_future_tubes,
)

from .helpers.gradient import gradient_checkpoint_forward

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, ctx_mask: Optional[torch.Tensor] = None, compatibility_mode=True):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        return x
    else:
        raise NotImplementedError("Only compatibility mode is implemented for flash attention. Please set compatibility_mode=True.")



def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def create_group_causal_attn_mask(
    num_temporal_groups: int, num_query_per_group: int, num_key_per_group: int, mode: str = "causal"
) -> torch.Tensor:
    """
    Creates a group-based attention mask for scaled dot-product attention with two modes:
    'causal' and 'group_diagonal'.

    Parameters:
    - num_temporal_groups (int): The number of temporal groups (e.g., frames in a video sequence).
    - num_query_per_group (int): The number of query tokens per temporal group. (e.g., latent tokens in a frame, H x W).
    - num_key_per_group (int): The number of key tokens per temporal group. (e.g., action tokens per frame).
    - mode (str): The mode of the attention mask. Options are:
        - 'causal': Query tokens can attend to key tokens from the same or previous temporal groups.
        - 'group_diagonal': Query tokens can attend only to key tokens from the same temporal group.

    Returns:
    - attn_mask (torch.Tensor): A boolean tensor of shape (L, S), where:
        - L = num_temporal_groups * num_query_per_group (total number of query tokens)
        - S = num_temporal_groups * num_key_per_group (total number of key tokens)
      The mask indicates where attention is allowed (True) and disallowed (False).

    Example:
    Input:
        num_temporal_groups = 3
        num_query_per_group = 4
        num_key_per_group = 2
    Output:
        Causal Mask Shape: torch.Size([12, 6])
        Group Diagonal Mask Shape: torch.Size([12, 6])
        if mode='causal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True]])

        if mode='group_diagonal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True]])

    """
    assert mode in ["causal", "group_diagonal"], f"Mode {mode} must be 'causal' or 'group_diagonal'"

    # Total number of query and key tokens
    total_num_query_tokens = num_temporal_groups * num_query_per_group  # Total number of query tokens (L)
    total_num_key_tokens = num_temporal_groups * num_key_per_group  # Total number of key tokens (S)

    # Generate time indices for query and key tokens (shape: [L] and [S])
    query_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_query_per_group)  # Shape: [L]
    key_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_key_per_group)  # Shape: [S]

    # Expand dimensions to compute outer comparison
    query_time_indices = query_time_indices.unsqueeze(1)  # Shape: [L, 1]
    key_time_indices = key_time_indices.unsqueeze(0)  # Shape: [1, S]

    if mode == "causal":
        # Causal Mode: Query can attend to keys where key_time <= query_time
        attn_mask = query_time_indices >= key_time_indices  # Shape: [L, S]
    elif mode == "group_diagonal":
        # Group Diagonal Mode: Query can attend only to keys where key_time == query_time
        attn_mask = query_time_indices == key_time_indices  # Shape: [L, S]

    assert attn_mask.shape == (total_num_query_tokens, total_num_key_tokens), "Attention mask shape mismatch"
    return attn_mask


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, ctx_mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return x


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)
        
        # self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, self_attn_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6,):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)
            
        # self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self,  hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, hidden_dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, context_mask=None, self_attn_mask: Optional[torch.Tensor] = None):
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1) # (B, 1, seq_len, context_len), 1 for heads
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask))
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanVideoDiT(torch.nn.Module):
    """Wan Transformer with asymmetric pixel video I/O.

    This is the only WanVideoDiT implementation in the pixelgen branch.  The
    pretrained latent Wan checkpoint is read as a state dict by the loader; a
    second latent model is never constructed.
    """

    pixel_patch_size = 32
    future_tube_size = 4

    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
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
        action_group_causal_mask_mode = "causal",
        video_attention_mask_mode: str = "bidirectional",
        use_gradient_checkpointing: bool = False,
        inference_sigma_min: float = 1e-4,
        projection_artifact: Optional[Dict[str, Any]] = None,
    ):
        """Construct the pretrained Wan Transformer body with pixel I/O.

        ``in_dim``, ``out_dim``, and ``patch_size`` still describe the source
        Wan checkpoint and must yield its 192-D latent patch
        ``48*1*2*2``.  They are used to validate and rewrite source weights,
        not as runtime input geometry.  Runtime I/O is instead one
        ``3*32*32=3072``-D clean Oklab first-frame patch and one
        ``3*4*32*32=12288``-D future Oklab tube.  ``projection_artifact``
        supplies both lift matrices and calibration scales.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.video_attention_mask_mode = str(video_attention_mask_mode)
        self.inference_sigma_min = float(inference_sigma_min)
        if self.inference_sigma_min <= 0:
            raise ValueError(
                "`inference_sigma_min` must be positive, got "
                f"{self.inference_sigma_min}"
            )

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(
                f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}"
            )
        
        self.action_conditioned = action_conditioned
        self.action_dim = action_dim
        assert has_image_input == False
        assert require_clip_embedding == False

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, hidden_dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

        if self.action_conditioned:
            self.action_embedding = nn.Linear(action_dim, hidden_dim)
            self.action_group_causal_mask_mode = action_group_causal_mask_mode
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        if self.use_gradient_checkpointing:
            logger.info("Using gradient checkpointing for DiT blocks. This will save memory but use more computation.")

        if projection_artifact is None:
            raise ValueError(
                "Pixel WanVideoDiT requires a fitted projection artifact"
            )
        if tuple(projection_artifact["A_first"].shape) != (3072, 192):
            raise ValueError(
                "A_first must map 192-D Wan latent tokens to "
                "3072-D first-frame pixel patches"
            )
        if tuple(projection_artifact["A_future"].shape) != (12288, 192):
            raise ValueError(
                "A_future must map 192-D Wan latent tokens to "
                "12288-D four-frame pixel tubes"
            )
        if in_dim * math.prod(patch_size) != 192:
            raise ValueError(
                "The Wan checkpoint source input must be "
                f"48x1x2x2=192-D, got {in_dim}x{tuple(patch_size)}"
            )
        if out_dim * math.prod(patch_size) != 192:
            raise ValueError(
                "The Wan checkpoint source output must be "
                f"48x1x2x2=192-D, got {out_dim}x{tuple(patch_size)}"
            )

        # Source checkpoint input:
        #   Conv3d[48, 1, 2, 2] -> 192-D latent token -> hidden_dim.
        # Pixel first-frame input:
        #   Conv2d[3, 32, 32] -> 3072-D Oklab patch -> hidden_dim.
        # Loader initialization: W_first = W_in^z @ A_first.T.
        self.first_patch_embedding = nn.Conv2d(
            3, hidden_dim, kernel_size=32, stride=32
        )
        # Pixel future input:
        #   3 channels x 4 sampled frames x 32 x 32 = 12288-D Oklab tube
        #   -> hidden_dim. Loader initialization:
        #   W_future = W_in^z @ A_future.T.
        self.future_patch_embedding = nn.Linear(
            3 * 4 * 32 * 32, hidden_dim
        )
        # Source checkpoint output: hidden_dim -> 192-D latent token.
        # Pixel output: hidden_dim -> 12288-D calibrated asymmetric velocity
        # tube. Loader initialization: W_out^x = A_future @ W_out^z.
        self.head = Head(hidden_dim, 3, (4, 32, 32), eps)
        self.register_buffer("A_first", projection_artifact["A_first"])
        self.register_buffer("A_future", projection_artifact["A_future"])
        self.register_buffer(
            "scale_first",
            torch.as_tensor(
                projection_artifact["scale_first"], dtype=torch.float32
            ).reshape(()),
        )
        self.register_buffer(
            "scale_future",
            torch.as_tensor(
                projection_artifact["scale_future"], dtype=torch.float32
            ).reshape(()),
        )
        self.flow_num_train_timesteps = 1000

    def _apply(self, fn):
        """Move the model while preserving AsymFlow calibration in FP32.

        PyTorch normally casts floating-point buffers together with model
        weights.  Upstream AsymFlow explicitly performs ``A A.T`` projection
        and scale calibration outside autocast in float32, so retaining only
        BF16 copies of these fitted buffers would irreversibly lose precision.
        """
        super()._apply(fn)
        self.A_first = self.A_first.float()
        self.A_future = self.A_future.float()
        self.scale_first = self.scale_first.float()
        self.scale_future = self.scale_future.float()
        return self

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_group: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return video-token visibility using pixel temporal groups.

        A group has ``H/32 * W/32`` tokens.  Group 0 is the clean first RGB
        frame; every later group represents four sampled pixel frames.  RoPE
        order follows these token groups, so the causal mask must use the
        same grouping rather than the source Wan latent-frame count.
        """
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_group <= 0:
            raise ValueError(
                "`video_tokens_per_group` must be positive, "
                f"got {video_tokens_per_group}"
            )

        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if self.video_attention_mask_mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_group != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by "
                    "`video_tokens_per_group` in `per_frame_causal` mode, "
                    f"got {video_seq_len} and {video_tokens_per_group}"
                )
            num_temporal_groups = video_seq_len // video_tokens_per_group
            group_causal = torch.tril(
                torch.ones(
                    (num_temporal_groups, num_temporal_groups),
                    dtype=torch.bool,
                    device=device,
                )
            )
            return group_causal.repeat_interleave(
                video_tokens_per_group, dim=0
            ).repeat_interleave(
                video_tokens_per_group, dim=1
            )

        if self.video_attention_mask_mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_group, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    def _pixel_context(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        action: Optional[torch.Tensor],
        video_tokens: int,
        tokens_per_group: int,
        num_temporal_groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project text/action conditions and construct cross-attention masks.

        Text is always visible to every video token.  In action-conditioned
        video models, action embeddings are concatenated to the text K/V
        sequence; they are never added to video tokens.  The first-frame
        group cannot read action K/V, while future four-frame groups use the
        configured causal action-to-video grouping.
        """
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], device=context.device, dtype=torch.bool)
        context = self.text_embedding(context)
        text_len = context.shape[1]
        if self.action_conditioned and action is not None:
            action_len = action.shape[1]
            if action_len % (num_temporal_groups - 1):
                raise ValueError(
                    "action horizon must be divisible by future pixel-frame groups"
                )
            action_emb = self.action_embedding(action)
            action_emb = action_emb + sinusoidal_embedding_1d(
                self.hidden_dim,
                torch.arange(
                    action_len, device=action.device, dtype=action.dtype
                ),
            ).unsqueeze(0)
            context = torch.cat([context, action_emb], dim=1)
            action_mask = create_group_causal_attn_mask(
                num_temporal_groups - 1,
                tokens_per_group,
                action_len // (num_temporal_groups - 1),
                self.action_group_causal_mask_mode,
            ).to(context.device)
            final_mask = torch.zeros(
                (context.shape[0], video_tokens, context.shape[1]),
                device=context.device,
                dtype=torch.bool,
            )
            final_mask[:, :, :text_len] = context_mask.unsqueeze(1)
            final_mask[:, tokens_per_group:, text_len:] = action_mask.unsqueeze(0)
            return context, final_mask
        if (
            self.action_conditioned
            and action is None
            and num_temporal_groups != 1
        ):
            # IDM has a video-only branch. Text-only cross attention is valid.
            return context, context_mask.unsqueeze(1).expand(-1, video_tokens, -1)
        return context, context_mask.unsqueeze(1).expand(-1, video_tokens, -1)

    def pre_dit(
        self,
        x: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        *,
        first_frame: Optional[torch.Tensor] = None,
        future_x: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Prepare pixel tokens, calibrated time modulation, RoPE and context.

        ``first_frame`` is a clean standardized Oklab tensor ``[B,3,H,W]``.
        ``future_x`` is the noised pixel-flow state ``[B,3,4n,H,W]``;
        LIBERO normally uses ``n=2`` from offsets ``4,...,32``.  The first
        path applies ``first_frame / scale_first`` then a stride-32 Conv2d,
        producing one token group.  Each future path applies
        ``k(sigma) * future_x``, packs four frames into a 12288-D tube, and
        projects it with ``future_patch_embedding``.

        The first token group gets diffusion timestep 0.  Every future group
        gets the same calibrated diffusion timestep ``k(sigma)*sigma``;
        temporal position is instead distinguished by RoPE's group index.
        The returned ``freqs`` are therefore ordered as
        ``[first group, future tube 0, future tube 1, ...]``, not by original
        raw frames or source VAE latent frames.
        """
        if first_frame is None:
            if x is None or x.ndim != 5:
                raise ValueError(
                    "pixel Wan needs first_frame [B,3,H,W] and "
                    "future_x [B,3,8,H,W]"
                )
            first_frame, future_x = x[:, :, 0], x[:, :, 1:]
        if timestep is None or context is None:
            raise ValueError("timestep and context are required")
        if first_frame.ndim != 4 or first_frame.shape[1] != 3:
            raise ValueError("first_frame must be [B,3,H,W]")
        B, _, H, W = first_frame.shape
        if H % 32 or W % 32:
            raise ValueError("pixel H/W must be divisible by 32")
        spatial = (H // 32) * (W // 32)
        first_input = (first_frame / self.scale_first.float()).to(
            self.first_patch_embedding.weight.dtype
        )
        first_tokens = rearrange(
            self.first_patch_embedding(first_input),
            "b c h w -> b (h w) c",
        )
        if future_x is None:
            all_tokens = first_tokens
            num_temporal_groups = 1
            t_values = torch.zeros((B, spatial), dtype=timestep.dtype, device=timestep.device)
            future_packed = None
        else:
            if (
                future_x.ndim != 5
                or future_x.shape[:2] != (B, 3)
                or future_x.shape[3:] != (H, W)
                or future_x.shape[2] == 0
                or future_x.shape[2] % self.future_tube_size
            ):
                raise ValueError(
                    f"future_x must be [B,3,T,{H},{W}] with positive T "
                    f"divisible by {self.future_tube_size}, got "
                    f"{tuple(future_x.shape)}"
                )
            sigma = timestep / float(self.flow_num_train_timesteps)
            calibrated_sigma, k = asymflow_calibration(sigma, self.scale_future)
            future_scaled = future_x * k.to(future_x).reshape(B, 1, 1, 1, 1)
            future_packed = patchify_future_tubes(future_scaled)
            future_tokens = self.future_patch_embedding(
                future_packed.to(self.future_patch_embedding.weight.dtype)
            )
            all_tokens = torch.cat([first_tokens, future_tokens], dim=1)
            num_temporal_groups = (
                1 + future_x.shape[2] // self.future_tube_size
            )
            t_values = torch.cat(
                [
                    torch.zeros(
                        (B, spatial),
                        dtype=timestep.dtype,
                        device=timestep.device,
                    ),
                    (
                        calibrated_sigma.to(timestep)
                        * float(self.flow_num_train_timesteps)
                    )[:, None].expand(B, (num_temporal_groups - 1) * spatial),
                ],
                dim=1,
            )
        time_input = sinusoidal_embedding_1d(
            self.freq_dim, t_values.reshape(-1)
        ).to(self.time_embedding[0].weight.dtype)
        t = self.time_embedding(time_input).reshape(B, -1, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        context, pixel_mask = self._pixel_context(
            context,
            context_mask,
            action,
            all_tokens.shape[1],
            spatial,
            num_temporal_groups,
        )
        f, h, w = num_temporal_groups, H // 32, W // 32
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(all_tokens.device)
        return {
            "tokens": all_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": pixel_mask,
            "meta": {
                "grid_size": (f, h, w),
                "tokens_per_group": spatial,
                "batch_size": B,
                "height": H,
                "width": W,
                "future_x": future_x,
                "future_packed": future_packed,
                "sigma": timestep / float(self.flow_num_train_timesteps),
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        """Reconstruct full future-pixel velocity from the asymmetric head output.

        The head skips the clean first-frame token group and emits one
        12288-D calibrated value ``u_A = P eps - x0 / s`` for each future
        tube.  ``asymflow_velocity`` applies ``P=A_future A_future.T``
        implicitly and returns ordinary ``eps - x0`` tubes, which are then
        unpacked to ``[B,3,4n,H,W]`` for the scheduler and video loss.
        """
        future_x = pre_state["meta"]["future_x"]
        if future_x is None:
            raise ValueError("pixel Wan has no velocity output for first-frame-only input")
        spatial = pre_state["meta"]["tokens_per_group"]
        u_asym = self.head(x_tokens[:, spatial:], pre_state["t"][:, spatial:])
        x_packed = patchify_future_tubes(future_x)
        full_velocity = asymflow_velocity(
            u_asym,
            x_packed,
            pre_state["meta"]["sigma"],
            self.scale_future,
            self.A_future,
            sigma_min=1e-6 if self.training else self.inference_sigma_min,
        )
        return unpatchify_future_tubes(
            full_velocity,
            future_x.shape[2],
            pre_state["meta"]["height"],
            pre_state["meta"]["width"],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ):
        """Run the unchanged Wan Transformer body on pixel video states.

        ``x`` is accepted as ``[B,3,1+4n,H,W]`` for compatibility with
        video-only IDM inference.  ``pre_dit`` creates pixel tokens and
        calibrated time modulation, while ``post_dit`` converts the head's
        asymmetric output back to full pixel velocity.
        """
        pre_state = self.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
        )
        x_tokens = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_attn_mask = pre_state["context_mask"]
        self_attn_mask = self.build_video_to_video_mask(
            video_seq_len=x_tokens.shape[1],
            video_tokens_per_group=int(
                pre_state["meta"]["tokens_per_group"]
            ),
            device=x_tokens.device,
        ) if self.video_attention_mask_mode != "bidirectional" else None

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x_tokens = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x_tokens,
                    context_emb,
                    t_mod,
                    freqs,
                    context_mask=context_attn_mask,
                    self_attn_mask=self_attn_mask,
                )
            else:
                x_tokens = block(
                    x_tokens,
                    context_emb,
                    t_mod,
                    freqs,
                    context_mask=context_attn_mask,
                    self_attn_mask=self_attn_mask,
                )
        return self.post_dit(x_tokens, pre_state)
