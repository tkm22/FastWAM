from __future__ import annotations

from dataclasses import dataclass
import inspect
import time
from typing import Any, Mapping

import torch

from fastwam.utils.logging_config import get_logger

from ..jit_pixel_video_dit import JiTPixelWanVideoDiT
from ..wan_video_dit import WanVideoDiT
from ..wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from .io import ModelConfig, hash_model_file, load_state_dict

logger = get_logger(__name__)

WAN22_TI2V_5B_DIT_HASH = "1f5ab7703c6fc803fdded85ff040c316"
WAN_TEXT_ENCODER_HASH = "9c8818c2cbea55eca56c7b447df170da"
SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"


@dataclass
class JiTPixelWan22LoadedComponents:
    dit: JiTPixelWanVideoDiT
    text_encoder: WanTextEncoder | None
    tokenizer: HuggingfaceTokenizer | None
    dit_path: str
    text_encoder_path: str | None
    tokenizer_path: str | None


def _validate_jit_pixel_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must be a dict, got {type(dit_config)}")
    signature = inspect.signature(JiTPixelWanVideoDiT.__init__)
    allowed = {name for name in signature.parameters if name != "self"}
    required = {
        name
        for name, param in signature.parameters.items()
        if name != "self" and param.default is inspect.Signature.empty
    }
    unknown = sorted(set(dit_config) - allowed)
    missing = sorted(required - set(dit_config))
    if unknown or missing:
        raise ValueError(
            f"Invalid JiT pixel DiT config: unknown={unknown}, missing={missing}"
        )
    return dict(dit_config)


def _source_wan_dit_config(dit_config: Mapping[str, Any]) -> dict[str, Any]:
    pixel_only = {"pixel_patch_size", "future_tube_size", "bottleneck_dim"}
    source_config = {key: value for key, value in dit_config.items() if key not in pixel_only}
    signature = inspect.signature(WanVideoDiT.__init__)
    allowed = {name for name in signature.parameters if name != "self"}
    required = {
        name
        for name, param in signature.parameters.items()
        if name != "self" and param.default is inspect.Signature.empty
    }
    unknown = sorted(set(source_config) - allowed)
    missing = sorted(required - set(source_config))
    if unknown or missing:
        raise ValueError(f"Invalid source Wan DiT config: unknown={unknown}, missing={missing}")
    return source_config


def _load_wan_text_encoder(
    path: str | list[str],
    *,
    torch_dtype: torch.dtype,
    device: str,
) -> WanTextEncoder:
    model_hash = hash_model_file(path)
    if model_hash != WAN_TEXT_ENCODER_HASH:
        raise ValueError(f"Cannot detect Wan text encoder: {path}; hash={model_hash}")
    model = WanTextEncoder()
    model.load_state_dict(
        load_state_dict(path, torch_dtype=torch_dtype, device="cpu"),
        strict=True,
    )
    return model.to(device=device, dtype=torch_dtype)


def build_jit_pixel_state_dict(
    source_sd: Mapping[str, torch.Tensor],
    target: JiTPixelWanVideoDiT,
) -> dict[str, torch.Tensor]:
    """Strictly copy the Wan body and bridge its 192-D input projection."""
    target_sd = target.state_dict()
    new_target_keys = {
        "first_patch_down.weight",
        "future_patch_down.weight",
        "wan_patch_projection.weight",
        "wan_patch_projection.bias",
        "head.head.weight",
        "head.head.bias",
        "head.modulation",
    }
    source_boundary_keys = {
        "patch_embedding.weight",
        "patch_embedding.bias",
        "head.head.weight",
        "head.head.bias",
        "head.modulation",
    }
    target_common = set(target_sd) - new_target_keys
    source_common = set(source_sd) - source_boundary_keys
    if target_common != source_common:
        raise RuntimeError(
            "Wan/JiT shared state mismatch: "
            f"missing={sorted(target_common - source_common)}, "
            f"unexpected={sorted(source_common - target_common)}"
        )
    if target.bottleneck_dim != target.source_patch_dim:
        raise ValueError(
            "Pretrained Wan bridge requires bottleneck_dim=source patch dim=192, got "
            f"{target.bottleneck_dim}"
        )

    source_patch_weight = source_sd["patch_embedding.weight"].flatten(1)
    source_patch_bias = source_sd["patch_embedding.bias"]
    if source_patch_weight.shape != target_sd["wan_patch_projection.weight"].shape:
        raise RuntimeError(
            "Wan patch projection shape mismatch: "
            f"source={tuple(source_patch_weight.shape)}, "
            f"target={tuple(target_sd['wan_patch_projection.weight'].shape)}"
        )

    adapted: dict[str, torch.Tensor] = {}
    for key, target_value in target_sd.items():
        if key in target_common:
            candidate = source_sd[key]
        elif key == "wan_patch_projection.weight":
            candidate = source_patch_weight
        elif key == "wan_patch_projection.bias":
            candidate = source_patch_bias
        else:
            candidate = target_value
        if candidate.shape != target_value.shape:
            raise RuntimeError(
                f"Adapted key {key} has shape {tuple(candidate.shape)}, "
                f"expected {tuple(target_value.shape)}"
            )
        adapted[key] = candidate.to(device=target_value.device, dtype=target_value.dtype)
    return adapted


def load_jit_pixel_wan22_components(
    *,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 512,
    redirect_common_files: bool = True,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
) -> JiTPixelWan22LoadedComponents:
    """Load a VAE-free JiT pixel expert with a strict Wan body transfer."""
    start = time.time()
    if dit_config is None:
        raise ValueError("`dit_config` is required for JiT pixel Wan loading")
    validated = _validate_jit_pixel_dit_config(dit_config)
    source_config = _source_wan_dit_config(validated)

    dit_model_config = ModelConfig(
        model_id=model_id,
        origin_file_pattern="diffusion_pytorch_model*.safetensors",
    )
    text_config = ModelConfig(
        model_id=model_id,
        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
    )
    tokenizer_config = ModelConfig(
        model_id=tokenizer_model_id,
        origin_file_pattern="google/umt5-xxl/",
    )
    if redirect_common_files:
        text_config.model_id = "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
        text_config.origin_file_pattern = "models_t5_umt5-xxl-enc-bf16.safetensors"

    target = JiTPixelWanVideoDiT(**validated)
    if skip_dit_load_from_pretrain:
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    else:
        dit_model_config.download_if_necessary()
        model_hash = hash_model_file(dit_model_config.path)
        if model_hash != WAN22_TI2V_5B_DIT_HASH:
            raise ValueError(
                f"Cannot detect Wan2.2-TI2V-5B DiT: {dit_model_config.path}; hash={model_hash}"
            )
        source_sd = load_state_dict(
            dit_model_config.path,
            torch_dtype=torch_dtype,
            device="cpu",
        )
        target.load_state_dict(build_jit_pixel_state_dict(source_sd, target), strict=True)
        dit_path = str(dit_model_config.path)
    target = target.to(device=device, dtype=torch_dtype)

    text_encoder: WanTextEncoder | None = None
    tokenizer: HuggingfaceTokenizer | None = None
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    if load_text_encoder:
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()
        text_encoder = _load_wan_text_encoder(
            text_config.path,
            torch_dtype=torch_dtype,
            device=device,
        )
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=int(tokenizer_max_len),
            clean="whitespace",
        )
        text_encoder_path = str(text_config.path)
        tokenizer_path = str(tokenizer_config.path)

    logger.info(
        "Loaded VAE-free JiT pixel Wan components in %.2f seconds",
        time.time() - start,
    )
    return JiTPixelWan22LoadedComponents(
        dit=target,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dit_path=dit_path,
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
    )
