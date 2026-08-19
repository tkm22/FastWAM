import inspect
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from asymflow.projection import ProjectionArtifact
from fastwam.utils.logging_config import get_logger

from .io import ModelConfig, hash_model_file, load_state_dict
from ..wan_video_dit import WanVideoDiT
from ..wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder

logger = get_logger(__name__)
SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"
WAN_VIDEO_DIT_MODEL_HASH = "1f5ab7703c6fc803fdded85ff040c316"


@dataclass
class Wan22LoadedComponents:
    dit: WanVideoDiT
    text_encoder: WanTextEncoder | None
    tokenizer: HuggingfaceTokenizer | None
    dit_path: str
    text_encoder_path: str | None
    tokenizer_path: str | None


WAN22_MODEL_REGISTRY = [
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": WanTextEncoder,
    },
]


def _validate_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must be a dict, got {type(dit_config)}")

    validated = dict(dit_config)

    signature = inspect.signature(WanVideoDiT.__init__)
    allowed_keys = set()
    required_keys = set()
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        allowed_keys.add(name)
        if param.default is inspect.Signature.empty:
            required_keys.add(name)

    unknown_keys = sorted(set(validated) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in `dit_config`: {unknown_keys}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )

    missing_keys = sorted(required_keys - set(validated))
    if missing_keys:
        raise ValueError(
            f"Missing required keys in `dit_config`: {missing_keys}. "
            "Please specify all required WanVideoDiT constructor args."
        )

    return validated


def _load_registered_model(
    path,
    model_name: str,
    torch_dtype: torch.dtype,
    device: str,
):
    model_hash = hash_model_file(path)

    matched_config = None
    for config in WAN22_MODEL_REGISTRY:
        if config["model_hash"] == model_hash and config["model_name"] == model_name:
            matched_config = config
            break
    if matched_config is None:
        raise ValueError(
            f"Cannot detect model type for {model_name}. File: {path}. "
            f"Model hash: {model_hash}. This standalone package follows DiffSynth hash-based loading."
        )
    model_class = matched_config["model_class"]
    model = model_class()
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch_dtype)
    return model


def load_wan_video_dit_state_dict(
    path,
    torch_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Load the source latent Wan checkpoint as tensors only.

    The source ``WanVideoDiT`` class is not instantiated: this branch's
    public class already has pixel I/O.  The returned source tensors feed
    :func:`build_asym_pixel_state_dict`, which preserves every Transformer
    body parameter and rewrites only input/output video projections.
    """
    model_hash = hash_model_file(path)
    if model_hash != WAN_VIDEO_DIT_MODEL_HASH:
        raise ValueError(
            f"Cannot detect Wan video DiT checkpoint: {path}; hash={model_hash}"
        )
    return load_state_dict(path, torch_dtype=torch_dtype, device="cpu")


def build_asym_pixel_state_dict(
    source_sd: Mapping[str, torch.Tensor],
    target: WanVideoDiT,
    artifact: ProjectionArtifact,
    rewrite_device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Build every target key and fail before any partial pretrained load.

    Common Transformer keys are copied by exact shape.  The three Wan video
    boundary projections are the only deliberate shape changes:

    ``first_patch = W_in @ A_first.T``;
    ``future_patch = W_in @ A_future.T``; and
    ``pixel_head = A_future @ W_out``.

    Source input/output bias is copied/lifted consistently.  Artifact
    matrices and scales fill target buffers.  Every target key must be
    present, and the caller loads this result with ``strict=True``.
    """
    target_sd = target.state_dict()
    pixel_only_keys = {
        "first_patch_embedding.weight",
        "first_patch_embedding.bias",
        "future_patch_embedding.weight",
        "future_patch_embedding.bias",
        "A_first",
        "A_future",
        "scale_first",
        "scale_future",
    }
    expected_source_keys = (set(target_sd) - pixel_only_keys) | {
        "patch_embedding.weight",
        "patch_embedding.bias",
    }
    missing_source = sorted(expected_source_keys - set(source_sd))
    unexpected_source = sorted(set(source_sd) - expected_source_keys)
    if missing_source or unexpected_source:
        raise RuntimeError(
            "Wan source checkpoint key mismatch: "
            f"missing={missing_source}, unexpected={unexpected_source}"
        )

    rewrite_device = torch.device(rewrite_device)
    source_input = source_sd["patch_embedding.weight"]
    A_first = artifact.A_first.to(
        device=rewrite_device,
        dtype=source_input.dtype,
    )
    A_future = artifact.A_future.to(
        device=rewrite_device,
        dtype=source_input.dtype,
    )
    if (
        tuple(A_first.shape) != (3072, 192)
        or tuple(A_future.shape) != (12288, 192)
    ):
        raise ValueError(
            "Expected A_first [3072,192], A_future [12288,192]; "
            f"got {tuple(A_first.shape)}, {tuple(A_future.shape)}"
        )
    W_in = source_input.flatten(1).to(rewrite_device)  # [3072,192]
    W_out = source_sd["head.head.weight"].to(rewrite_device)  # [192,3072]
    b_out = source_sd["head.head.bias"].to(rewrite_device)
    rewritten = {
        "first_patch_embedding.weight": (W_in @ A_first.T).reshape_as(
            target_sd["first_patch_embedding.weight"]
        ),
        "first_patch_embedding.bias": source_sd["patch_embedding.bias"].to(
            rewrite_device
        ),
        "future_patch_embedding.weight": W_in @ A_future.T,
        "future_patch_embedding.bias": source_sd["patch_embedding.bias"].to(rewrite_device),
        "head.head.weight": A_future @ W_out,
        "head.head.bias": A_future @ b_out,
        "A_first": artifact.A_first.to(rewrite_device),
        "A_future": artifact.A_future.to(rewrite_device),
        "scale_first": artifact.scale_first.reshape(()),
        "scale_future": artifact.scale_future.reshape(()),
    }
    adapted: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for key, value in target_sd.items():
        if key in rewritten:
            candidate = rewritten[key]
        elif key in source_sd and source_sd[key].shape == value.shape:
            candidate = source_sd[key]
        else:
            missing.append(key)
            continue
        if candidate.shape != value.shape:
            raise RuntimeError(
                f"adapted key {key} has {tuple(candidate.shape)}, "
                f"expected {tuple(value.shape)}"
            )
        adapted[key] = candidate.to(
            device=value.device,
            dtype=value.dtype,
        )
    if missing:
        raise RuntimeError(
            "Asymmetric pixel target has unfilled learned/buffer keys: "
            + ", ".join(missing)
        )
    return adapted


def _resolve_configs(model_id: str, tokenizer_model_id: str, redirect_common_files: bool = True):
    dit_config = ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors")
    text_config = ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
    tokenizer_config = ModelConfig(model_id=tokenizer_model_id, origin_file_pattern="google/umt5-xxl/")

    if redirect_common_files:
        redirect_dict = {
            "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
        }
        text_config.model_id, text_config.origin_file_pattern = redirect_dict[text_config.origin_file_pattern]
    return dit_config, text_config, tokenizer_config


def load_wan22_ti2v_5b_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 512,
    redirect_common_files: bool = True,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
    projection_artifact_path: Optional[str] = None,
):
    """Build a VAE-free pixel Wan component bundle from a Wan checkpoint.

    Runtime loading downloads the original DiT checkpoint and optional text
    encoder/tokenizer.  It deliberately does not download or instantiate the
    Wan VAE.  The fitted projection artifact is required to construct pixel
    I/O and to adapt source checkpoint weights before the resulting DiT is
    moved to the requested device/dtype.
    """
    logger.info("Loading Wan2.2-TI2V-5B components...")
    start = time.time()

    if dit_config is None:
        raise ValueError("`dit_config` is required for Wan2.2-TI2V-5B loading.")
    validated_dit_config = _validate_dit_config(dit_config)

    dit_model_config, text_config, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )

    if load_text_encoder:
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()

    if projection_artifact_path is None:
        raise ValueError(
            "Pixel Wan construction requires `projection_artifact_path`. "
            "Run tools/fit_asym_fastwam_procrustes.py first."
        )
    artifact = ProjectionArtifact.load(projection_artifact_path)
    if skip_dit_load_from_pretrain:
        logger.info(
            "Skipping pretrained video DiT load (`skip_dit_load_from_pretrain=True`); "
            "initializing video expert randomly and expecting checkpoint override."
        )
        dit = WanVideoDiT(
            **validated_dit_config,
            projection_artifact=artifact.state_dict(),
        ).to(device=device, dtype=torch_dtype)
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    else:
        dit_model_config.download_if_necessary()
        source_sd = load_wan_video_dit_state_dict(
            dit_model_config.path,
            torch_dtype=torch_dtype,
        )
        dit = WanVideoDiT(
            **validated_dit_config,
            projection_artifact=artifact.state_dict(),
        )
        adapted = build_asym_pixel_state_dict(
            source_sd,
            dit,
            artifact,
            rewrite_device=device,
        )
        dit.load_state_dict(adapted, strict=True)
        del source_sd, adapted
        dit = dit.to(device=device, dtype=torch_dtype)
        dit_path = str(dit_model_config.path)
    text_encoder: WanTextEncoder | None = None
    tokenizer: HuggingfaceTokenizer | None = None
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    if load_text_encoder:
        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
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
    else:
        logger.info(
            "Skipping pretrained text encoder/tokenizer load (`load_text_encoder=False`); "
            "training must provide cached `context/context_mask`."
        )
    logger.info("Finished loading Wan2.2-TI2V-5B components in %.2f seconds.", time.time() - start)
    return Wan22LoadedComponents(
        dit=dit,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dit_path=dit_path,
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
    )
