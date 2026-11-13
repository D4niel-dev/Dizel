"""
configs/v16_pro.py — Dizel 1.6 Pro-family configuration.

This is the planned Pro capability tier for the v1.6 MoE family: a 1.5B–1.7B
total-parameter decoder with 8 experts and dropless top-4 routing. It shares
the Qwen2.5 151,936-row vocabulary, tokenizer, 2,048-token context, and token
cache layout with the Flash path, so data preparation does not need to be
repeated for the larger model.

The Pro build is not instantiated yet. Use ``DIZEL_16_PRO_CONFIG`` (alias
``V16_PRO_CONFIG``) as the single source of truth when the training run is
scheduled. ``ModelConfigV16Pro.param_estimate()`` returns the total parameter
count and ``active_param_estimate()`` the per-token active count.

Training defaults preserve the Flash effective batch: micro-batch 1 with
32-step accumulation, gradient checkpointing, mixed precision (bf16 where
native, fp16 on T4), and less frequent evaluation. The token cache is shared
with Flash instead of being rebuilt.

Memory contract: fp32 parameters plus gradients and fp32 AdamW states need
about 26 GB before activations, so this config does not fit a 16 GB T4 with
the current trainer. Run it on a 40 GB+ GPU, or add a sharded/8-bit optimizer
path before scheduling the 3B-token run.
"""

from dataclasses import dataclass, field

from .v16_flash import (
    ModelConfigV16Flash,
    PosttrainConfigV16Flash,
    PretrainConfigV16Flash,
    TokenizerConfigV16Flash,
)

__all__ = [
    "ModelConfigV16Pro",
    "TokenizerConfigV16Pro",
    "PretrainConfigV16Pro",
    "PosttrainConfigV16Pro",
    "SFTConfigV16Pro",
    "DistillConfigV16Pro",
    "DPOConfigV16Pro",
    "DizelV16ProConfig",
    "DIZEL_16_PRO_CONFIG",
    "V16_PRO_CONFIG",
]


@dataclass
class ModelConfigV16Pro(ModelConfigV16Flash):
    """Dizel 1.6 Pro MoE (~1.634B total, ~973M active per token).

    Pro doubles the expert count to 8 and activates 4 per token. The width and
    depth are kept modest so the sparse experts carry the capacity increase
    without making the dense attention path disproportionately expensive.
    """

    vocab_size: int = 151936
    context_length: int = 2048
    d_model: int = 1280
    n_layers: int = 24
    n_heads: int = 16
    kv_heads: int = 8
    head_dim: int = 80
    intermediate_size: int = 1792       # per-expert SwiGLU hidden size
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6
    use_rope: bool = True
    rope_theta: float = 10000.0
    bias: bool = False
    weight_tying: bool = True
    dropout: float = 0.0

    # Sparse MoE (dropless top-4; no shared expert)
    num_experts: int = 8
    top_k: int = 4
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    router_init_std: float = 0.02
    router_logits_dtype: str = "float32"


@dataclass
class TokenizerConfigV16Pro(TokenizerConfigV16Flash):
    """Qwen2.5 tokenizer shared with Flash for compatible distillation."""


@dataclass
class PretrainConfigV16Pro(PretrainConfigV16Flash):
    """Flash-compatible 3B-token Phase A with a lower-memory micro-batch.

    ``1 × 32 × 2,048`` gives the same 65,536 tokens per optimizer step as the
    Flash recipe, so the two runs remain comparable while activations per
    device stay lower. Reduce ``max_steps`` if compute, not memory, is the
    constraint.
    """

    data_path: str = "data/formatted/v16_pretrain.txt"
    tokenized_data_dir: str = ""
    batch_size: int = 1
    grad_accum: int = 32
    max_length: int = 2048

    # Flat aliases (kept in sync with phase1_* below)
    max_steps: int = 45777
    lr: float = 1.5e-4
    min_lr: float = 1.5e-5
    warmup_steps: int = 2000

    phase1_steps: int = 45777
    phase1_lr: float = 1.5e-4
    phase1_min_lr: float = 1.5e-5
    phase1_warmup: int = 2000

    weight_decay: float = 0.1
    grad_clip: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True

    # Pro evaluation is more expensive; keep the same wall-clock budget.
    eval_interval: int = 1000
    eval_iters: int = 25
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints/dizel-1.6-pro"
    log_dir: str = "runs/dizel-1.6-pro"
    run_name: str = "dizel-1.6-pro-pretrain"

    # Same tokenizer and sequence length as Flash: reuse its packed shards.
    train_split: float = 0.90
    seed: int = 42
    shard_size_mb: int = 100
    cache_dir: str = ".cache/tokenized/dizel-1.6"
    num_workers: int = 2
    resume_enabled: bool = True
    reshuffle_every_n_steps: int = 500

    # Release/logging cadence
    drive_checkpoint_dir: str = ""
    log_interval: int = 50
    milestone_interval: int = 5000


@dataclass
class PosttrainConfigV16Pro(PosttrainConfigV16Flash):
    """Shared post-training controls for the larger Pro checkpoint."""

    base_checkpoint: str = ""
    data_path: str = ""
    output_dir: str = "checkpoints/dizel-1.6-pro-posttrain"
    run_name: str = "dizel-1.6-pro-posttrain"

    batch_size: int = 1
    grad_accum: int = 32
    max_steps: int = 2000
    warmup_steps: int = 100
    lr: float = 1e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    grad_clip: float = 1.0

    max_length: int = 1024
    val_fraction: float = 0.05
    seed: int = 42

    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    eval_interval: int = 250
    eval_iters: int = 25
    save_interval: int = 500
    log_interval: int = 25

    drive_checkpoint_dir: str = ""


@dataclass
class SFTConfigV16Pro(PosttrainConfigV16Pro):
    """ChatML assistant-only SFT for the Pro base checkpoint."""

    data_path: str = "data/v16/posttrain/sft.jsonl"
    output_dir: str = "checkpoints/dizel-1.6-pro-sft"
    run_name: str = "dizel-1.6-pro-sft"
    batch_size: int = 1
    max_steps: int = 3000
    lr: float = 1e-5
    min_lr: float = 1e-6
    max_length: int = 1024


@dataclass
class DistillConfigV16Pro(PosttrainConfigV16Pro):
    """Full-parameter token-level KD from a Qwen2.5 1.5B teacher.

    Teacher and student share the Qwen2.5 vocabulary, so logits align directly.
    Swap in a larger Qwen2.5 teacher when teacher memory allows.
    """

    data_path: str = "data/v16/posttrain/distill.jsonl"
    output_dir: str = "checkpoints/dizel-1.6-pro-distill"
    run_name: str = "dizel-1.6-pro-distill"
    teacher_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    teacher_local_files_only: bool = False
    batch_size: int = 1
    max_steps: int = 2000
    lr: float = 5e-6
    min_lr: float = 5e-7
    max_length: int = 1024
    kl_weight: float = 0.5
    temperature: float = 2.0


@dataclass
class DPOConfigV16Pro(PosttrainConfigV16Pro):
    """Experimental preference tuning on top of the Pro SFT checkpoint."""

    data_path: str = "data/v16/posttrain/dpo.jsonl"
    output_dir: str = "checkpoints/dizel-1.6-pro-dpo"
    run_name: str = "dizel-1.6-pro-dpo"
    reference_checkpoint: str = ""
    batch_size: int = 1
    max_steps: int = 1000
    lr: float = 2e-6
    min_lr: float = 2e-7
    max_length: int = 1536
    max_prompt_length: int = 768
    beta: float = 0.5
    label_smoothing: float = 0.05


@dataclass
class DizelV16ProConfig:
    """Complete Dizel 1.6 Pro configuration + planned release identity."""

    model: ModelConfigV16Pro = field(default_factory=ModelConfigV16Pro)
    tokenizer: TokenizerConfigV16Pro = field(
        default_factory=TokenizerConfigV16Pro
    )
    pretrain: PretrainConfigV16Pro = field(
        default_factory=PretrainConfigV16Pro
    )
    sft: SFTConfigV16Pro = field(default_factory=SFTConfigV16Pro)
    distill: DistillConfigV16Pro = field(
        default_factory=DistillConfigV16Pro
    )
    dpo: DPOConfigV16Pro = field(default_factory=DPOConfigV16Pro)

    display_name: str = "Dizel 1.6 Pro"
    family: str = "Dizel 1.6"
    tier: str = "Pro"
    canonical_slug: str = "dizel-1.6-pro"
    preview_slug: str = "dizel-1.6-pro"
    release_stage: str = "planned"


DIZEL_16_PRO_CONFIG = DizelV16ProConfig()
V16_PRO_CONFIG = DIZEL_16_PRO_CONFIG
