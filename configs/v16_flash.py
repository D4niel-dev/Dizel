"""
configs/v16_flash.py — Dizel 1.6 Flash-family configuration.

This file configures the Dizel 1.6 Flash-family architecture, currently
instantiated as the Preview M1 build. Architecture and training values come
from the approved v1.6 MoE plan: a 4-expert, top-2 dropless MoE decoder with
GQA, SwiGLU, RMSNorm and RoPE, trained on the Qwen2.5 151,936-token
vocabulary.

The future ``Dizel 1.6 Pro`` capability tier (1B–3B parameters) is not
configured here.

Use ``DIZEL_16_FLASH_CONFIG`` (alias ``V16_FLASH_CONFIG``) as the single
source of truth. ``ModelConfigV16Flash.param_estimate()`` returns the total
parameter count and ``active_param_estimate()`` the per-token active count.
"""

from dataclasses import dataclass, field

from config import ModelConfigV15, PretrainConfigV15

__all__ = [
    "ModelConfigV16Flash",
    "TokenizerConfigV16Flash",
    "PretrainConfigV16Flash",
    "PosttrainConfigV16Flash",
    "SFTConfigV16Flash",
    "DistillConfigV16Flash",
    "DPOConfigV16Flash",
    "DizelV16FlashConfig",
    "DIZEL_16_FLASH_CONFIG",
    "V16_FLASH_CONFIG",
]


@dataclass
class ModelConfigV16Flash(ModelConfigV15):
    """Dizel 1.6 Flash-family architecture (~435.3M total, ~293.7M active).

    Instantiated as the Preview M1 build; ``Dizel 1.6 Pro`` is not
    configured here.
    """

    vocab_size: int = 151936
    context_length: int = 2048
    d_model: int = 768
    n_layers: int = 20
    n_heads: int = 12
    kv_heads: int = 6
    head_dim: int = 64
    intermediate_size: int = 1536       # per-expert SwiGLU hidden size
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6
    use_rope: bool = True
    rope_theta: float = 10000.0
    bias: bool = False
    weight_tying: bool = True
    dropout: float = 0.0

    # Sparse MoE (dropless top-2; no shared expert)
    num_experts: int = 4
    top_k: int = 2
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    router_init_std: float = 0.02
    router_logits_dtype: str = "float32"

    # ------------------------------------------------------------------
    # Parameter accounting
    # ------------------------------------------------------------------
    def _layer_params(self, active: bool) -> float:
        d = self.d_model
        attn = d * d * (1 + 2 * self.kv_heads / self.n_heads) + d * d
        experts = 3 * d * self.intermediate_size * (
            self.top_k if active else self.num_experts
        )
        router = d * self.num_experts
        # Two RMSNorm gains per layer (input_norm + post_attn_norm).
        norms = 2 * d
        return attn + experts + router + norms

    def param_estimate(self) -> int:
        """Total parameters, including inactive experts (tied LM head adds 0)."""
        embed = self.vocab_size * self.d_model
        total = (
            embed
            + self.n_layers * self._layer_params(active=False)
            + self.d_model  # final RMSNorm
        )
        if not self.weight_tying:
            total += self.vocab_size * self.d_model
        return int(round(total))

    def active_param_estimate(self) -> int:
        """Parameters used per token: top-k experts + shared attention/embeddings."""
        embed = self.vocab_size * self.d_model
        total = (
            embed
            + self.n_layers * self._layer_params(active=True)
            + self.d_model  # final RMSNorm
        )
        if not self.weight_tying:
            total += self.vocab_size * self.d_model
        return int(round(total))


@dataclass
class TokenizerConfigV16Flash:
    """Qwen2.5 tokenizer, loaded locally at training time.

    The tokenizer actually emits 151,665 ids.  The model keeps 151,936
    embedding rows because that is Qwen2.5's padded model vocabulary and the
    target for future token-level distillation.  The extra rows are never
    produced by the tokenizer and are ignored by cross-entropy.
    """

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    vocab_size: int = 151665
    trust_remote_code: bool = False
    local_files_only: bool = False


@dataclass
class PretrainConfigV16Flash(PretrainConfigV15):
    """Flash-family 3B-token Phase A run (45,777 steps at 65,536 tokens/step).

    Instantiates the Preview M1 build directories and run name.
    """

    # Raw-text fallback.  When tokenized_data_dir is set, the trainer reads
    # laptop-packed uint32 shards and never touches data_path.
    data_path: str = "data/formatted/v16_pretrain.txt"
    tokenized_data_dir: str = ""
    batch_size: int = 2
    grad_accum: int = 16
    max_length: int = 2048

    # Flat aliases (kept in sync with phase1_* below)
    max_steps: int = 45777
    lr: float = 2e-4
    min_lr: float = 2e-5
    warmup_steps: int = 1000

    phase1_steps: int = 45777
    phase1_lr: float = 2e-4
    phase1_min_lr: float = 2e-5
    phase1_warmup: int = 1000

    weight_decay: float = 0.1
    grad_clip: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    eval_interval: int = 500
    eval_iters: int = 50
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints/dizel-1.6-preview-m1"
    log_dir: str = "runs/dizel-1.6-preview-m1"
    run_name: str = "dizel-1.6-preview-m1-pretrain"

    # Data loading
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
class PosttrainConfigV16Flash:
    """Shared post-training controls for the v1.6 Flash path.

    The v1.5 post-training scripts assumed a SentencePiece tokenizer and a
    dense FFN.  These configs are intentionally separate: v1.5 checkpoints and
    its dense control remain loadable through the old entry points, while
    v1.6 SFT/KD/DPO runs always use Qwen ChatML and the MoE checkpoint format.
    """

    base_checkpoint: str = ""
    data_path: str = ""
    output_dir: str = "checkpoints/dizel-1.6-preview-m1-posttrain"
    run_name: str = "dizel-1.6-preview-m1-posttrain"

    batch_size: int = 1
    grad_accum: int = 16
    max_steps: int = 2000
    warmup_steps: int = 100
    lr: float = 2e-5
    min_lr: float = 2e-6
    weight_decay: float = 0.05
    grad_clip: float = 1.0

    max_length: int = 1024
    val_fraction: float = 0.05
    seed: int = 42

    use_amp: bool = True
    amp_dtype: str = "float16"
    gradient_checkpointing: bool = True
    eval_interval: int = 250
    eval_iters: int = 50
    save_interval: int = 500
    log_interval: int = 25

    drive_checkpoint_dir: str = ""


@dataclass
class SFTConfigV16Flash(PosttrainConfigV16Flash):
    """ChatML assistant-only SFT for the v1.6 base checkpoint."""

    data_path: str = "data/v16/posttrain/sft.jsonl"
    output_dir: str = "checkpoints/dizel-1.6-preview-m1-sft"
    run_name: str = "dizel-1.6-preview-m1-sft"
    batch_size: int = 2
    max_steps: int = 3000
    lr: float = 2e-5
    min_lr: float = 2e-6
    max_length: int = 1024


@dataclass
class DistillConfigV16Flash(PosttrainConfigV16Flash):
    """Full-parameter token-level KD from Qwen2.5-0.5B-Instruct.

    v1.5 used LoRA because its non-Qwen vocabulary made logit alignment
    impossible.  v1.6 deliberately shares Qwen's 151,936-row vocabulary, so
    the teacher and student logits align directly and the plain model remains
    checkpoint-compatible.
    """

    data_path: str = "data/v16/posttrain/distill.jsonl"
    output_dir: str = "checkpoints/dizel-1.6-preview-m1-distill"
    run_name: str = "dizel-1.6-preview-m1-distill"
    teacher_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    teacher_local_files_only: bool = False
    batch_size: int = 1
    max_steps: int = 2000
    lr: float = 1e-5
    min_lr: float = 1e-6
    max_length: int = 1024
    kl_weight: float = 0.5
    temperature: float = 2.0


@dataclass
class DPOConfigV16Flash(PosttrainConfigV16Flash):
    """Experimental preference tuning on top of the v1.6 SFT checkpoint."""

    data_path: str = "data/v16/posttrain/dpo.jsonl"
    output_dir: str = "checkpoints/dizel-1.6-preview-m1-dpo"
    run_name: str = "dizel-1.6-preview-m1-dpo"
    reference_checkpoint: str = ""
    batch_size: int = 1
    max_steps: int = 1000
    lr: float = 5e-6
    min_lr: float = 5e-7
    max_length: int = 1536
    max_prompt_length: int = 768
    beta: float = 0.5
    label_smoothing: float = 0.05


@dataclass
class DizelV16FlashConfig:
    """Complete Dizel 1.6 Flash-family configuration + release identity.

    Instantiates the Preview M1 build. ``Dizel 1.6 Pro`` is a separate
    future capability tier and is not configured here.
    """

    model: ModelConfigV16Flash = field(default_factory=ModelConfigV16Flash)
    tokenizer: TokenizerConfigV16Flash = field(
        default_factory=TokenizerConfigV16Flash
    )
    pretrain: PretrainConfigV16Flash = field(
        default_factory=PretrainConfigV16Flash
    )
    sft: SFTConfigV16Flash = field(default_factory=SFTConfigV16Flash)
    distill: DistillConfigV16Flash = field(
        default_factory=DistillConfigV16Flash
    )
    dpo: DPOConfigV16Flash = field(default_factory=DPOConfigV16Flash)

    display_name: str = "Dizel 1.6 Flash Preview"
    family: str = "Dizel 1.6"
    tier: str = "Flash"
    canonical_slug: str = "dizel-1.6-flash"
    preview_slug: str = "dizel-1.6-preview-m1"
    release_stage: str = "preview"


DIZEL_16_FLASH_CONFIG = DizelV16FlashConfig()
V16_FLASH_CONFIG = DIZEL_16_FLASH_CONFIG
