# -*- coding: utf-8 -*-
"""
configuration_dizel.py -- Dizel v1.5 config for HuggingFace remote code
───────────────────────────────────────────────────────────────────────
Dizel v1.5 -- a custom HF remote-code configuration class.

This file mirrors the hyper-parameters of ``ModelConfigV15`` in the Dizel
repository (config.py). It is shipped NEXT TO the exported model directory so
that ``AutoConfig.from_pretrained(<dir>, trust_remote_code=True)`` can
reconstruct the architecture exactly.

Architecture: Qwen-style, GQA + SwiGLU + RMSNorm + RoPE, optional weight tying.
"""

from transformers import PretrainedConfig

# Peculiar of the config: keep everything in a plain dataclass-less class.
# HF expects ``model_type`` to match the ``model_type`` key in config.json.


class DizelConfig(PretrainedConfig):
    """Configuration class for the Dizel v1.5 causal language model.

    Mirrors ``ModelConfigV15`` from the Dizel repo (config.py) exactly so the
    exported ``model.safetensors`` maps 1:1 onto the checkpoint keys.
    """

    model_type = "dizel"

    # A few aliases HF's generic pieces look up from ``config``
    attribute_map = {
        "hidden_size": "d_model",
        "max_position_embeddings": "context_length",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
        "num_key_value_heads": "kv_heads",
    }

    def __init__(
        self,
        vocab_size: int = 100000,
        context_length: int = 2048,
        d_model: int = 1024,
        n_layers: int = 28,
        n_heads: int = 16,
        kv_heads: int = 8,
        head_dim: int = 64,
        intermediate_size: int = 3072,
        norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        bias: bool = False,
        weight_tying: bool = True,
        dropout: float = 0.0,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.max_position_embeddings = context_length
        self.d_model = d_model
        self.hidden_size = d_model
        self.n_layers = n_layers
        self.num_hidden_layers = n_layers
        self.n_heads = n_heads
        self.num_attention_heads = n_heads
        self.kv_heads = kv_heads
        self.num_key_value_heads = kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.norm_eps = norm_eps
        self.rms_norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.bias = bias
        self.weight_tying = weight_tying
        self.tie_word_embeddings = weight_tying
        self.dropout = dropout
        self.initializer_range = initializer_range

        super().__init__(**kwargs)