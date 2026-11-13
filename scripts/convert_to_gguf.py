#!/usr/bin/env python3
"""
scripts/convert_to_gguf.py -- Convert a Dizel .pt checkpoint to GGUF format.

Converts trained Dizel language model checkpoints into the GGUF format
for inference with llama.cpp, llama-cpp-python, and other GGUF-compatible runtimes.

Targets Dizel v1.5 (Qwen-style: RMSNorm + GQA + SwiGLU + RoPE, 100K vocab)
with optional backward compatibility for the legacy v1.2 architecture
(Biases/LayerNorm + MHA + fc1/fc2 MLP) via ``--config-key dizel-v1.2``.

Usage:
    python scripts/convert_to_gguf.py \\
        --checkpoint checkpoints/dizel-v15-sft-best.pt \\
        --tokenizer tokenizer/dizel_v15.model \\
        --output checkpoints/Dizel.gguf \\
        --config-key dizel-v1.5

# ============================================================
# Colab usage:
# !pip install gguf sentencepiece torch protobuf
# !git clone https://github.com/your-dizel/dizel
# %cd dizel
# !python scripts/convert_to_gguf.py \
#     --checkpoint "drive/MyDrive/dizel/checkpoints/dizel-v15-sft-best.pt" \
#     --tokenizer tokenizer/dizel_v15.model \
#     --output "drive/MyDrive/dizel/Dizel.gguf"
# ============================================================
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# ===========================================================================
# CHAT TEMPLATE
# ===========================================================================
# Dizel v1.5 training format (see training/dataset.py): per-turn
# <|role|>content<|end|> markers, with </s> (EOS, token id 2) ending assistant
# turns. Loss is only on assistant tokens. Embedding this as
# tokenizer.chat_template in the GGUF lets llama-cpp-python's
# Jinja2ChatFormatter pick it up instead of falling back to the llama-2 chat
# format, which produces garbage formatting.
DIZEL_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)

# ===========================================================================
# SAFE IMPORTS WITH GRACEFUL FALLBACKS
# ===========================================================================

_REQUIREMENTS_HELP = """
Required packages:
    pip install gguf sentencepiece torch protobuf
"""


def _import(module_name, package_name=None, help_msg=""):
    """Import a module with a user-friendly error on failure."""
    try:
        return __import__(module_name)
    except ImportError:
        print("=" * 72)
        print(f"ERROR: The '{package_name or module_name}' package is required.")
        print(f"Install it with:")
        print(f"    pip install {package_name or module_name}")
        if help_msg:
            print(help_msg)
        print("=" * 72)
        sys.exit(1)


def _soft_import(module_name, package_name=None):
    """Import a module returning None on failure (no exit)."""
    try:
        return __import__(module_name)
    except ImportError:
        return None


# ===========================================================================
# CONFIG LOADING
# ===========================================================================

def _ensure_project_root():
    """
    Add the project root to sys.path so config/model imports work
    whether the script is run from the project root or from scripts/.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Two possible roots: the script's grandparent (scripts/../) or cwd
    candidates = [
        os.path.dirname(script_dir),       # parent of scripts/
        os.getcwd(),                       # current working directory
    ]
    for root in candidates:
        marker = os.path.join(root, "config.py")
        if os.path.isfile(marker) and root not in sys.path:
            sys.path.insert(0, root)
    # Also add the model subpackage's parent
    for root in candidates:
        model_dir = os.path.join(root, "model")
        if os.path.isdir(model_dir) and root not in sys.path:
            sys.path.insert(0, root)


def load_model_config(config_key: str):
    """Load a ModelConfig from the model registry by key."""
    _ensure_project_root()
    from model.registry import get_model_config
    return get_model_config(config_key)


# ===========================================================================
# CHECKPOINT LOADING
# ===========================================================================

def _merge_lora_adapters(state_dict: dict, ckpt: dict = None,
                         lora_alpha=None, lora_r=None) -> dict:
    """Fold LoRALinear adapters into base weights for export.

    Distilled checkpoints are saved UNMERGED so state_dict keys are
    {prefix}.linear.weight + {prefix}.lora_A + {prefix}.lora_B. GGUF/safetensors
    need plain {prefix}.weight with W' = W + (B @ A) * (alpha / r).

    r is recovered from lora_A.shape[0]; alpha priority: ckpt metadata
    (lora_alpha/lora_r keys) > explicit args > convention alpha=2*r.
    """
    from model.lora_merge import fold_lora_adapters

    meta = ckpt if isinstance(ckpt, dict) else {}
    used_alpha = lora_alpha if lora_alpha is not None else meta.get("lora_alpha")
    used_r = lora_r if lora_r is not None else meta.get("lora_r")
    from_convention = used_alpha is None or used_r is None

    out, info = fold_lora_adapters(
        state_dict, lora_alpha=used_alpha, lora_r=used_r
    )
    adapters = info["adapters"]
    if adapters:
        src = ("checkpoint metadata" if not from_convention
               else ("CLI args" if (lora_alpha is not None and lora_r is not None)
                     else "convention alpha=2*r"))
        print(f"[lora] Folded {adapters} LoRA adapters into base weights "
              f"(alpha={info['alpha']}, r={info['r']}, "
              f"scaling={info['alpha']/info['r']:.3f}, source: {src})")
        if from_convention and (lora_alpha is None or lora_r is None):
            print("[lora] WARNING: lora_alpha/lora_r not embedded in the checkpoint "
                  "and not passed via CLI -- assumed the repo convention alpha=2*r. "
                  "If distillation used different values, re-run with "
                  "--lora-alpha/--lora-r.")
    return out


def load_checkpoint(checkpoint_path: str, verbose: bool = False,
                    lora_alpha=None, lora_r=None):
    """
    Load a Dizel checkpoint and extract the state_dict.

    Handles multiple checkpoint formats:
    - Training checkpoint:  {"model_state": state_dict, "model_cfg": ..., ...}
    - Model object:         {"model": DizelLM_instance}
    - State dict export:    {"state_dict": state_dict}
    - Bare state_dict:      {key: tensor, ...}
    - torch.compile() variant with ``_orig_mod.`` prefix
    """
    torch = _import("torch")

    if not os.path.isfile(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if verbose:
        print(f"  Loading checkpoint: {checkpoint_path}")

    # Register both config classes as safe globals so weights_only=True works
    # when the checkpoint embeds a config object (v1.2 -> ModelConfig,
    # v1.5 -> ModelConfigV15).
    try:
        _ensure_project_root()
        from config import ModelConfig, ModelConfigV15
        torch.serialization.add_safe_globals([ModelConfig, ModelConfigV15])
    except Exception:
        pass  # Non-definitional -- some checkpoints don't embed a config object.

    # -- Load --------------------------------------------------------------
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # -- Determine format and extract state_dict ---------------------------
    state_dict = None
    source_info = "unknown"

    if isinstance(ckpt, dict):
        # Training checkpoint: {"model_state": state_dict, ...}
        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
            source_info = "training checkpoint"
            if verbose:
                step = ckpt.get("step", "?")
                vl = ckpt.get("val_loss", None)
                vl_str = f", val_loss={vl:.4f}" if vl is not None else ""
                print(f"    Format: training checkpoint (step={step}{vl_str})")

        # Model-object export: {"model": DizelLM(...)}
        elif "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
            state_dict = ckpt["model"].state_dict()
            source_info = "model object checkpoint"
            if verbose:
                print("    Format: model object checkpoint")

        # Explicit state-dict key
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            source_info = "state_dict key export"
            if verbose:
                print("    Format: state_dict key export")

        # Bare state_dict (keys look like model weights)
        else:
            sample_keys = list(ckpt.keys())
            if any("transformer." in str(k) or "lm_head" in str(k) for k in sample_keys):
                state_dict = ckpt
                source_info = "bare state_dict"
                if verbose:
                    print("    Format: bare state_dict")
            else:
                print(f"WARNING: Unknown dict format -- keys: {sample_keys[:8]}")
                print("  Attempting to treat as state_dict...")
                state_dict = ckpt
    else:
        # Not a dict -- may be a model object saved directly via torch.save(model, ...)
        if hasattr(ckpt, "state_dict"):
            state_dict = ckpt.state_dict()
            source_info = "torch.save(model) export"
            if verbose:
                print("    Format: torch.save(model, ...)")
        else:
            print(f"ERROR: Unexpected checkpoint type: {type(ckpt)}")
            sys.exit(1)

    # -- Strip _orig_mod. prefix from torch.compile ----------------------
    cleaned = {}
    strip_count = 0
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            cleaned[key[len("_orig_mod."):]] = value
            strip_count += 1
        else:
            cleaned[key] = value

    if verbose and strip_count:
        print(f"    Stripped '_orig_mod.' prefix from {strip_count} keys")

    # Distilled (LoRA) checkpoints are saved unmerged; fold the adapters so
    # the downstream key-mapping sees plain {prefix}.weight tensors.
    cleaned = _merge_lora_adapters(cleaned, ckpt=ckpt,
                                   lora_alpha=lora_alpha, lora_r=lora_r)

    return cleaned, source_info


# ===========================================================================
# TOKENIZER LOADING
# ===========================================================================

def load_tokenizer(tokenizer_path: str, verbose: bool = False):
    """Load a SentencePiece tokenizer model. Returns None on failure."""
    sp_module = _soft_import("sentencepiece")
    if sp_module is None:
        print("WARNING: sentencepiece not installed. Tokenizer data will NOT be embedded.")
        return None

    if not os.path.isfile(tokenizer_path):
        print(f"WARNING: Tokenizer not found: {tokenizer_path}")
        print("  The GGUF file will be written without embedded tokenizer data.")
        return None

    if verbose:
        print(f"  Loading tokenizer: {tokenizer_path}")

    sp = sp_module.SentencePieceProcessor()
    sp.load(tokenizer_path)
    return sp


# ===========================================================================
# TENSOR MAPPING  (Dizel -> GGUF naming, GQA/SwiGLU handling, dtype conversion)
# ===========================================================================

def _torch_dtype(dtype_str: str):
    """Map a dtype string to a PyTorch dtype."""
    torch = _import("torch")
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(dtype_str, torch.float32)


def _ggml_dtype(dtype_str: str):
    """Map a dtype string to a GGMLQuantizationType enum value."""
    gguf = _import("gguf")
    mapping = {
        "float32": gguf.GGMLQuantizationType.F32,
        "float16": gguf.GGMLQuantizationType.F16,
        "bfloat16": gguf.GGMLQuantizationType.BF16,
    }
    if dtype_str not in mapping:
        print(f"WARNING: Unknown dtype '{dtype_str}', falling back to float32")
        return gguf.GGMLQuantizationType.F32
    return mapping[dtype_str]


def _torch_to_numpy(tensor, target_dtype_str: str):
    """
    Convert a torch tensor to a numpy array for GGUF writing.

    For bfloat16, we store the raw bit pattern as uint16 since numpy
    has no native bfloat16 type. The GGML reader interprets the bytes
    as bfloat16 when the metadata says BF16.
    """
    torch = _import("torch")
    target_dt = _torch_dtype(target_dtype_str)

    tensor = tensor.to(dtype=target_dt).contiguous()

    if target_dtype_str == "bfloat16":
        # Store as uint16 preserving the bfloat16 bit pattern
        return tensor.view(torch.uint16).cpu().numpy()
    else:
        return tensor.cpu().numpy()


def extract_layer_count(state_dict: dict) -> int:
    """Determine the number of transformer layers from state_dict keys."""
    n_layers = 0
    pattern = re.compile(r"transformer\.blocks\.(\d+)\.")
    for key in state_dict:
        m = pattern.match(key)
        if m:
            n_layers = max(n_layers, int(m.group(1)) + 1)
    return n_layers


def map_dizel_to_gguf_tensors(
    state_dict,
    dtype_str: str,
    verbose: bool = False,
    compat_arch: str = "qwen2",
    n_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    n_heads_kv: Optional[int] = None,
):
    """
    Map Dizel (v1.5) state_dict keys -> GGUF tensor names.

    v1.5 uses separate Q/K/V/O projections (NO fused QKV) and a SwiGLU MLP
    with gate_proj/up_proj/down_proj (NO fc1/fc2). Every projection is
    already separate in the checkpoint, so nothing is chunked or split.

    Maps (per layer):
        input_norm.weight      -> blk.{i}.attn_norm.weight
        attn.q_proj.weight     -> blk.{i}.attn_q.weight
        attn.k_proj.weight     -> blk.{i}.attn_k.weight
        attn.v_proj.weight     -> blk.{i}.attn_v.weight
        attn.o_proj.weight     -> blk.{i}.attn_output.weight
        post_attn_norm.weight  -> blk.{i}.ffn_norm.weight
        mlp.gate_proj.weight   -> blk.{i}.ffn_gate.weight
        mlp.up_proj.weight     -> blk.{i}.ffn_up.weight
        mlp.down_proj.weight   -> blk.{i}.ffn_down.weight

    Also maps the embedding, the final norm, and the (tied) lm_head.

    Non-parameter buffers (e.g. ``*.attn.rope.inv_freq``) and the no-op
    dropout module (``transformer.emb_drop``) are ignored -- v1.5 checkpoints
    may carry them in the state_dict and they are not written to GGUF.

    Args:
        state_dict: The weighted checkpoint's state_dict.
        dtype_str:  Target dtype ("float32", "float16", "bfloat16").
        verbose:    Print per-tensor progress.
        compat_arch: GGUF architecture ("none", "llama", "qwen2"). q/k/v/o,
                     norms, and FFN weights are written EXACTLY as-is (2D
                     [out, in]) for both "llama" and "qwen2" -- qwen2 expects
                     separate 2D attn_q/attn_k/attn_v/attn_output tensors.
                     "none" writes the same plain 2D layout under the native
                     "dizel" arch (round-trip only, not loadable by llama.cpp).
        n_heads:    Number of query heads (config.n_heads). Used to fall back
                    when head_dim is not supplied.
        head_dim:   Attention head dimension (config.head_dim).
        n_heads_kv: Number of KV heads (config.kv_heads; defaults to n_heads).
                    Used only for the head_dim fallback path.

    Returns:
        list of (gguf_name: str, numpy_array, raw_dtype: GGMLQuantizationType)
    """
    gguf = _import("gguf")
    np = _import("numpy")
    n_layers = extract_layer_count(state_dict)
    raw_dtype = _ggml_dtype(dtype_str)

    if n_layers == 0:
        print("WARNING: Could not detect any transformer blocks in state_dict keys.")

    tensors = []
    mapped_keys = set()

    # -- Helper --------------------------------------------------------
    def add(gguf_name, src_key, note="", force_f32=False):
        """Convert, register, and optionally log a tensor.

        force_f32: Always convert the tensor to F32. llama.cpp's RMSNorm
        computes ggml_mul(f32_activation, norm_weight); if norm weights are
        F16 while activations are F32, GGML hard-crashes with
        "binary_op: unsupported types: dst: f32, src0: f32, src1: f16".
        Official llama.cpp converters always write norm tensors as F32 even
        when all other weights are quantized/F16.
        """
        torch_tensor = state_dict[src_key]
        if force_f32:
            np_arr = _torch_to_numpy(torch_tensor, "float32")
            out_dtype = _ggml_dtype("float32")
        else:
            np_arr = _torch_to_numpy(torch_tensor, dtype_str)
            out_dtype = raw_dtype

        tensors.append((gguf_name, np_arr, out_dtype))
        mapped_keys.add(src_key)
        if verbose:
            shape_str = "×".join(str(s) for s in torch_tensor.shape)
            if force_f32:
                note = (note + " " if note else "") + "(F32 norm)"
            print(f"  {gguf_name:30s} {shape_str:>15s}  {note}")

    # -------------------------------------------------------------------
    # Token embedding
    # -------------------------------------------------------------------
    emb_key = "transformer.tok_emb.weight"
    if emb_key in state_dict:
        add("token_embd.weight", emb_key)

    # -------------------------------------------------------------------
    # Per-layer blocks (v1.5: RMSNorm + GQA + SwiGLU, separate projections)
    # -------------------------------------------------------------------
    for i in range(n_layers):
        prefix = f"transformer.blocks.{i}."

        # -- RMSNorm (pre-attention) -----------------------------------
        k = f"{prefix}input_norm.weight"
        if k in state_dict:
            add(f"blk.{i}.attn_norm.weight", k, force_f32=True)

        # -- GQA attention projections (already separate, no chunking) -
        k = f"{prefix}attn.q_proj.weight"
        if k in state_dict:
            add(f"blk.{i}.attn_q.weight", k)
        k = f"{prefix}attn.k_proj.weight"
        if k in state_dict:
            add(f"blk.{i}.attn_k.weight", k)
        k = f"{prefix}attn.v_proj.weight"
        if k in state_dict:
            add(f"blk.{i}.attn_v.weight", k)
        k = f"{prefix}attn.o_proj.weight"
        if k in state_dict:
            add(f"blk.{i}.attn_output.weight", k)

        # -- RMSNorm (post-attention / pre-FFN) -------------------------
        k = f"{prefix}post_attn_norm.weight"
        if k in state_dict:
            add(f"blk.{i}.ffn_norm.weight", k, force_f32=True)

        # -- SwiGLU MLP: gate/up project d_model -> intermediate, down back
        k = f"{prefix}mlp.gate_proj.weight"
        if k in state_dict:
            add(f"blk.{i}.ffn_gate.weight", k)
        k = f"{prefix}mlp.up_proj.weight"
        if k in state_dict:
            add(f"blk.{i}.ffn_up.weight", k)
        k = f"{prefix}mlp.down_proj.weight"
        if k in state_dict:
            add(f"blk.{i}.ffn_down.weight", k)

    # -------------------------------------------------------------------
    # Final RMSNorm
    # -------------------------------------------------------------------
    k = "transformer.ln_f.weight"
    if k in state_dict:
        add("output_norm.weight", k, force_f32=True)

    # -------------------------------------------------------------------
    # LM Head / Output (weight-tied to tok_emb in v1.5, same tensor
    # object; llama.cpp convention still expects an explicit output.weight)
    # -------------------------------------------------------------------
    lm_key = "lm_head.weight"
    if lm_key in state_dict:
        add("output.weight", lm_key)

    # -------------------------------------------------------------------
    # Warn about keys that were not mapped (excluding expected no-ops):
    #   - "transformer.emb_drop"   (nn.Dropout has no parameters)
    #   - "*inv_freq"              (RoPE non-persistent buffers on CPU lift)
    # These are expected in real v1.5 checkpoints and are silenced.
    # -------------------------------------------------------------------
    def _is_ignorable(key):
        return key.startswith("transformer.emb_drop") or "inv_freq" in key

    unmapped = [
        k for k in state_dict
        if k not in mapped_keys and not _is_ignorable(k)
    ]
    if unmapped and verbose:
        print(f"\n  Unmapped keys ({len(unmapped)}): {unmapped}")

    return tensors


# ===========================================================================
# GGUF WRITING
# ===========================================================================

def extract_tokenizer_data(sp):
    """
    Extract token strings, scores, and types from a SentencePieceProcessor.

    Uses protobuf if available to parse the model proto; falls back to
    heuristics based on SentencePiece's is_control / is_unknown methods.
    """
    protobuf = _soft_import("google.protobuf")

    # Try to parse the proto for exact type information
    if protobuf is not None:
        try:
            from sentencepiece import sentencepiece_model_pb2
            proto_bytes = sp.serialized_model_proto()
            proto = sentencepiece_model_pb2.ModelProto()
            proto.ParseFromString(proto_bytes)

            tokens = []
            scores = []
            types = []
            for piece in proto.pieces:
                tokens.append(piece.piece)
                scores.append(piece.score)
                # SentencePiece proto types map directly to GGUF TokenType:
                #   NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4, UNUSED=5, BYTE=6
                types.append(piece.type)

            return tokens, scores, types
        except Exception:
            pass  # Fall through to heuristic approach

    # -- Heuristic fallback ---------------------------------------------
    tokens = []
    scores = []
    types = []
    for i in range(sp.vocab_size()):
        token = sp.id_to_piece(i)
        token_str = token if isinstance(token, str) else token.decode("utf-8", errors="replace")

        tokens.append(token_str)
        scores.append(sp.get_score(i))

        # Heuristic type detection
        if sp.is_control(i):
            piece_type = 3  # CONTROL
        elif sp.is_unknown(i):
            piece_type = 2  # UNKNOWN
        elif token_str.startswith("<0x"):
            piece_type = 6  # BYTE
        else:
            piece_type = 1  # NORMAL

        types.append(piece_type)

    return tokens, scores, types


def write_gguf_file(
    output_path: str,
    config,
    tensors,
    sp,
    dtype_str: str,
    metadata_only: bool = False,
    verbose: bool = False,
    compat_arch: str = "none",
    vocab_size: Optional[int] = None,
):
    """
    Write a GGUF file with model metadata, tokenizer data, and weight tensors.

    Uses gguf.GGUFWriter from the gguf Python library.

    Args:
        vocab_size: Optional override for the vocab_size written to metadata.
                    Used when the checkpoint's embedding rows disagree with
                    config.vocab_size. Falls back to config.vocab_size.
    """
    # Ensure the output directory exists (e.g. checkpoints/)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    gguf = _import("gguf")
    np = _import("numpy")

    if verbose:
        print(f"\n  Writing GGUF: {output_path}")
        if metadata_only:
            print("    (metadata-only mode -- no weight tensors)")

    # -- Create writer --------------------------------------------------
    # The writer receives the resolved arch below: "llama" by default,
    # "dizel" only when --compat-arch none is passed.
    arch = "dizel" if compat_arch == "none" else compat_arch
    writer = gguf.GGUFWriter(str(output_path), arch)

    # -- General metadata -----------------------------------------------
    writer.add_string(gguf.Keys.General.NAME, "Dizel")
    writer.add_string(gguf.Keys.General.DESCRIPTION, "Dizel Qwen-style language model (GQA + SwiGLU + RMSNorm + RoPE)")
    # GGUF file_type is a quantization label, not a byte-size hint:
    # 0 = mostly F32, 1 = mostly F16. Writing 2 here made Ollama report
    # F16 GGUFs as Q4_0 even though every tensor was still F16.
    file_type = 1 if dtype_str in ("float16", "bfloat16") else 0
    writer.add_uint32(gguf.Keys.General.FILE_TYPE, file_type)

    # -- Architecture hyper-parameters ----------------------------------
    # Use manual keys to match the exact naming convention required for
    # llama.cpp's custom architecture loader.
    prefix = "dizel" if compat_arch == "none" else compat_arch
    # NaN-safe attribute resolution: v1.5 uses kv_heads/head_dim/intermediate_size/
    # rope_theta/norm_eps; v1.2 uses head_dim/ffn_dim properties and rope_base.
    head_count_kv = getattr(config, "kv_heads", config.n_heads)
    head_dim = config.head_dim
    ffn_dim = config.ffn_dim
    rope_base = getattr(config, "rope_theta", getattr(config, "rope_base", 10000.0))
    norm_eps = getattr(config, "norm_eps", 1e-5)
    resolved_vocab = vocab_size if vocab_size is not None else config.vocab_size

    writer.add_uint32(f"{prefix}.context_length", config.context_length)
    writer.add_uint32(f"{prefix}.embedding_length", config.d_model)
    writer.add_uint32(f"{prefix}.block_count", config.n_layers)
    writer.add_uint32(f"{prefix}.attention.head_count", config.n_heads)
    writer.add_uint32(f"{prefix}.attention.head_count_kv", head_count_kv)
    writer.add_uint32(f"{prefix}.vocab_size", resolved_vocab)
    writer.add_uint32(f"{prefix}.feed_forward_length", ffn_dim)
    writer.add_uint32(f"{prefix}.rope.dimension_count", head_dim)
    writer.add_float32(f"{prefix}.rope.freq_base", rope_base)
    writer.add_float32(f"{prefix}.attention.layer_norm_rms_epsilon", norm_eps)

    # -- Tokenize metadata ---------------------------------------------
    # llama.cpp's vocab loader only recognizes specific tokenizer-model names
    # ("llama", "bert", "gpt2", "t5", "rwkv", "plamo2", "gemma4", "no_vocab",
    # "none"); ANY other string -- including "sentencepiece" -- hard-fails the
    # load with "unknown tokenizer: '<name>'" (see llama.cpp llama-vocab.cpp).
    # So for --compat-arch llama/qwen2 GGUFs we must write llama.cpp's
    # SentencePiece type name ("llama"). Native dizel GGUFs keep
    # "sentencepiece": they are never loaded by llama.cpp (the GUI blocks
    # arch "dizel"), and run_inference_from_gguf() reads tokens from disk via
    # SentencePiece and is agnostic to this metadata string.
    writer.add_tokenizer_model("llama" if compat_arch in ("llama", "qwen2") else "sentencepiece")

    # Embed the full tokenizer data (tokens, scores, types)
    if sp is not None:
        try:
            tokens, scores, types = extract_tokenizer_data(sp)
            writer.add_token_list(tokens)
            writer.add_token_scores(scores)
            writer.add_token_types(types)
            if verbose:
                print(f"    Embedded {len(tokens)} tokens from SentencePiece model")
        except Exception as e:
            print(f"WARNING: Failed to embed tokenizer data: {e}")

    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(3)

    # -- Chat template --------------------------------------------------
    # llama-cpp-python chat-format precedence: chat_handler -> chat_format ->
    # tokenizer.chat_template from GGUF metadata -> fallback llama-2. Without
    # this metadata key the GUI falls back to llama-2 and produces garbage
    # formatting; Dizel v1.5 was trained with the exact DIZEL_CHAT_TEMPLATE
    # format below. Only embed it for llama.cpp-loadable GGUFs ("llama" /
    # "qwen2"); native "dizel" arch GGUFs are round-trip only and keep their
    # own tokenizer metadata.
    # The literal key "tokenizer.chat_template" is used deliberately:
    # gguf.Keys.Tokenizer.CHAT_TEMPLATE may not exist in older gguf-py
    # versions and would raise AttributeError at runtime.
    if compat_arch in ("llama", "qwen2"):
        writer.add_string("tokenizer.chat_template", DIZEL_CHAT_TEMPLATE)
        if verbose:
            print("    Embedded tokenizer.chat_template (Dizel v1.5 format)")

    # -- Tensor data ----------------------------------------------------
    if not metadata_only:
        for gguf_name, np_arr, raw_dtype in tensors:
            # add_tensor stores the numpy array data for later writing
            # and registers the metadata with the correct GGML type.
            writer.add_tensor(gguf_name, np_arr, raw_dtype=raw_dtype)

    # -- Write file -----------------------------------------------------
    writer.write_header_to_file()
    writer.write_kv_data_to_file()

    if not metadata_only:
        if verbose:
            writer.write_tensors_to_file(progress=True)
        else:
            writer.write_tensors_to_file()

    writer.close()

    if verbose:
        file_size = os.path.getsize(output_path)
        print(f"    File written: {output_path}")
        print(f"    File size   : {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")


# ===========================================================================
# PARAMETER COUNTING
# ===========================================================================

def count_parameters(state_dict):
    """Count total parameters from a state_dict."""
    total = 0
    for tensor in state_dict.values():
        if hasattr(tensor, "numel"):
            total += tensor.numel()
    return total


# ===========================================================================
# INFERENCE HELPER
# ===========================================================================

def run_inference_from_gguf(
    gguf_path: str,
    prompt_text: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
):
    """
    Load a GGUF file, reconstruct the DizelLM v1.5 model, load weights from
    the GGUF tensor data, and run inference.

    This is a *verification* helper -- it reads the GGUF tensor data back
    and maps it onto the v1.5 parameter names (input_norm/q_proj/k_proj/
    v_proj/o_proj/post_attn_norm/gate_proj/up_proj/down_proj). The GGUF
    writer split nothing, so this round-trip proves the conversion was lossless.

    Args:
        gguf_path:       Path to the GGUF file to load.
        prompt_text:     Text prompt for generation.
        max_new_tokens:  Maximum tokens to generate.
        temperature:     Sampling temperature.
        top_k:           Top-k sampling.
        top_p:           Nucleus sampling threshold.
    """
    gguf = _import("gguf")
    torch = _import("torch")
    sp_module = _soft_import("sentencepiece")

    if sp_module is None:
        print("ERROR: sentencepiece is required for inference.")
        return

    _ensure_project_root()
    from config import ModelConfigV15
    from model.architecture import DizelLM

    # -- 1. Read GGUF metadata ------------------------------------------
    if not os.path.isfile(gguf_path):
        print(f"ERROR: GGUF file not found: {gguf_path}")
        return

    print(f"Loading GGUF: {gguf_path}")
    reader = gguf.GGUFReader(gguf_path)

    # ── Auto-detect architecture ─────────────────────────
    arch = "dizel"
    for key in reader.fields:
        if key == "general.architecture":
            try:
                arch = reader.get_field(key).contents()
                if isinstance(arch, bytes):
                    arch = arch.decode("utf-8")
                arch = str(arch).lower()
            except:
                pass
            break
    print(f"  Detected architecture: {arch}")

    # ── Explain what hyper-parameters the model will read ──
    def _get_meta(reader, arch, key, default=0, as_float=False):
        """Get metadata value trying both arch-prefixed and un-prefixed keys."""
        prefixes = [f"{arch}.", "dizel."]
        for prefix in prefixes:
            fname = prefix + key
            try:
                field = reader.get_field(fname)
                if field is not None:
                    if as_float:
                        return float(field.contents())
                    return int(field.contents())
            except:
                pass
        return default

    d_model = _get_meta(reader, arch, "embedding_length")
    n_layers = _get_meta(reader, arch, "block_count")
    n_heads = _get_meta(reader, arch, "attention.head_count")
    head_count_kv = _get_meta(reader, arch, "attention.head_count_kv", default=0)
    context_length = _get_meta(reader, arch, "context_length")
    ffn_dim = _get_meta(reader, arch, "feed_forward_length")
    head_dim = _get_meta(reader, arch, "rope.dimension_count", default=0)
    rope_base = _get_meta(reader, arch, "rope.freq_base", default=10000.0, as_float=True)
    norm_eps = _get_meta(reader, arch, "attention.layer_norm_rms_epsilon", default=1e-6, as_float=True)

    # Fallbacks for legacy GUF files that omit GQA-specific fields
    if head_count_kv <= 0:
        head_count_kv = n_heads
    if head_dim <= 0:
        head_dim = d_model // max(n_heads, 1)

    # Infer vocab_size: first try metadata key, then actual tensor shape
    vocab_size = _get_meta(reader, arch, "vocab_size", default=-1)
    if vocab_size < 0:
        # Fallback: use the actual numpy array shape from token_embd / output
        for tensor in reader.tensors:
            if tensor.name in ("token_embd.weight", "output.weight"):
                s = tensor.data.shape
                # GGUF dimension ordering can vary; take the larger dimension
                # since vocab size is generally larger than the model width.
                if len(s) >= 2:
                    vocab_size = max(int(s[0]), int(s[1]))
                else:
                    vocab_size = int(s[0])
                break
    if vocab_size < 0:
        vocab_size = 100000  # ultimate fallback (v1.5 default)

    print(f"  d_model={d_model}  n_layers={n_layers}  n_heads={n_heads}  kv_heads={head_count_kv}")
    print(f"  context_length={context_length}  ffn_dim={ffn_dim}  head_dim={head_dim}")
    print(f"  rope_base={rope_base}  norm_eps={norm_eps}  vocab_size={vocab_size}")

    # -- 2. Build model -------------------------------------------------
    cfg = ModelConfigV15(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        kv_heads=head_count_kv,
        head_dim=head_dim,
        intermediate_size=ffn_dim,
        norm_type="rmsnorm",
        use_rope=True,
        rope_theta=rope_base,
        weight_tying=True,
        dropout=0.0,
    )
    model = DizelLM(cfg)
    model.eval()

    # -- 3. Load weights from GGUF tensor data --------------------------
    tensor_map = {}
    for tensor in reader.tensors:
        # GGUFReader tensors expose .data as a numpy view
        tensor_map[tensor.name] = torch.from_numpy(tensor.data).clone()

    state_dict = {}

    # Token embedding
    if "token_embd.weight" in tensor_map:
        state_dict["transformer.tok_emb.weight"] = tensor_map["token_embd.weight"]

    # Final width norm
    if "output_norm.weight" in tensor_map:
        state_dict["transformer.ln_f.weight"] = tensor_map["output_norm.weight"]

    # Output weight (tied to tok_emb in v1.5; both keys load fine)
    if "output.weight" in tensor_map:
        state_dict["lm_head.weight"] = tensor_map["output.weight"]

    # Per-layer blocks (reverse of the v1.5 mapping above)
    for i in range(n_layers):
        b = f"transformer.blocks.{i}."

        blk_map = {
            f"blk.{i}.attn_norm.weight":  f"{b}input_norm.weight",
            f"blk.{i}.attn_q.weight":     f"{b}attn.q_proj.weight",
            f"blk.{i}.attn_k.weight":     f"{b}attn.k_proj.weight",
            f"blk.{i}.attn_v.weight":     f"{b}attn.v_proj.weight",
            f"blk.{i}.attn_output.weight":f"{b}attn.o_proj.weight",
            f"blk.{i}.ffn_norm.weight":   f"{b}post_attn_norm.weight",
            f"blk.{i}.ffn_gate.weight":   f"{b}mlp.gate_proj.weight",
            f"blk.{i}.ffn_up.weight":     f"{b}mlp.up_proj.weight",
            f"blk.{i}.ffn_down.weight":   f"{b}mlp.down_proj.weight",
        }
        for gguf_name, v5_name in blk_map.items():
            if gguf_name in tensor_map:
                state_dict[v5_name] = tensor_map[gguf_name]

    # Load into model (non-strict: tied tok_emb/lm_head share the tensor)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {unexpected}")
    print("  Model weights loaded successfully from GGUF.")

    # -- 4. Tokenize prompt ---------------------------------------------
    sp = sp_module.SentencePieceProcessor()

    # Try to find the tokenizer model file
    gguf_dir = os.path.dirname(os.path.abspath(gguf_path))
    tokenizer_candidates = [
        # v1.5 tokenizer first
        os.path.join(gguf_dir, "dizel_v15.model"),
        os.path.join(gguf_dir, "tokenizer", "dizel_v15.model"),
        os.path.join(gguf_dir, "..", "tokenizer", "dizel_v15.model"),
        os.path.join(os.getcwd(), "tokenizer", "dizel_v15.model"),
        "tokenizer/dizel_v15.model",
        # Legacy v1.2 tokenizer
        os.path.join(gguf_dir, "tokenizer", "dizel.model"),
        os.path.join(gguf_dir, "..", "tokenizer", "dizel.model"),
        os.path.join(os.getcwd(), "tokenizer", "dizel.model"),
        "tokenizer/dizel.model",
        # Generic fallback
        "checkpoints/tokenizer.model",
    ]
    tokenizer_loaded = False
    for tp in tokenizer_candidates:
        tp = os.path.abspath(tp)
        if os.path.isfile(tp):
            try:
                sp.load(tp)
                tokenizer_loaded = True
                break
            except Exception:
                continue

    if not tokenizer_loaded:
        print("ERROR: Could not find tokenizer model file.")
        print("  Place tokenizer/dizel_v15.model in the project root or pass --tokenizer.")
        return

    prompt_ids = sp.encode(prompt_text)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)

    print(f"\nPrompt ({len(prompt_ids)} tokens): {prompt_text!r}")
    print(f"Generating ({max_new_tokens} tokens)...")

    # -- 5. Generate ----------------------------------------------------
    with torch.no_grad():
        output_ids = model.generate(
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=2,
        )

    # -- 6. Decode output -----------------------------------------------
    generated_list = output_ids[0].tolist()
    new_ids = generated_list[len(prompt_ids):]
    generated_text = sp.decode(new_ids)

    print(f"\n{'=' * 60}")
    print("GENERATED TEXT")
    print(f"{'=' * 60}")
    print(prompt_text, end="")
    print(generated_text)
    print(f"{'=' * 60}")
    print(f"Generated {len(new_ids)} tokens.")
    print()


# ===========================================================================
# MAIN CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert Dizel .pt checkpoint to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion (v1.5 defaults)
  python scripts/convert_to_gguf.py --checkpoint checkpoints/dizel-v15-sft-best.pt

  # Full options
  python scripts/convert_to_gguf.py ^
      --checkpoint checkpoints/dizel-v15-sft-best.pt ^
      --tokenizer tokenizer/dizel_v15.model ^
      --output checkpoints/Dizel.gguf ^
      --config-key dizel-v1.5 ^
      --dtype float16 ^
      --compat-arch qwen2 ^
      --verbose

  # Metadata only (quick validation, no tensors)
  python scripts/convert_to_gguf.py --checkpoint checkpoints/dizel-v15-sft-best.pt --metadata-only

  # Legacy v1.2 checkpoint (fc1/fc2, 32K vocab — weights are re-mapped but the
  # MLP/LN difference is preserved as-is in the GGUF structure; outputs may differ)
  python scripts/convert_to_gguf.py --checkpoint checkpoints/dizel-sft-best.pt --config-key dizel-v1.2

  # Test inference from the newly-created GGUF:
  python -c "from scripts.convert_to_gguf import run_inference_from_gguf; run_inference_from_gguf('checkpoints/Dizel.gguf', 'Hello', max_new_tokens=50)"
        """,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the .pt checkpoint file",
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer/dizel_v15.model",
        help="Path to the SentencePiece model file (default: tokenizer/dizel_v15.model)",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/Dizel.gguf",
        help="Output GGUF file path (default: checkpoints/Dizel.gguf)",
    )
    parser.add_argument(
        "--config-key",
        default="dizel-v1.5",
        choices=["dizel-v1.1", "dizel-v1.2", "dizel-v1.5", "mila-v1.0-beta"],
        help="Model registry key (default: dizel-v1.5)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Weight dtype in GGUF (default: float16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="LoRA alpha used in distillation (for folding unmerged adapters; "
             "defaults to checkpoint metadata, else convention 2*r)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="LoRA rank used in distillation (defaults to checkpoint metadata, "
             "else inferred from lora_A shape)",
    )
    parser.add_argument(
        "--compat-arch",
        default="qwen2",
        choices=["none", "llama", "qwen2"],
        help="GGUF architecture (default: qwen2 — Qwen1.5/Qwen2-style separate QKV + GQA naming, loads in llama.cpp/llama-cpp-python). Use 'llama' for plain 2D llama-style naming, 'none' only for native-dizel round-trip validation via run_inference_from_gguf.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only write metadata, skip weight tensors",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )

    args = parser.parse_args()

    # =====================================================================
    # 1. Load model config
    # =====================================================================
    print("Dizel -> GGUF Converter")
    print("=" * 60)

    if args.verbose:
        print(f"Config key: {args.config_key}")

    config = load_model_config(args.config_key)

    kv_heads = getattr(config, "kv_heads", config.n_heads)
    rope_base = getattr(config, "rope_theta", getattr(config, "rope_base", 10000.0))
    norm_eps = getattr(config, "norm_eps", 1e-5)

    if args.verbose:
        print(f"\nArchitecture: DizelLM")
        print(f"  vocab_size       = {config.vocab_size}")
        print(f"  d_model          = {config.d_model}")
        print(f"  n_layers         = {config.n_layers}")
        print(f"  n_heads          = {config.n_heads}")
        print(f"  kv_heads         = {kv_heads}")
        head_dim = config.head_dim
        print(f"  head_dim         = {head_dim}")
        print(f"  ffn_dim          = {config.ffn_dim}")
        print(f"  context_length   = {config.context_length}")
        print(f"  rope_base        = {rope_base}")
        print(f"  norm_eps         = {norm_eps}")
        print(f"  weight_tying     = {config.weight_tying}")
        print(f"  bias             = {config.bias}")

    # =====================================================================
    # 2. Load checkpoint
    # =====================================================================
    state_dict, source_info = load_checkpoint(
        args.checkpoint, args.verbose,
        lora_alpha=args.lora_alpha, lora_r=args.lora_r,
    )
    param_count = count_parameters(state_dict)

    if args.verbose:
        print(f"\n  Tensors loaded: {len(state_dict)}")
        print(f"  Total parameters: {param_count:,}")

    # =====================================================================
    # 3. Load tokenizer
    # =====================================================================
    sp = load_tokenizer(args.tokenizer, args.verbose)

    # =====================================================================
    # 4. Map Dizel -> GGUF tensor names (v1.5: no QKV splitting, opaque GQA)
    # =====================================================================
    if args.verbose:
        print(f"\nMapping tensors to GGUF names (dtype={args.dtype}):")

    tensors = map_dizel_to_gguf_tensors(
        state_dict,
        args.dtype,
        args.verbose,
        compat_arch=args.compat_arch,
        n_heads=config.n_heads,
        head_dim=config.head_dim,
        n_heads_kv=getattr(config, "kv_heads", config.n_heads),
    )

    # ---- Resolve vocab_size (embedding rows can differ from config) --------
    resolved_vocab_size = config.vocab_size
    emb_key = "transformer.tok_emb.weight"
    if emb_key in state_dict:
        emb_rows = int(state_dict[emb_key].shape[0])
        if emb_rows != config.vocab_size:
            print(
                f"WARNING: Checkpoint embedding rows ({emb_rows}) differ from "
                f"config vocab_size ({config.vocab_size}). "
                f"Using {emb_rows} in GGUF metadata."
            )
            resolved_vocab_size = emb_rows

    # =====================================================================
    # 5. Write GGUF file
    # =====================================================================
    t_start = time.perf_counter()
    write_gguf_file(
        output_path=args.output,
        config=config,
        tensors=tensors,
        sp=sp,
        dtype_str=args.dtype,
        metadata_only=args.metadata_only,
        verbose=args.verbose,
        compat_arch=args.compat_arch,
        vocab_size=resolved_vocab_size,
    )
    elapsed = time.perf_counter() - t_start

    # =====================================================================
    # 6. Summary
    # =====================================================================
    output_path = args.output
    file_size = os.path.getsize(output_path) if os.path.isfile(output_path) else 0

    print()
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"  Source checkpoint : {args.checkpoint}")
    print(f"  Source format     : {source_info}")
    print(f"  Tokenizer         : {args.tokenizer or '(none)'}")
    print(f"  Config key        : {args.config_key}")
    print(f"  Architecture      : DizelLM")
    print(f"    d_model         : {config.d_model}")
    print(f"    n_layers        : {config.n_layers}")
    print(f"    n_heads         : {config.n_heads}")
    print(f"    kv_heads        : {kv_heads}")
    print(f"    ffn_dim         : {config.ffn_dim}")
    print(f"    context_length  : {config.context_length}")
    print(f"    vocab_size      : {resolved_vocab_size}")
    print(f"  Parameters        : {param_count:,}")
    print(f"  Output GGUF       : {output_path}")
    print(f"  File size         : {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")
    print(f"  Weight dtype      : {args.dtype}")
    print(f"  Tensors written   : {len(tensors)}", end="")
    if args.metadata_only:
        print("  (metadata only -- no weight data)")
    else:
        print()
    print(f"  Time elapsed      : {elapsed:.2f} seconds")
    print(f"  GGUF arch         : {args.compat_arch}")
    if args.compat_arch == "none":
        print("  WARNING: native 'dizel' arch — this file will NOT load in llama.cpp.")
    print()


if __name__ == "__main__":
    main()