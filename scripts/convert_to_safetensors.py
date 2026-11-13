#!/usr/bin/env python3
"""
scripts/convert_to_safetensors.py -- Convert a Dizel .pt checkpoint to a
HuggingFace ``transformers``-loadable directory with ``model.safetensors``.

Dizel is a CUSTOM architecture (GQA + SwiGLU + RMSNorm + RoPE, class
``DizelLM`` in ``model/architecture.py``), so a bare weights file is NOT
enough -- this script also ships the remote-code modeling files
(``configuration_dizel.py`` / ``modeling_dizel.py``) into the output
directory. The result can be loaded with:

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("hf_export/", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("hf_export/")

(``--trust-remote-code`` is a *loader* flag, not a converter flag.)

Usage:
    python scripts/convert_to_safetensors.py \\
        --checkpoint checkpoints/dizel-v15-sft-best.pt \\
        --output hf_export/ \\
        --dtype float32 \\
        --verbose

Tokenizers:
    The SentencePiece model is copied in as ``tokenizer.model``; the special
    tokens are READ from the .model file at runtime (``id_to_piece`` for
    ids 0..3) and written into ``tokenizer_config.json`` /
    ``special_tokens_map.json``.  LlamaTokenizer loads a SentencePiece
    ``.model`` directly -- no ``tokenizer.json`` is built.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# ===========================================================================
# SAFE IMPORTS WITH GRACEFUL FALLBACKS
# ===========================================================================

_REQUIREMENTS_HELP = """
Required packages:
    pip install safetensors torch sentencepiece
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


def _ensure_project_root():
    """
    Add the project root to sys.path so ``config``/``model`` imports work
    whether the script is run from the project root or from scripts/.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.dirname(script_dir),       # parent of scripts/
        os.getcwd(),                       # current working directory
    ]
    for root in candidates:
        marker = os.path.join(root, "config.py")
        if os.path.isfile(marker) and root not in sys.path:
            sys.path.insert(0, root)


# ===========================================================================
# CHECKPOINT LOADING  (mirrored from scripts/convert_to_gguf.py::load_checkpoint)
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

    Returns: (cleaned_state_dict, source_info, ckpt_dict)
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

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

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

    return cleaned, source_info, ckpt


# ===========================================================================
# HYPERPARAMETER RESOLUTION (embedded model_cfg -> fallback ModelConfigV15)
# ===========================================================================

# v1.5 defaults mirroring ModelConfigV15 exactly
_V15_DEFAULTS = {
    "vocab_size": 100000,
    "context_length": 2048,
    "d_model": 1024,
    "n_layers": 28,
    "n_heads": 16,
    "kv_heads": 8,
    "head_dim": 64,
    "intermediate_size": 3072,
    "norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "bias": False,
    "weight_tying": True,
    "dropout": 0.0,
    "initializer_range": 0.02,
}


def load_hyperparams(model_cfg, state_dict, verbose: bool = False) -> dict:
    """
    Resolve the v1.5 hyper-parameters.

    If the checkpoint embeds a ``model_cfg`` object (e.g. a pickled
    ``ModelConfigV15`` dataclass), read the attributes off it.  Otherwise fall
    back to ``from config import ModelConfigV15; ModelConfigV15()``.
    """
    hp = dict(_V15_DEFAULTS)

    cfg_source = None
    if model_cfg is not None:
        cfg_source = model_cfg
    else:
        try:
            _ensure_project_root()
            from config import ModelConfigV15
            cfg_source = ModelConfigV15()
        except Exception as e:
            if verbose:
                print(f"  WARNING: could not import ModelConfigV15 ({e}); "
                      f"using baked-in v1.5 defaults.")

    if cfg_source is not None:
        for k in _V15_DEFAULTS:
            if hasattr(cfg_source, k):
                hp[k] = getattr(cfg_source, k)
        if verbose:
            print(f"  Hyper-parameters from {type(cfg_source).__name__}")

    # -- vocab_size sanity-check vs. actual embedding rows ------------------
    emb_key = "transformer.tok_emb.weight"
    if emb_key in state_dict:
        rows = int(state_dict[emb_key].shape[0])
        if rows != hp["vocab_size"]:
            print(
                f"WARNING: checkpoint embedding rows ({rows}) differ from "
                f"model_cfg vocab_size ({hp['vocab_size']}). "
                f"Using {rows} in config.json."
            )
            hp["vocab_size"] = rows

    return hp


# ===========================================================================
# DTYPE HELPERS
# ===========================================================================

def _torch_dtype(dtype_str: str):
    torch = _import("torch")
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(dtype_str, torch.float32)


def _dtype_name(dtype_str: str) -> str:
    """The exact string we want in config.json's ``torch_dtype``."""
    return dtype_str


# ===========================================================================
# TOKENIZER EXPORT  (subjectpieces-model derived, NOT assumed)
# ===========================================================================

def load_tokenizer(tokenizer_path: str, verbose: bool = False):
    """Load a SentencePiece processor. Returns None on failure."""
    sp_module = _soft_import("sentencepiece")
    if sp_module is None:
        print("WARNING: sentencepiece not installed; tokenizer export skipped.")
        return None
    if not os.path.isfile(tokenizer_path):
        print(f"WARNING: Tokenizer not found: {tokenizer_path}")
        return None
    if verbose:
        print(f"  Loading SentencePiece model: {tokenizer_path}")
    sp = sp_module.SentencePieceProcessor()
    sp.load(tokenizer_path)
    return sp


def resolve_special_tokens(sp, verbose: bool = False) -> dict:
    """
    Derive the 4 special tokens from the SentencePiece model at runtime.

    TokenizerConfigV15 declares pad=0, bos=1, eos=2, unk=3 -- but we READ the
    actual pieces from the model (``id_to_piece``) instead of assuming names.
    """
    if sp is None:
        print("WARNING: no SentencePiece model available; falling back to "
              "v1.5 defaults (<pad>/<s></s>/<unk> strings).")
        return {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }

    def _piece(i: int) -> str:
        piece = sp.id_to_piece(i)
        if isinstance(piece, bytes):
            piece = piece.decode("utf-8", errors="replace")
        return piece

    pieces = {i: _piece(i) for i in range(4)}  # 0=pad 1=bos 2=eos 3=unk
    if verbose:
        print(f"    id 0 (pad) -> {pieces[0]!r}")
        print(f"    id 1 (bos) -> {pieces[1]!r}")
        print(f"    id 2 (eos) -> {pieces[2]!r}")
        print(f"    id 3 (unk) -> {pieces[3]!r}")

    return {
        "bos_token": pieces[1],
        "eos_token": pieces[2],
        "unk_token": pieces[3],
        "pad_token": pieces[0],
    }


def write_tokenizer_files(out_dir: str, sp, tokenizer_path: str,
                          context_length: int, verbose: bool = False):
    """Copy the SentencePiece model + write tokenizer_config.json &
    special_tokens_map.json (LlamaTokenizer-compatible)."""
    special = resolve_special_tokens(sp, verbose=verbose)

    # -- tokenizer.model ----------------------------------------------------
    src = tokenizer_path
    dst = os.path.join(out_dir, "tokenizer.model")
    if os.path.isfile(src):
        shutil.copyfile(src, dst)
        if verbose:
            print(f"  Copied {src} -> {dst}")
    elif sp is not None:
        # Fall back to serializing the loaded processor's proto directly.
        from sentencepiece import sentencepiece_model_pb2
        proto = sentencepiece_model_pb2.ModelProto()
        proto.ParseFromString(sp.serialized_model_proto())
        with open(dst, "wb") as f:
            f.write(proto.SerializeToString())
        if verbose:
            print(f"  Serialized loaded processor -> {dst}")
    else:
        raise FileNotFoundError(
            f"Tokenizer model not found at {src} and no processor available."
        )

    # -- tokenizer_config.json (LlamaTokenizer reads SentencePiece .model) --
    tokenizer_config = {
        "tokenizer_class": "LlamaTokenizer",
        "bos_token": special["bos_token"],
        "eos_token": special["eos_token"],
        "unk_token": special["unk_token"],
        "pad_token": special["pad_token"],
        "model_max_length": int(context_length),
        "bos_token_id": 1,
        "eos_token_id": 2,
        "add_bos_token": True,
        "add_eos_token": False,
        "cleanup_tokenization_spaces": True,
        "legacy": True,
        "use_fast": False,
    }
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # -- special_tokens_map.json -------------------------------------------------
    special_map = {
        "bos_token": special["bos_token"],
        "eos_token": special["eos_token"],
        "unk_token": special["unk_token"],
        "pad_token": special["pad_token"],
    }
    with open(os.path.join(out_dir, "special_tokens_map.json"), "w",
              encoding="utf-8") as f:
        json.dump(special_map, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if verbose:
        print(f"  Wrote tokenizer_config.json + special_tokens_map.json")


# ===========================================================================
# CONFIG.JSON  (HF format for a remote-code model)
# ===========================================================================

def build_hf_config_dict(hp: dict, dtype_str: str, source_info: str) -> dict:
    """The HF ``config.json`` payload (auto_map + all v1.5 hyperparams)."""
    cfg = {
        # --- remote-code metadata ----------------------------------------
        "model_type": "dizel",
        "architectures": ["DizelLMForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_dizel.DizelConfig",
            "AutoModelForCausalLM": "modeling_dizel.DizelLMForCausalLM",
        },
        "torch_dtype": _dtype_name(dtype_str),
        # --- v1.5 hyper-parameters -------------------------------------
        "vocab_size": int(hp["vocab_size"]),
        "context_length": int(hp["context_length"]),
        "max_position_embeddings": int(hp["context_length"]),
        "d_model": int(hp["d_model"]),
        "n_layers": int(hp["n_layers"]),
        "n_heads": int(hp["n_heads"]),
        "kv_heads": int(hp["kv_heads"]),
        "head_dim": int(hp["head_dim"]),
        "intermediate_size": int(hp["intermediate_size"]),
        "norm_eps": float(hp["norm_eps"]),
        "rope_theta": float(hp["rope_theta"]),
        "bias": bool(hp["bias"]),
        "weight_tying": bool(hp["weight_tying"]),
        "tie_word_embeddings": bool(hp["weight_tying"]),
        "dropout": float(hp["dropout"]),
        "initializer_range": float(hp.get("initializer_range", 0.02)),
    }
    return cfg


# ===========================================================================
# REMOTE CODE FILES
# ===========================================================================

def copy_remote_code(script_dir: str, out_dir: str, verbose: bool = False):
    """Copy configuration_dizel.py + modeling_dizel.py into the output dir."""
    for name in ("configuration_dizel.py", "modeling_dizel.py"):
        src = os.path.join(script_dir, name)
        if not os.path.isfile(src):
            print(f"ERROR: remote-code file not found next to this script: {src}")
            sys.exit(1)
        shutil.copyfile(src, os.path.join(out_dir, name))
        if verbose:
            print(f"  Copied {src} -> {os.path.join(out_dir, name)}")


# ===========================================================================
# MAIN CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert Dizel .pt checkpoint to a HuggingFace safetensors export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python scripts/convert_to_safetensors.py --checkpoint checkpoints/dizel-v15-sft-best.pt

  # Full options
  python scripts/convert_to_safetensors.py ^
      --checkpoint checkpoints/dizel-v15-sft-best.pt ^
      --output hf_export/ ^
      --dtype float16 ^
      --tokenizer tokenizer/dizel_v15.model ^
      --verbose

  # Load the result (trust_remote_code is a LOADER flag, not a converter flag):
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained("hf_export/", trust_remote_code=True)
  tokenizer = AutoTokenizer.from_pretrained("hf_export/")
        """,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the .pt checkpoint file",
    )
    parser.add_argument(
        "--output",
        default="hf_export/",
        help="Output directory (default: hf_export/)",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Weight dtype in model.safetensors (default: float32)",
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
        "--tokenizer",
        default="tokenizer/dizel_v15.model",
        help="Path to the SentencePiece model (default: tokenizer/dizel_v15.model)",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only write config.json + tokenizer + remote code, skip tensors",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )

    args = parser.parse_args()

    torch = _import("torch")

    print("Dizel -> HF safetensors Converter")
    print("=" * 60)

    # =====================================================================
    # 1. Load checkpoint
    # =====================================================================
    state_dict, source_info, ckpt = load_checkpoint(
        args.checkpoint, args.verbose,
        lora_alpha=args.lora_alpha, lora_r=args.lora_r,
    )

    if args.verbose:
        print(f"  Tensors loaded: {len(state_dict)}")
        print(f"  Format        : {source_info}")

    # =====================================================================
    # 2. Resolve hyper-parameters
    # =====================================================================
    model_cfg = ckpt.get("model_cfg") if isinstance(ckpt, dict) else None
    hp = load_hyperparams(model_cfg, state_dict, verbose=args.verbose)

    if args.verbose:
        print("\n  Resolved hyper-parameters:")
        for k in ("vocab_size", "context_length", "d_model", "n_layers",
                  "n_heads", "kv_heads", "head_dim", "intermediate_size",
                  "norm_eps", "rope_theta", "bias", "weight_tying", "dropout"):
            print(f"    {k:<20s}= {hp[k]}")

    # =====================================================================
    # 3. Make output dir + config.json
    # =====================================================================
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    hf_config = build_hf_config_dict(hp, args.dtype, source_info)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    if args.verbose:
        print(f"\n  Wrote config.json")

    # =====================================================================
    # 4. Copy remote-code modeling files (self-contained, from this dir)
    # =====================================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    copy_remote_code(script_dir, out_dir, verbose=args.verbose)

    # =====================================================================
    # 5. Tokenizer export
    # =====================================================================
    sp = load_tokenizer(args.tokenizer, verbose=args.verbose)
    write_tokenizer_files(
        out_dir, sp, args.tokenizer, hp["context_length"], verbose=args.verbose
    )

    # =====================================================================
    # 6. Model weights -> model.safetensors  (metadata_only skips this)
    # =====================================================================
    tensors_written = 0
    if not args.metadata_only:
        try:
            from safetensors.torch import save_file
        except ImportError:
            print("=" * 72)
            print("ERROR: 'safetensors' is required to write model.safetensors.")
            print("Install it with:")
            print("    pip install safetensors")
            print("=" * 72)
            sys.exit(1)

        dtype = _torch_dtype(args.dtype)
        converted = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                # `.clone()` breaks memory aliasing -- Dizel ties
                # `lm_head.weight` to `transformer.tok_emb.weight`, and
                # safetensors rejects tensors sharing memory. Both keys are
                # kept (identical values) so the file maps 1:1 to weights.
                converted[k] = v.to(dtype=dtype).clone().contiguous()
            else:
                print(f"  WARNING: skipping non-tensor key '{k}' "
                      f"(type={type(v).__name__})")

        out_path = os.path.join(out_dir, "model.safetensors")
        save_file(converted, out_path)
        tensors_written = len(converted)
        if args.verbose:
            print(f"  Wrote {tensors_written} tensors -> {out_path} "
                  f"({os.path.getsize(out_path):,} bytes)")
    else:
        if args.verbose:
            print("  (metadata-only mode -- no weight tensors written)")

    # =====================================================================
    # 7. Summary
    # =====================================================================
    param_count = sum(
        v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor)
    )
    files = sorted(os.listdir(out_dir))

    print()
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"  Source checkpoint : {args.checkpoint}")
    print(f"  Source format     : {source_info}")
    print(f"  Parameters        : {param_count:,}")
    print(f"  Dtype             : {args.dtype}")
    print(f"  Tensors written   : {tensors_written}" +
          ("  (metadata only)" if args.metadata_only else ""))
    print(f"  Hyperparameters:")
    print(f"    vocab_size      : {hp['vocab_size']}")
    print(f"    d_model         : {hp['d_model']}")
    print(f"    n_layers        : {hp['n_layers']}")
    print(f"    n_heads         : {hp['n_heads']}")
    print(f"    kv_heads        : {hp['kv_heads']}")
    print(f"    head_dim        : {hp['head_dim']}")
    print(f"    intermediate    : {hp['intermediate_size']}")
    print(f"    context_length  : {hp['context_length']}")
    print(f"    rope_theta      : {hp['rope_theta']}")
    print(f"    norm_eps        : {hp['norm_eps']}")
    print(f"    weight_tying    : {hp['weight_tying']}")
    print(f"  Output dir        : {out_dir}")
    print(f"  Files written     : {len(files)}")
    for f in files:
        print(f"    - {f}")
    print()
    print("Ready-to-paste load snippet:")
    print("=" * 60)
    print(f'    from transformers import AutoModelForCausalLM, AutoTokenizer')
    print(f'    model = AutoModelForCausalLM.from_pretrained("{out_dir}", trust_remote_code=True)')
    print(f'    tokenizer = AutoTokenizer.from_pretrained("{out_dir}")')
    print("=" * 60)
    elapsed = time.perf_counter() - _START
    print(f"  Time elapsed      : {elapsed:.2f} seconds")
    print()


_START = time.perf_counter()


if __name__ == "__main__":
    main()