# model/dizel_info.py

## Model CLI
APP_NAME = "Dizel CLI"
MODEL_NAME = "Dizel"
VERSION = "v1.5"
MODEL_SIZE = "~454.8M parameters"
CONTEXT_LENGTH = "2048 Tokens"
VOCAB_SIZE = "100,000 tokens"
AUTHOR = "D4niel-dev"
DESCRIPTION = "A Structured Analytical Reasoning & Math LLM"

## Model Architecture
ARCHITECTURE = "Transformer (SwiGLU + GQA + RMSNorm + RoPE)"
NUM_LAYERS = 28
NUM_HEADS = 16
KV_HEADS = 8
HIDDEN_DIM = 1024
MLP_SIZE = 3072
DROPOUT = 0.0
CAPABILITIES = {
    "Reasoning",
    "Structured JSON Output",
    "Step-by-steps Analysis (WIP)",
    "Tool-ready Responses (Planned)",
    "2,048-token Context",
}

## Model Training Data
TRAINING_DATA = "Mixed Web + Synthetic Reasoning + Code"
TRAIN_TOKENS = "TBD"
TRAIN_STEPS = "TBD"

## Model Defaults
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 1000
SPECIAL_TOKENS = {
    "bos": 1,
    "eos": 2,
    "pad": 0,
}

## Model Others
REPOSITORY = "https://github.com/d4niel-dev/dizel"
LICENSE = "MIT"
BUILD = "2026-04-26"
