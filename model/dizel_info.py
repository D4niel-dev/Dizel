# model/dizel_info.py

## Model CLI
APP_NAME = "Dizel CLI"
MODEL_NAME = "Dizel"
VERSION = "v1.2.1"
MODEL_SIZE = "205M parameters"
CONTEXT_LENGTH = "2048 Tokens"
VOCAB_SIZE = "32,000 tokens"
AUTHOR = "D4niel-dev"
DESCRIPTION = "A Structured Analytical Reasoning & Math LLM"

## Model Architecture
ARCHITECTURE = "Transformer (RoPE)"
NUM_LAYERS = 20
NUM_HEADS = 16
HIDDEN_DIM = 896
MLP_SIZE = 3136
DROPOUT = 0.05
CAPABILITIES = {
    "Reasoning",
    "Structured JSON Output",
    "Step-by-steps Analysis (WIP)",
    "Tool-ready Responses (Planned)",
    "Extended Context (4096 tokens)",
}

## Model Training Data
TRAINING_DATA = "Mixed Web + Synthetic Reasoning + Code"
TRAIN_TOKENS = "TBD"
TRAIN_STEPS = "TBD"

## Model Defaults
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 400
SPECIAL_TOKENS = {
    "bos": 1,
    "eos": 2,
    "pad": 0,
}

## Model Others
REPOSITORY = "https://github.com/d4niel-dev/dizel"
LICENSE = "MIT"
BUILD = "2026-04-26"