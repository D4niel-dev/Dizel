# model/dizel_info.py

## Model CLI
APP_NAME = "Dizel CLI"
MODEL_NAME = "Dizel"
VERSION = "v1.0.0"
MODEL_SIZE = "8.8M parameters"
CONTEXT_LENGTH = "256 Tokens"
VOCAB_SIZE = "16,000 tokens"
AUTHOR = "D4niel-dev"
DESCRIPTION = "A Structured Analytical LLM"

## Model Architecture
ARCHITECTURE = "Transformer"
NUM_LAYERS = 6
NUM_HEADS = 4
HIDDEN_DIM = 256
MLP_SIZE = 1024
DROPOUT = 0.3
CAPABILITIES = {
    "Reasoning",
    "Structured JSON Output",
    "Step-by-steps Analysis (WIP)",
    "Tool-ready Responses (Planned)"
}

## Model Training Data
TRAINING_DATA = "Mixed Web + Synthetic Reasoning"
TRAIN_TOKENS = "50M Tokens"
TRAIN_STEPS = "4000 ~ 5000"

## Model Defaults
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 256
SPECIAL_TOKENS = {
    "bos": 1,
    "eos": 2,
    "pad": 0,
}

## Model Others
REPOSITORY = "https://github.com/d4niel-dev/dizel"
LICENSE = "MIT"
BUILD = "2026-03-08"