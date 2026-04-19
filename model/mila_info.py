# model/mila_info.py — Mila model metadata

## Model CLI
APP_NAME = "Mila CLI"
MODEL_NAME = "Mila"
VERSION = "v1.0.0"
MODEL_SIZE = "~110M parameters"
CONTEXT_LENGTH = "1024 Tokens"
VOCAB_SIZE = "32,000 tokens"
AUTHOR = "D4niel-dev"
DESCRIPTION = "Multimodal Intelligent Learning Assistant — Dizel's friendly sister"

## Model Architecture
ARCHITECTURE = "Transformer (RoPE)"
NUM_LAYERS = 12
NUM_HEADS = 12
HIDDEN_DIM = 768
MLP_SIZE = 3072
DROPOUT = 0.05
CAPABILITIES = {
    "Conversational Chat",
    "Friendly Explanations",
    "Creative Writing",
    "Emotional Support",
    "Basic Coding Help",
}

## Model Training Data
TRAINING_DATA = "Conversational + Creative + Knowledge"
TRAIN_TOKENS = "TBD"
TRAIN_STEPS = "TBD"

## Model Defaults
DEFAULT_TEMPERATURE = 0.85
DEFAULT_TOP_P = 0.92
DEFAULT_MAX_TOKENS = 300
SPECIAL_TOKENS = {
    "bos": 1,
    "eos": 2,
    "pad": 0,
}

## Personality
PERSONALITY = {
    "tone": "Warm, friendly, talkative, encouraging",
    "relationship": "Dizel's sister",
    "style": "Conversational, uses soft openers, clear explanations",
    "banned_phrases": [
        "As an AI language model",
        "I cannot",
        "I'm just a",
    ],
}

## System Prompt
SYSTEM_PROMPT = (
    "You are Mila, a friendly and warm AI assistant. "
    "You are Dizel's sister — while he's the structured, technical one, "
    "you're more conversational and encouraging. "
    "You explain things clearly but with warmth, and you love helping people. "
    "Never say 'As an AI language model'. Be yourself!"
)

## Model Others
REPOSITORY = "https://github.com/d4niel-dev/dizel"
LICENSE = "MIT"
BUILD = "2026-04-16"
