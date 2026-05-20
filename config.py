# config.py
# -----------------------------------------------------------------------------
# global configuration and path settings for speechbrain ASR system dev
# -----------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github)
# iker g.f.
# paula g.c.
# may-2026

import os
from pathlib import Path

# ----------------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
LOCAL_DATA_DIR = BASE_DIR / "workspace" 

RAW_DATA_DIR = LOCAL_DATA_DIR / "raw" / "LibriSpeech"
PROCESSED_DATA_DIR = LOCAL_DATA_DIR / "processed"
MODELS_DIR = LOCAL_DATA_DIR / "models"
RESULTS_DIR = LOCAL_DATA_DIR / "results"

# ----------------------------------------------------------------------- params
HF_SOURCE = "speechbrain/asr-crdnn-rnnlm-librispeech"
HF_SAVEDIR = MODELS_DIR / "hf_model"