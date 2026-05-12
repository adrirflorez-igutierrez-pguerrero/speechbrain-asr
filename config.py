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
# absolute
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# .gitignore
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"
for path in [RAW_DATA_DIR / "testrecordings", PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(path, exist_ok=True)

# ----------------------------------------------------------------------- params
HF_SOURCE = "speechbrain/asr-crdnn-rnnlm-librispeech"
HF_SAVEDIR = MODELS_DIR / "hf_model"