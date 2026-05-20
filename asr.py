# asr.py
# -----------------------------------------------------------------------------
# automatic audio transcription with pre-trained HF models + their evaluation
# results saved into local dirs
# -----------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github)
# iker g.f.
# paula g.c.
# may-2026

import os
import logging
from speechbrain.inference.ASR import EncoderDecoderASR
import config

import subprocess

def train_asr(script_dir="ASR", yaml_file="train.yaml", batch_size=2):
    """Launches the ASR training script (CRDNN + LM + CTC)."""
    cmd = f"cd {script_dir} && python train.py {yaml_file} --batch_size={batch_size}"
    logging.info(f"Starting ASR training: {cmd}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        logging.info(">>> ASR training completed.")
    except subprocess.CalledProcessError as e:
        logging.error(f"\t> (!) ASR training failed: {e}")

def asr_inference(audio_file_path: str):
    """Loads the pre-trained CRDNN model and transcribes the provided audio."""
    logging.info(f"(Down)Loading weights from {config.HF_SOURCE}...")
    
    try:
        asr_model = EncoderDecoderASR.from_hparams(
            source=config.HF_SOURCE, 
            savedir=str(config.HF_SAVEDIR)
        )
    except Exception as e:
        logging.error(f"\t> (!) Failed to load HuggingFace model: {e}")
        return None
    
    logging.info(f"Analyzing audio: {audio_file_path}")
    if not os.path.exists(audio_file_path):
        logging.error(f"\t> (!) Audio file not found: {audio_file_path}")
        return None

    # ** run inference with asr model **
    transcription = asr_model.transcribe_file(str(audio_file_path))
    logging.info(f"[ASR]: {transcription}")
    
    result_log = config.RESULTS_DIR / "asr_log.txt"
    with open(result_log, "a", encoding="utf-8") as f:
        f.write(f"File: {os.path.basename(audio_file_path)}\n")
        f.write(f"Result: {transcription}\n")
        f.write("-" * 50 + "\n")
        
    logging.info(f"Result appended to log: {result_log}")
    return transcription