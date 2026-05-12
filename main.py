# config.py
# -----------------------------------------------------------------------------
# asr system entry point (set-up, training and inference from CLI)
# -----------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github)
# iker g.f.
# paula g.c.
# may-2026

import argparse
import subprocess
import config
import logging
from data import prepare, tokenize
from inference import asr_inference
from pathlib import Path
# from model_lm import train_lm
# from model_asr import train_asr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def _run(script_dir: str, yaml_file: str):
    """Helper function to run heavy SpeechBrain training scripts via CLI
    (better to just run our own commands)."""
    cmd = f"cd {script_dir} && python train.py {yaml_file}"
    logging.info(f"$ {cmd}")
    subprocess.run(cmd, shell=True)

def main():
    parser = argparse.ArgumentParser(description="SpeechBrain ASR pipeline")
    parser.add_argument("--step", type=str, required=True,
                        choices=["prepare", "tokenize", "lm", "asr", "inference"],
                        help="ASR pipeline phase to execute")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to the .wav file for inference (optional)")
    
    args = parser.parse_args()

    if args.step == "prepare":
        prepare()
        
    elif args.step == "tokenize":
        tokenize()
        
    elif args.step == "lm":
        logging.info("Launching Language Model training...")
        _run("LM", "RNNLM.yaml")
        
    elif args.step == "asr":
        logging.info("Launching Acoustic Model (CRDNN) training...")
        _run("ASR", "asr.yaml") 
        
    elif args.step == "inference":
        if args.audio:
            audio_path = Path(args.audio)
        else:
            # ** fallback: first available audio if none provided **
            wavs = list(config.RAW_DATA_DIR.glob("*.wav")) + list(config.RAW_DATA_DIR.glob("*.flac"))
            if not wavs:
                logging.error(f"No audio files found in {config.RAW_DATA_DIR}")
                return
            audio_path = wavs
            
        asr_inference(str(audio_path))

if __name__ == "__main__":
    main()