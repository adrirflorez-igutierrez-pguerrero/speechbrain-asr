# main.py
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
from manifest import prepare, tokenize
from asr import train_asr, asr_inference

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

    # 1. manifests > python main.py --step prepare &> workspace/1_prepare.log
    if args.step == "prepare":
        prepare()

    # 2. tokenizer > python main.py --step tokenize &> workspace/2_tokenize.log
    elif args.step == "tokenize":
        tokenize()

    # 3. language model > python main.py --step lm &> workspace/3_lm.log
    elif args.step == "lm":
        logging.info("Launching Language Model training...")
        _run("LM", "RNNLM.yaml")
        
    elif args.step == "asr":
        train_asr(batch_size=2) 
        
    elif args.step == "inference":
        if args.audio:
            asr_inference(args.audio)
        else:
            wavs = list(config.RAW_DATA_DIR.glob("*.wav")) + list(config.RAW_DATA_DIR.glob("*.flac"))
            if not wavs:
                logging.error(f"\t> (!) No audio files found in {config.RAW_DATA_DIR}")
                return
            for wav in wavs:
                asr_inference(str(wav))

if __name__ == "__main__":
    main()