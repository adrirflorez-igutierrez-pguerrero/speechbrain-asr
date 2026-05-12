# audio.py
# -----------------------------------------------------------------------------
# pre-processing audio data for speechbrain.
# - data preparation (.json manifest)
# - tokenization (bpe/sentencepiece training)
# -----------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github)
# iker g.f.
# paula g.c.
# may-2026

import json
import logging
import subprocess
import torchaudio
from tqdm import tqdm

import config

def prepare():
    """Scans raw audio files, extracts their exact duration, and generates a data.json manifest for SpeechBrain.
    """
    logging.info(f"Scanning recordings in: {config.RAW_DATA_DIR}")
    
    if not config.RAW_DATA_DIR.exists():
        logging.error(f"> (!) Directory not found: {config.RAW_DATA_DIR}")
        return

    # .wav and .flac (librispeech)
    audio_files = list(config.RAW_DATA_DIR.glob("*.wav")) + list(config.RAW_DATA_DIR.glob("*.flac"))
    
    if not audio_files:
        logging.info("Directory is empty: please place .wav or .flac files here")
        return
        
    logging.info(f"Found {len(audio_files)} audio files. Extracting metadata for manifest...")
    data_manifest = {}
    
    for file_path in tqdm(audio_files, desc="Generating JSON"):
        audio_id = file_path.stem
        
        try:
            info = torchaudio.info(str(file_path))
            duration_seconds = round(info.num_frames / info.sample_rate, 2)
        except Exception as e:
            logging.warning(f"\t> (!) Failed to read metadata for {file_path.name}: {e}")
            duration_seconds = 0.0
            
        data_manifest[audio_id] = {
            "wav": str(file_path.resolve()),
            "length": duration_seconds,
            "words": "DUMMY TRANSCRIPTION" # Only relevant for inference testing
        }
        
    output_json = config.PROCESSED_DATA_DIR / "data.json"
    
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(data_manifest, json_file, indent=4, ensure_ascii=False)
        
    logging.info(f">>> Data manifest saved to: {output_json}")


def tokenize(script_dir="Tokenizer", yaml_file="tokenizer.yaml"):
    """Launches the tokenizer training (#TODO:scripts downloaded from aholab).
    """
    manifest_path = config.PROCESSED_DATA_DIR / "data.json"
    
    if not manifest_path.exists():
        logging.error(f"\t > (!) Data manifest does not exist: {manifest_path}; run 'prepare' step first!")
        return
    
    cmd = f"cd {script_dir} && python train.py {yaml_file}"
    logging.info(f"Starting tokenizer training: {cmd}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        logging.info(">>> Tokenizer training completed.")
    except subprocess.CalledProcessError as e:
        logging.error(f"\t> (!) Tokenizer training failed: {e}")