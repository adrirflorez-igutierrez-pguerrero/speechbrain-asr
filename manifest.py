# manifest.py
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
    """Scans raw audio files and .trans.txt files to generate train, valid, and test JSON manifests."""
    logging.info(f"Scanning recordings in: {config.RAW_DATA_DIR}")
    
    if not config.RAW_DATA_DIR.exists():
        logging.error(f"> (!) Directory not found: {config.RAW_DATA_DIR}")
        return

    logging.info("Reading transcription files...")
    trans_files = list(config.RAW_DATA_DIR.rglob("*.trans.txt"))
    trans_dict = {}
    
    # transcriptions: 8455-210777-0000 I REMAINED THERE ALONE FOR MANY HOURS ....
    for trans_file in trans_files:
        with open(trans_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()     # .split() handles multiple spaces automatically
                if not parts:
                    continue
                uttid = parts[0]               
                text = " ".join(parts[1:])       
                trans_dict[uttid] = text
    logging.info(f"Loaded {len(trans_dict)} transcriptions.")

    # * find raw audio files *
    audio_files = list(config.RAW_DATA_DIR.rglob("*.flac")) + list(config.RAW_DATA_DIR.rglob("*.wav"))
    
    if not audio_files:
        logging.info("> (!) Directory is empty: please place .wav or .flac files here")
        return
        
    logging.info(f">> Found {len(audio_files)} audio files. Extracting metadata for manifests...")
    
    manifests = {
        "train": {},
        "valid": {},
        "test": {}
    }
    
    for file_path in tqdm(audio_files, desc="Generating JSON"):
        audio_id = file_path.stem
        path_str = str(file_path)
        
        if "train-clean" in path_str:
            split = "train"
        elif "dev-clean" in path_str:
            split = "valid"
        elif "test-clean" in path_str:
            split = "test"
        else:
            continue
            
        try:
            waveform, sample_rate = torchaudio.load(str(file_path))
            duration_seconds = round(waveform.shape[0] / sample_rate, 2)
        except Exception as e:
            logging.warning(f"\t> (!) Failed to read metadata for {file_path.name}: {e}")
            duration_seconds = 0.0
            
        # ** manifest entry: audio, length and transcription **
        manifests[split][audio_id] = {
            "wav": str(file_path.absolute()),
            "length": duration_seconds,
            "words": trans_dict.get(audio_id, "") 
        }
        
    # * data splits for manifests *
    for split_name, data_dict in manifests.items():
        if data_dict:
            output_json = config.PROCESSED_DATA_DIR / f"{split_name}.json"
            with open(output_json, "w", encoding="utf-8") as json_file:
                json.dump(data_dict, json_file, indent=4, ensure_ascii=False)
            logging.info(f">>> {split_name.capitalize()} manifest saved to: {output_json} ({len(data_dict)} items)")


def tokenize(script_dir="Tokenizer", yaml_file="tokenizer.yaml"):
    """Launches the tokenizer training (scripts downloaded from aholab).
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
        
def tokenize(script_dir="Tokenizer", yaml_file="tokenizer.yaml", split="train"):
    """Launches the tokenizer training (scripts downloaded from aholab into the Tokenizer/ dir).
    """
    manifest_path = config.PROCESSED_DATA_DIR / f"{split}.json"
    
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