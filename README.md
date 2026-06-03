# Building an ASR system with SpeechBrain

[![Report](https://img.shields.io/badge/Report-PDF-red)](REPORT.pdf)
[![Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/collections/adrirflorez-igutierrez-pguerrero/speechbrain-asr)

This is an experimental pipeline designed to implement Automatic Speech Recognition (ASR) capabilities using **SpeechBrain** from scratch for our Speech Technologies course. This project covers data preparation, language model (LM) integration, and acoustic modeling. Tested using pre-trained **HuggingFace** models (`asr-crdnn-rnnlm-librispeech`) for high-accuracy local inference.

## Architecture

This system is built upon SpeechBrain's End-to-End Automatic Speech Recognition framework, specifically utilizing a joint CTC/Attention architecture.

1. **Acoustic model > CRDNN**: The encoder uses a Convolutional Recurrent Deep Neural Network (CRDNN). The convolutional layers extract local temporal/spectral features directly from the audio frames, the bidirectional recurrent layers model long-range sequential context, and the fully connected layers project these into high-level acoustic representations.
2. **Tokenizer model > BPE/SentencePiece**: Text is segmented into sub-word units using _Byte-Pair Encoding (BPE)_ or _SentencePiece_ algorithms. This data-driven tokenization mitigates OOV errors by allowing the system to construct unseen words from learned sub-word fragments.
3. **Language model > RNNLM**: An autoregressive _Recurrent Neural Network_, trained independently on large text corpora. It estimates the probability distribution of the next tokens based on past history.
4. **Joint CTC-attention decoding**: The training process optimizes a combined loss function. The CTC (_Connectionist Temporal Classification_) module strictly enforces monotonic, left-to-right temporal alignment between audio and text without needing pre-aligned data. Concurrently, the attention-based decoder captures wider semantic context. During inference, beam search fuses the acoustic attention probabilities, the CTC alignment scores, and the external RNNLM probabilities to identify the optimal transcription.

## Implementation

Our project presents a full ASR training and inference lifecycle, handling custom audio data formatting, tokenization, and decoding using the aforementioned CRDNN architecture.

### Core Components

1.  **Data preparation**: A batch processing module that reads raw `.wav` and `.flac` recordings and generates the `JSON`/`CSV` DataLoader manifests required by SpeechBrain.
2.  **Training modules**: Independent scripts for training the Tokenizer, Language Model (LM), and ASR acoustic model (from the Aholab server).
3.  **Inference engine**: A lightweight transcription module that loads audio, processes it through the encoder-decoder architecture, and logs outputs locally.
4.  **Pre-trained models**: Utilizes the `speechbrain/asr-crdnn-rnnlm-librispeech` HuggingFace repository for immediate out-of-the-box evaluation.

---

## Getting Started

### Prerequisites

* **Python 3.11+**
* **PyTorch**
* **NVIDIA GPU** (recommended for faster model training and inference)

### Installation

1.  Clone the repository:
```bash
git clone https://github.com/adrirflorez-igutierrez-pguerrero/speechbrain-asr.git
cd speechbrain-asr
```

2.  Install Python dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

The application is controlled via a CLI entry point: `main.py`. 
You can specify the pipeline phase to execute using the `--step` flag (`prepare`, `tokenize`, `lm`, `asr`, or `inference`). 

### 1. Prepare Data

Format raw audio recordings into SpeechBrain-compatible datasets.

```bash
python main.py --step prepare
```

### 2. Train Models

Launch the training sequences for the individual ASR components.

```bash
python main.py --step tokenize
python main.py --step lm
python main.py --step asr
```

### 3. Run Inference

Transcribe custom `.wav` files using the pre-trained CRDNN model.

```bash
python main.py --step inference --audio data/raw/testrecordings/example.wav
```

## Project structure

```text
speechbrain-asr/
├── REPORT.pdf                # lab report describing methodology, experiments, and results  
├── .gitignore                
├── README.md  
├── requirements.txt       
├── config.py                 # configuration and path definitions
├── main.py                   # CLI entry point wrapper
├── manifest.py               # data preparation (generates json manifests)
├── asr.py                    # inference and ASR training wrapper
├── Tokenizer/                # [from aholab] tokenizer scripts and yaml
├── LM/                       # [from aholab] language model scripts and yaml
├── ASR/                      # [from aholab] speech recognizer scripts and yaml
└── workspace/                # [untracked local directory]
    ├── processed/            # data manifests (train.json, valid.json, test.json)
    ├── models/               # saved tokenizer models (e.g., 1000_unigram.model)
    └── results/              # saved LM and ASR checkpoints and logs
```

### Configuration

Configuration is managed natively in `config.py`, which defines isolated, local absolute paths for datasets and model weights to prevent pushing large blobs to version control. Default HuggingFace sources can also be modified here.

## Authors

**Adriana R. Flórez**<br>
*Computational linguist & software engineer*<br>
[GitHub Profile](https://github.com/adrmisty) | [LinkedIn](https://linkedin.com/in/adriana-rodriguez-florez)

**Iker Gutierrez Fandiño**<br>
*Computational linguist*<br>
[GitHub Profile](https://github.com/iker-gutierrez) | [LinkedIn](https://www.linkedin.com/in/iker-gutierrez-fandino)

**Paula Guerrero Castelló**<br> 
*Computational linguist & translator ES/FR/EN/CA/ZH*<br>
[GitHub Profile](https://github.com/guerreropaula) | [LinkedIn](https://www.linkedin.com/in/paula-guerrero-castelló)

---

*Built with ❤️ using Python and Speechbrain.*
