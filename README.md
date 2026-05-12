# ASR system with SpeechBrain

An experimental pipeline designed to implement Automatic Speech Recognition (ASR) capabilities using **SpeechBrain** from scratch for our Speech Technologies course. This project covers data preparation, language model (LM) integration, and acoustic modeling. Tested using pre-trained **HuggingFace** models (`asr-crdnn-rnnlm-librispeech`) for high-accuracy local inference.

## Architecture

This system is built upon SpeechBrain's End-to-End Automatic Speech Recognition framework, specifically utilizing a joint CTC/Attention architecture.

1. **Acoustic model > CRDNN**: The encoder uses a Convolutional Recurrent Deep Neural Network (CRDNN). The convolutional layers extract local temporal/spectral features directly from the audio frames, the bidirectional recurrent layers model long-range sequential context, and the fully connected layers project these into high-level acoustic representations.
2. **Tokenizer model > BPE/SentencePiece**: Text is segmented into sub-word units using _Byte-Pair Encoding (BPE)_ or _SentencePiece_ algorithms. This data-driven tokenization mitigates OOV errors by allowing the system to construct unseen words from learned sub-word fragments.
3. **Language model > RNNLM**: An autoregressive _Recurrent Neural Network_, trained independently on large text corpora. It estimates the probability distribution of the next tokens based on past history.
4. **Joint CTC-attention decoding**: The training process optimizes a combined loss function. The CTC (_Connectionist Temporal Classification_) module strictly enforces monotonic, left-to-right temporal alignment between audio and text without needing pre-aligned data. Concurrently, the attention-based decoder captures wider semantic context. During inference, beam search fuses the acoustic attention probabilities, the CTC alignment scores, and the external RNNLM probabilities to identify the optimal transcription.

## Implementation

Our project presents a full ASR training and inference lifecycle, handling custom audio data formatting, tokenization, and decoding using the aforementioned CRDNN architecture.

### Core Components

1.  **Data preparation**: A batch processing module that reads raw `.wav` recordings and generates the `JSON`/`CSV` DataLoader manifests required by SpeechBrain.
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
git clone git@github.com:adrmisty/speechbrain-asr-practice.git
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
speechbrain-asr-practice/
├── requirements.txt            
├── README.md                   
├── config.py                 
├── main.py                   # CLI entry point
├── prepare.py                
├── tokenizer.py              # [from aholab]
├── model_lm.py               # [from aholab]
├── model_asr.py              # [from aholab]
├── inference.py              # asr/eval
└── data/                     
    ├── raw/testrecordings/   # input .wav files
    ├── processed/            # dataloaders
    ├── models/               # pre-trained HF models
    └── results/              # output logs

```

### Configuration

Configuration is managed natively in `config.py`, which defines isolated, local absolute paths for datasets and model weights to prevent pushing large blobs to version control. Default HuggingFace sources can also be modified here.

## Authors

**Adriana R. Flórez**
*Computational Linguist & Software Engineer*
[GitHub Profile](https://github.com/adrmisty) | [LinkedIn](https://linkedin.com/in/adriana-rodriguez-florez)

**Paula Guerrero Castello**

**Íker Gutiérrez Fandiño**

---

*Built with ❤️ using Python and Speechbrain.*