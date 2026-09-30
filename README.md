# 🎙️ Our STT Model — Proprietary Multilingual Speech-to-Text

A production-oriented, modular Speech-to-Text (STT) engine built from scratch using **Python** and **PyTorch**. Designed to support **English**, **Hindi**, and **Gujarati** with CPU-friendly inference on macOS.

---

## 🎯 Project Purpose

Build a proprietary multilingual STT model that:

- Transcribes speech in **English 🇬🇧**, **Hindi 🇮🇳**, and **Gujarati** accurately
- Runs inference on **Mac / CPU** without requiring CUDA
- Supports **real-time** speech-to-text
- Is fully modular, testable, and production-ready

---

## 🏗️ Planned STT Pipeline

```
Audio Input (.wav / .mp3 / .flac)
        │
        ▼
  ┌──────────────┐
  │ Preprocessing │  ← Resampling, Noise Removal, Normalization
  └──────┬───────┘
         ▼
  ┌──────────────────┐
  │ Mel Spectrogram   │  ← Feature Extraction (log-mel, MFCC)
  └──────┬───────────┘
         ▼
  ┌──────────────┐
  │   Encoder     │  ← Convolutional + Transformer Layers
  └──────┬───────┘
         ▼
  ┌──────────────────────┐
  │ Attention / Transformer │  ← Multi-Head Self-Attention
  └──────┬───────────────┘
         ▼
  ┌──────────────┐
  │   Decoder     │  ← Autoregressive Token Prediction
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │   Tokens      │  ← Subword / Character Token IDs
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │    Text       │  ← Final Transcription Output
  └──────────────┘
```

---

## 📂 Project Structure

```
our-stt-model/
│
├── data/                  # Raw audio, processed features, metadata
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── preprocessing/         # Audio loading, cleaning, feature extraction
│   ├── __init__.py
│   ├── audio.py
│   └── features.py
│
├── tokenizer/             # Text tokenization for multilingual output
│   ├── __init__.py
│   └── tokenizer.py
│
├── dataset/               # PyTorch Dataset / DataLoader definitions
│   ├── __init__.py
│   └── stt_dataset.py
│
├── model/                 # Core model architecture
│   ├── __init__.py
│   ├── encoder.py
│   ├── attention.py
│   ├── decoder.py
│   └── stt_model.py
│
├── training/              # Training loop, validation, hyperparameters
│   ├── __init__.py
│   ├── train.py
│   ├── validate.py
│   └── config.py
│
├── evaluation/            # WER, CER, and other metrics
│   ├── __init__.py
│   └── metrics.py
│
├── inference/             # Real-time and batch inference
│   ├── __init__.py
│   └── inference.py
│
├── configs/               # YAML configuration files
│   └── config.yaml
│
├── scripts/               # Utility scripts (data download, conversion, etc.)
├── tests/                 # Unit and integration tests
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## 🌐 Supported Languages

| Language | Script     | Status   |
|----------|------------|----------|
| English  | Latin      | Planned  |
| Hindi    | Devanagari | Planned  |
| Gujarati | Gujarati   | Planned  |

---

## 💻 Development Environment

| Tool           | Purpose                                      |
|----------------|----------------------------------------------|
| **VS Code**    | Primary development IDE                      |
| **Google Colab** | GPU-accelerated experiments and training    |
| **GitHub**     | Version control and team collaboration       |
| **Python 3.11+** | Runtime                                    |
| **PyTorch**    | Deep learning framework                      |

---

## 👥 Team Responsibilities (4 People)

| Person | Role                                           | Key Modules                              |
|--------|------------------------------------------------|------------------------------------------|
| **P1** | Data Collection + Cleaning + Preprocessing     | `data/`, `preprocessing/`, `scripts/`    |
| **P2** | Tokenizer + Text Processing + Evaluation       | `tokenizer/`, `evaluation/`              |
| **P3** | Model Architecture (Encoder + Attention + Decoder) | `model/`                              |
| **P4** | Training + Validation + Optimization + Deployment | `training/`, `inference/`, `configs/`  |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-org>/our-stt-model.git
cd our-stt-model
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; import torchaudio; print(f'PyTorch {torch.__version__}, Torchaudio {torchaudio.__version__}')"
```

---

## 📌 Current Status

> **Phase: Project Scaffolding**
>
> The project skeleton is ready. No model has been trained yet, no datasets have been downloaded, and no fake/demo weights exist. This is a clean foundation for collaborative development.

---

## 📝 License

This project is proprietary. See [LICENSE](./LICENSE) for details.
