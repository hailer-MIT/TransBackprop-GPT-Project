# TransBackprop GPT Project

## 🏆 "From First Principles: Building a GPT Transformer from Scratch"

Welcome! This project is a complete, readable implementation of a GPT-style Transformer, designed to help others understand and experiment with modern large language model (LLM) architectures. It features:
- Fully transparent PyTorch code
- Clear modular design
- Self-contained training pipeline
- Thorough comments and reproducible results

---

## 🚀 What is this Project?

This project demystifies how GPT works under the hood by re-building the architecture and training pipeline from scratch. It uses the Penn Treebank (PTB) dataset and follows the principles that power state-of-the-art LLMs like GPT-2 and GPT-3.

**Key features:**
- From tokenization to text generation, every step from first principles.
- Optimized training loop with AdamW, learning rate scheduling, gradient clipping, and checkpointing.
- Detailed configuration for easy customization/experimenting.
- Pedagogically clean code—no framework magic.

---

## ⚙️ Model Configuration & Stats

- **Transformer Layers:** 6
- **Attention Heads per Layer:** 8
- **Embedding Dimension:** 256
- **Maximum Sequence Length:** 256 tokens
- **Vocabulary Size:** 50,257 (based on GPT-2 vocab)
- **Batch Size:** 8
- **Dropout:** 0.1
- **Epochs:** 20
- **Learning Rate:** 3e-4 (AdamW optimizer)
- **Gradient Clipping:** 1.0
- **Total Parameters:** ~10 million

---

## 🛠️ How to Run

#### 1. Clone & Enter Project
```sh
git clone https://github.com/hailer-MIT/TransBackprop_GPT_Project.git
cd TransBackprop_GPT_Project
```

#### 2. Create & Activate the Python venv
```sh
python -m venv gpt2venv
.\gpt2venv\Scripts\Activate.ps1   # (on Windows PowerShell)
```

#### 3. Install dependencies
```sh
pip install -r requirements.txt
```

#### 4. Move to Parent Directory
```sh
cd ..
```

#### 5. Train the Model
```sh
python -m TransBackprop_GPT_Project.train
```
- **Resumes automatically** from last checkpoint if present.

##### To start a fresh run (ignore previous checkpoints):
```sh
python -m TransBackprop_GPT_Project.train --no-resume
```

---

## ⚡ Project Structure
- `train.py` — Main script: handles training, validation, checkpointing, and generation
- `model.py` — The Transformer (GPT) architecture
- `dataset.py` — Data loaders, tokenization, batch generation
- `config.py` — Easy access to experiment settings, hyperparameters

---


## 📬 Questions?

Open an issue or contact the repo owner. Contributions, collaborations, and hiring inquiries are welcome!
