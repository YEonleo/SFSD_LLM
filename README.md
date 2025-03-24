# README

This script **decomposes (compresses) certain Linear layers in an LLM model (e.g., Llama) and evaluates the decomposed model**. Internally, it performs the following major steps:

1. **Load the base model and identify decomposable layers**  
2. **Inject decomposable modules**  
3. **Run a forward test**  
4. **Save the decomposed model**  
5. **Evaluate the baseline model**  
6. **Perform layer-by-layer rank (compression rate) search and prune**  
7. **Re-evaluate and log results**

---

## Requirements

- **Python 3.8+**  
- **PyTorch 2.x+**  
- **Hugging Face Transformers**  
- **Hugging Face Datasets**  
- **Weights & Biases (wandb)** (optional, remove if you prefer)  
- Other libraries: `numpy`, `pandas`, `tqdm`, `json`

**Installation**:

pip install torch torchvision torchaudio
pip install transformers datasets
pip install wandb
pip install numpy pandas tqdm


project/
├── sfds_main.py                # main script with main()
├── preprocess.py               # data preprocessing utilities
├── dataset_ppl.py              # dataset utilities
├── layers.py                   # custom decomposition logic
├── lm_eval/
│   ├── evaluator.py            # evaluation logic
│   └── models/
│       └── huggingface.py      # HFLM wrapper
├── logs/                       # logs/checkpoints (optional)
└── README.md                   # this file
