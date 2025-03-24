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


---
# Usage
## How to Run
python your_script_name.py [OPTIONS]

## Example:
python sfds_main.py \
  --model meta-llama/Llama-2-7b-hf \
  --dataset piqa \
  --layers "down_proj,up_proj" \
  --max_ratio 0.8 \
  --delta 0.01 \
  --start_layer 20 \
  --weights_name "my_decomposed_model" \
  --batch_size 256 \
  --seq_len 128

* Replace your_script_name.py with your actual script name (e.g., sfds_main.py).

* Use --help or -h to see available arguments.

## Key Arguments
Argument	Type	Default	Description
--layers	str	"o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj"	Comma-separated keywords for layers (e.g., o_proj,q_proj).
--dataset	str	"piqa"	Dataset name
--batch_size	int	512	Batch size
--seq_len	int	128	Sequence length
--log_path	str	"surgical_logs.txt"	Log file path
--algo	str	"eigen"	Decomposition algorithm (eigen, svd)
--weights_name	str	"decomposed_model_mistral_combination"	Name for saving decomposed model
--model	str	"meta-llama/Llama-2-7b-hf"	Hugging Face model name
--cache_dir	str	"../cache_dir"	Directory for cached models
--delta	float	0.0	Allowed accuracy drop
--start_layer	int	28	Start layer index
--base_model	str	"decomposed_model_mistral_combination.pt"	Base model file
--max_ratio	float	1.0	Max compression ratio

--- 
## Workflow Summary
- Load Base Model: Load via Hugging Face API.

- Identify Decomposable Layers: Locate decomposable linear layers.

- Inject Modules: Inject custom decomposable modules.

- Forward Test: Verify correctness with dataset forward pass.

- Save Model: Store decomposed model.

- Baseline Evaluation: Measure initial metrics.

- Layer Compression: Rank search for optimal compression.

- Final Logging: Record results (WandB/local files).

- Example Directory Structure

---

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
