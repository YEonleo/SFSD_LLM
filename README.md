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
- **Weights & Biases (wandb)** for logging (optional, you can remove the wandb parts if you prefer)
- Other libraries such as `numpy`, `pandas`, `tqdm`, `json`, etc.

Installation example (adjust as needed):

pip install torch torchvision torchaudio 
pip install transformers datasets 
pip install wandb 
pip install numpy pandas tqdm

Additionally, the script uses local modules (e.g., `preprocess.py`, `dataset_ppl.py`, `layers.py`, `lm_eval`), so ensure these files or folders are present in the same directory or in your Python path.

---

## Usage

### 1) How to Run

python your_script_name.py [OPTIONS]

- Replace `your_script_name.py` with the filename containing `main()` (e.g., `sfds_main.py`).
- Use `--help` or `-h` to see available arguments.

### 2) Key Arguments
Below are some important `argparse` parameters (run `--help` for the full list):

| Argument               | Type   | Default                                       | Description                                                                                                                      |
|------------------------|--------|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `--layers`            | str    | `"o_proj,q_proj,v_proj,k_proj,gate_proj,up_proj,down_proj"` | Comma-separated keywords for layers to decompose, e.g. `o_proj,q_proj`.                                                          |
| `--dataset`           | str    | `"piqa"`                                      | Name of the dataset for evaluation (depends on your local `get_dataset()` implementation).                                       |
| `--batch_size`        | int    | `512`                                         | Batch size                                                                                                                       |
| `--seq_len`           | int    | `128`                                         | Token sequence length                                                                                                            |
| `--log_path`          | str    | `"surgical_logs.txt"`                        | Path to the log file                                                                                                             |
| `--algo`              | str    | `"eigen"`                                     | Decomposition algorithm (e.g., `eigen` or `svd`)                                                                                 |
| `--weights_name`      | str    | `"decomposed_model_mistral_combination"`      | Name (without extension) for the saved decomposed model checkpoint                                                               |
| `--model`             | str    | `"meta-llama/Llama-2-7b-hf"`                 | The Hugging Face model name to decompose (e.g., a Llama model)                                                                   |
| `--cache_dir`         | str    | `"../cache_dir"`                              | Cache directory for model downloads                                                                                              |
| `--delta`             | float  | `0.0`                                         | Allowed accuracy (or metric) drop during compression                                                                             |
| `--start_layer`       | int    | `28`                                          | Index of the first layer to compress (useful for only compressing higher layers)                                                 |
| `--base_model`        | str    | `"decomposed_model_mistral_combination.pt"`   | Path to a decomposed model file for evaluation                                                                                   |
| `--max_ratio`         | float  | `1.0`                                         | Upper bound for compression ratio

### 3) Example Command

python sfds_main.py
--model meta-llama/Llama-2-7b-hf
--dataset piqa
--layers "down_proj,up_proj"
--max_ratio 0.8
--delta 0.01
--start_layer 20
--weights_name "my_decomposed_model"
--batch_size 256
--seq_len 128


When it finishes, it might produce logs in `my_decomposed_model_maxratio_0.8.pt` (or a similar filename) based on the script’s `save_decomposed_model()` function.

---

## Workflow Summary

1. **Load the Base Model**  
   - `AutoModelForCausalLM.from_pretrained(args.model)` downloads/loads the model and possibly places it on CPU or GPU.

2. **Identify Decomposable Layers**  
   - The script’s `get_decomposable_layers()` finds all `nn.Linear` modules whose names contain any of the substrings in `args.layers`.
   - It calculates a theoretical max rank for each layer.

3. **Inject Decomposable Modules**  
   - The original `nn.Linear` layers are replaced by custom modules (e.g., from `ModuleInjection.make_decomposable`) that allow SVD/eigen-based decomposition.

4. **Forward Test**  
   - Loads data (from the chosen dataset) into a DataLoader.
   - Runs a forward pass to ensure everything works without errors.

5. **Save the Decomposed Model**  
   - The partially decomposed model is saved as a `.pt` file and/or with `model.save_pretrained()`.

6. **Baseline Evaluation**  
   - Evaluates the model on metrics such as accuracy, perplexity, or specific task-based scores (e.g., ROUGE, exact match).

7. **Compression (Rank Search)**  
   - Iterates over each layer in reverse order or a specified range.
   - Tries multiple candidate ranks (like [1, 10, 20…]) and tests performance.
   - Retains the highest compression that stays within the `--delta` threshold or a similar condition.

8. **Final Logging**  
   - Logs results to `wandb` and/or local files (`args.log_path`).
   - May periodically save partial/compressed checkpoints.
   - Ends with `wandb.finish()` if wandb is used.

---

## Example Directory Structure

A typical layout could be:

project/ ├── sfds_main.py # main script (the one with main()) ├── preprocess.py # data preprocessing ├── dataset_ppl.py # dataset utilities ├── layers.py # custom layer decomposition logic ├── lm_eval/ │ ├── evaluator.py # LM evaluation logic │ └── models/ │ └── huggingface.py # HFLM class, etc. ├── logs/ # optional directory for logs/checkpoints └── README.md # this file


---

## Notes and Caveats

- **GPU Memory**: A 7B model typically requires a significant amount of GPU memory. Repeated decompositions and evaluations can increase usage.
- **Dataset Download**: If using Hugging Face `datasets`, the first run needs internet access to download. If you have a custom `get_dataset()` function, configure it as needed.
- **WandB**: The script uses `wandb.init(...)`, so you need a Weights & Biases account or token configured. Remove or comment out the wandb code if you don’t want to use it.
- **License**: For models like Llama 2, check Meta’s license. Commercial use may require additional permissions or conditions.

Feel free to adjust and expand this README to match your setup and specific requirements.
