# GS-MoE

GS-MoE is a Mixture-of-Experts Large Langguage Model compression method. Preprint will be available on arxiv.

## 1. Set-up

Install with [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12.6
uv sync
source .venv/bin/activate
```

Implementation was tested on NVIDIA B200 using CUDA 12.8. You may need to change `tool.uv.sources` and `tool.uv.index` to your needed CUDA version/wheel.

### Running Different Architectures

To run **Qwen** or **Mixtral**, uncomment the relevant line in `pyproject.toml`

```toml
    # "transformers[torch]>=4.57,<4.57.6", ### For Qwen2-57B-A14B
    "transformers[torch]>=4.54.0,<5.0.0", ### For Mixtral-8x7B
    # "transformers[torch]>=4.53.0,<4.54.0", ### For deepseek
```

Then:

```bash
uv sync
```

### Other Setups
Some paths are hard-coded to `$TMPDIR`, so set:

```bash
export TMPDIR=desired/tmp/path
```

in the shell file before running.

Make sure the storage device where `$TMPDIR` is located has enough space. 
`$TMPDIR/compressed_layers/` stores the compressed layers before reconstructing to runnable model. Optional `$TMPDIR/activations` saves activations (around 115GB for Mixtral-8x7b).
## 2. Run compression

Scripts are found in `scripts/`. Run them with:

```bash
bash scripts/mixtral.sh    # Mixtral-8x7B     (switch transformers version first, see above)
bash scripts/qwen.sh       # Qwen2-57B-A14B   (switch transformers version first, see above)
bash scripts/deepseek.sh   # DeepSeek-MoE-16B 
```
Example run:
```bash
MODEL="mistralai/Mixtral-8x7B-v0.1"
OUTPUT_DIR="$TMPDIR/compressed_output"
N_SHARDS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) # gets number of GPUs available

python -m src.experiments.run_compression_parallel \
  --shards $N_SHARDS \
  --model "$MODEL" \
  --tokenizer_src "$MODEL" \
  --device 0 \
  --calib_size 512 \
  --calibs_batch_size 32 \
  --layers_per_compression_step 4 \
  --output_dir "$OUTPUT_DIR" \
  --note "Benchmarks" \
  --compression_ratio 0.6214 \
  --nystrom_layer_smoothing 10.0 \
  --factorization_layer_smoothing 5 \
  --nys_to_fact_ratio 0.85 \
  --nystrom_ridge 0.01 \
  --Ex_ridge 1.0 \
  --dataset "alpaca" \
  --n_shared_projs 1 \
  --save \
  --benchmark \
  --compressed_acts_save_path "$TMPDIR/activations" \
```

Runs GS-MoE on Mixtral-8x7B reducing total parameters by 60% with `$N_SHARDS` subprocesses (1 proc/GPU) each calibrating/compressing `layers_per_compression_step` at a time and creates a single shared projection for each layer of experts.  Uses 512 calibration samples with batch size 32. Runs/saves benchmarks results to `benchmark_results.csv`. Saves/loads per-layer activations from path (Mixtral ~115GB).

### Memory

Each GPU owns a copy of the model to improve compression time. If one GPU does not fit or barely fits the model, simplest will be to load only active/running layers to GPU, and or update activations to be stored on CPU or spread across multiple GPUs.

- `--layers_per_compression_step` Lower if you OOM
- `--calibs_batch_size` Lower if you OOM during calibration
- `--shards` splits the layers across `shards` GPUs; one process per GPU. 
## 3. Outputs

- `<output_dir>/model/`: the compressed model (must set `--save`).
- `./benchmark_results.csv`: adds eval results as row to `benchmark_results.csv` if `--benchmark` is set.

## 4. All options

Every field in [src/adapters/compression_config.py](src/adapters/compression_config.py) is a
command-line flag (`--<name> <value>`). Boolean options are plain flags (`--save`).

**Model and output**

| Option | Default | Description |
|---|---|---|
| `--model` | `mistralai/Mixtral-8x7B-v0.1` | HF id or local path of the model to compress. |
| `--tokenizer_src` | `mistralai/Mixtral-8x7B-v0.1` | Tokenizer path. Set to the original HF id. |
| `--output_dir` | `compressed_output` | Compressed model save location. Requires `--save` |
| `--archive` | `""` | Copies `output_dir` to `<archive>/` Requires `--save`. |
| `--benchmark` | off | Run zero-shot + ppl saves to `benchmark_results.csv` |
| `--seed` | `1234` | Not actually used everywhere. `1234` is hardcoded often. |

**Compression budget**

| Option | Default | Description |
|---|---|---|
| `--compression_ratio` | `0.5` | Fraction of expert parameters to remove (see the table above). |
| `--nys_to_fact_ratio` | `0.6` | Nyström/factorization split: `1.0` = all Nyström, `0.0` = all factorization. |
| `--nystrom_layer_smoothing` | `1.0` | Smoothing of the per-layer Nyström budget. Higher values give a more uniform allocation. |
| `--factorization_layer_smoothing` | `1.0` | Same, for the per-layer factorization budget. |
| `--n_shared_projs` | `1` | Number of shared factorization projections per layer. |

**Regularization**

| Option | Default | Description |
|---|---|---|
| `--nystrom_ridge` | `0.01` | Ridge term for the Nyström reconstruction. |
| `--Ex_ridge` | `0.0` | Ridge term on the expert input covariance, scaled by its trace. The scripts use `1.0`. |

**Calibration**

| Option | Default | Description |
|---|---|---|
| `--dataset` | `alpaca` | Calibration dataset: `alpaca`, `wikitext` or `c4`. |
| `--calib_size` | `512` | Number of calibration samples. |
| `--calibs_batch_size` | `16` | Batch size for calibration forward passes. |
| `--layers_per_compression_step` | `4` | Layers calibrated and compressed together. Lower to save GPU memory. |
| `--compressed_acts_save_path` | `""` | Takes path. Caches the calibration covariances in this directory to reuse on later runs with the same layer ranges. Must clear if model, dataset or calibration size changed. Stored in FP32 and only upper triangle. Saved Mixtral-8x7b activations are around 115GB|

**Hardware**

| Option | Default | Description |
|---|---|---|
| `--shards` | `1` | Number of GPUs to split the layers across, one process each. Each process holds a full copy of the model. |

Don't touch `shard_rank`, `shard_start_layer`, `shard_end_layer`, `shard_config`
