"""
python -m src.experiments.run_compression_parallel --shards 2 ... (same flags as run_compression)
"""

from src.eval import compute_perplexity
import shutil
import os
import json
import gc

from dataclasses import fields

import torch
import transformers.modeling_utils as tm

from src.utils.kl_divergence import compute_forward_kl, create_teacher_logits
from src.eval import load_calibration_texts
from src.utils.model import (
    start_memory_usage_worker,
    reload_compressed_model,
    save_compressed_model,
)
from src.adapters.compression_config import CompressionConfig

from src.adapters.moe_adapter import MoEAdapter
from src.utils.logger import logger

from src.eval import compute_perplexity

ALPACA = False

tmpdir = os.environ.get("TMPDIR")

from src.eval import eff_eval


@torch.no_grad()
def main(
    trial=None,
    config: CompressionConfig = None,
    csv_name=None,
):

    model, tokenizer, adapter = None, None, None
    global decay_scores

    if not config:
        config = CompressionConfig.from_args()

    # start_memory_usage_worker()

    print(config.to_dict())

    metrics_path = os.path.join(config.output_dir, "metrics.json")

    method = config.order
    comp_ratio = config.compression_ratio

    model, tokenizer = reload_compressed_model(config.model, tokenizer_source=config.tokenizer_src)
    adapter = MoEAdapter.from_model(model, tokenizer, config=config)

    wt = compute_perplexity(model, tokenizer, 0, bs=16, dataset="wikitext", adapter=adapter)
    c4 = compute_perplexity(model, tokenizer, 0, bs=16, dataset="c4", adapter=adapter)
    ptb = compute_perplexity(model, tokenizer, 0, bs=16, dataset="ptb", adapter=adapter)

    print(f"wt = {wt}, c4 = {c4}, ptb = {ptb}")

    result = eff_eval(adapter)

    print(result)


main()
