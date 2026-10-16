import threading
import math
import time

import numpy as np
import torch
from datasets import load_dataset
import datasets
from tqdm import tqdm
import os

import pandas as pd

from src.adapters.model_adapter import MLPComponents, ModelAdapter

from concurrent.futures import ThreadPoolExecutor

from src.adapters.moe_adapter import MoEAdapter
from src.utils.model import d1, dtype_p
import random

from lm_eval.tasks import TaskManager, get_task_dict

from src.utils.logger import logger


tmpdir = os.environ.get("TMPDIR")


@torch.no_grad()
def benchmark_and_save_parallel(
    adapter: MoEAdapter, kl_loss: float = None, save_path=None, model2_path=None
):

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    save_path = adapter.config.output_dir if save_path is None else save_path

    # tasks = ["openbookqa", "hellaswag", "arc_challenge", "arc_easy", "piqa", "winogrande", "mathqa"]
    tasks = [
        "openbookqa",
        "hellaswag",
        "arc_challenge",
        "arc_easy",
        "piqa",
        "winogrande",
        "mathqa",
        "boolq",
    ]
    tasks_gpu0 = ["hellaswag"]
    tasks_gpu1 = [
        "openbookqa",
        "arc_challenge",
        "arc_easy",
        "piqa",
        "winogrande",
        "mathqa",
        "boolq",
    ]
    timestamp = adapter.metrics["RunName"]
    uniformity = adapter.config.uniformity
    order = adapter.config.order
    method = f"{adapter.config.order} ({uniformity})"
    if adapter.config.nys_to_fact_ratio == 0.0:
        method = "pure_fact"
    elif adapter.config.nys_to_fact_ratio == 1.0:
        method = "nystrom"
    compression_ratio = adapter.config.compression_ratio

    from src.eval import compute_perplexity

    wikitext_ppl = compute_perplexity(
        adapter.model, adapter.tokenizer, 2, dataset="wikitext", adapter=adapter
    )
    c4_ppl = compute_perplexity(adapter.model, adapter.tokenizer, 2, dataset="c4", adapter=adapter)
    ptb_ppl = compute_perplexity(
        adapter.model, adapter.tokenizer, 2, dataset="ptb", adapter=adapter
    )

    print(f"wt_ppl = {wikitext_ppl}")
    print(f"c4_ppl = {c4_ppl}")
    print(f"ptb_ppl = {ptb_ppl}")

    from pathlib import Path

    def get_dir_size_pathlib(path="."):
        root = Path(path)
        # rglob('*') recursively finds all files and subdirectories
        return sum(f.stat().st_size for f in root.rglob("*") if f.is_file() and not f.is_symlink())

    columns = [
        "model",
        "compression_ratio",
        "method",
        "calib_set",
        "nys_to_fact_ratio",
        "actual_size-GB",
        "timestamp",
        "wikitext-ppl",
        "c4-ppl",
        "ptb-ppl",
        # "alpaca-ppl"
    ]
    for task in tasks:
        columns.append(task)
    columns.append("average")
    for task in tasks:
        columns.append(f"{task}_norm")
    columns.append("average_norm")
    row = {
        "model": adapter.config.model,
        "compression_ratio": compression_ratio,
        "actual_size-GB": f"{get_dir_size_pathlib(adapter.config.output_dir) / (1000**3):.2f}",
        "method": method,
        "calib_set": adapter.config.dataset,
        "timestamp": timestamp,
        "wikitext-ppl": wikitext_ppl,
        "c4-ppl": c4_ppl,
        "ptb-ppl": ptb_ppl,
        "nys_to_fact_ratio": (
            adapter.config.nys_to_fact_ratio
            if not adapter.config.dynamic_fact_nys_ratio
            else "dynamic per-layer"
        ),
        "nys_layer_smoothing": adapter.config.nystrom_layer_smoothing,
        "fact_layer_smoothing": adapter.config.factorization_layer_smoothing,
        "n_calibs": adapter.config.calib_size,
        "nys_ridge": adapter.config.nystrom_ridge,
        "Ex_ridge": adapter.config.Ex_ridge,
        "n_shared_projs": adapter.config.n_shared_projs,
        "note": adapter.config.note,
    }

    for c in list(row.keys()):
        if c not in columns:
            columns.append(c)

    import copy

    results = {}
    errors = []

    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    task_manager = TaskManager()
    task_dict = get_task_dict(tasks_gpu0 + tasks_gpu1, task_manager)

    model_path = os.path.expandvars(os.path.join(adapter.config.output_dir, "model"))

    import gc

    def worker(model, tokenizer, device, task_subset, load_gpu=False):

        if model is None:
            m_path = model2_path if model2_path is not None else model_path
            model, tokenizer = reload_compressed_model(
                m_path, device="cuda:1", tokenizer_source=adapter.config.tokenizer_src
            )
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            device=device,
            batch_size="auto:2",
            max_batch_size=128,
        )

        res = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task_dict[name] for name in task_subset],
            num_fewshot=0,
            task_manager=task_manager,
        )

        local = {}
        for task in task_subset:
            tr = res["results"].get(task, {})
            local[task] = tr.get("acc,none")
            local[f"{task}_norm"] = tr.get("acc_norm,none")
        # results[key] = local

        if load_gpu:
            del model, tokenizer

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        return local

    if adapter.config.benchmark:

        from src.utils.model import reload_compressed_model

        # mid = len(tasks) // 2
        # tasks_gpu0, tasks_gpu1 = tasks[:mid], tasks[mid:]

        n_workers = adapter.config.shards

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            if n_workers > 1:
                f0 = ex.submit(worker, adapter.model, adapter.tokenizer, "cuda:0", tasks_gpu0)
                f1 = ex.submit(worker, None, adapter.tokenizer, "cuda:1", tasks_gpu1, load_gpu=True)
                row.update(f0.result())
                row.update(f1.result())
            else:
                f0 = ex.submit(
                    worker, adapter.model, adapter.tokenizer, "cuda:0", tasks_gpu0 + tasks_gpu1
                )
                row.update(f0.result())

        # t0 = threading.Thread(
        #     target=worker,
        #     args=(adapter.model, adapter.tokenizer, "cuda:0", tasks_gpu0, "gpu0"),
        # )
        # t1 = threading.Thread(
        #     target=worker,
        #     args=(model2, adapter.tokenizer, "cuda:1", tasks_gpu1, "gpu1"),
        # )
        # t0.start()
        # t1.start()
        # t0.join()
        # t1.join()

    df = pd.DataFrame([row], columns=columns)
    print(df.to_string())
    benchmark_results = df.to_dict()
    adapter.metrics["benchmarks"] = benchmark_results

    run_id = append_row_aligned(df, "./benchmark_results.csv")

    # csv_path = "./benchmark_results.csv"
    # df.to_csv(
    #     csv_path,
    #     mode="a",
    #     header=not os.path.exists(csv_path),
    #     index=False,
    # )
    return run_id


INDEX_NAME = "run_id"


def append_row_aligned(df: pd.DataFrame, csv_path: str) -> int:
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path, dtype=str, keep_default_na=False, index_col=INDEX_NAME)
        old.index = old.index.astype("int64")
    else:
        old = pd.DataFrame(index=pd.Index([], dtype="int64", name=INDEX_NAME))

    start = int(old.index.max()) + 1 if len(old) else 0
    df = df.copy()
    df.index = pd.RangeIndex(start, start + len(df), name=INDEX_NAME)

    all_cols = list(old.columns) + [c for c in df.columns if c not in old.columns]
    combined = pd.concat(
        [old.reindex(columns=all_cols), df.reindex(columns=all_cols)],
        verify_integrity=True,
    )

    tmp = f"{csv_path}.tmp"
    combined.to_csv(tmp, index=True, index_label=INDEX_NAME)
    os.replace(tmp, csv_path)

    return int(df.index[-1])
