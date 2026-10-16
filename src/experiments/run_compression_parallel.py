import time

time.sleep(1.5)  # here cause sometimes rsync hasn't completed over the network yet.

from src.eval import compute_perplexity
import shutil
import os
import json
import gc

from dataclasses import fields

import torch
import transformers.modeling_utils as tm

from src.eval import load_calibration_texts
from src.utils.model import (
    start_memory_usage_worker,
    reload_compressed_model,
    save_compressed_model,
)
from src.adapters.compression_config import CompressionConfig

from src.adapters.moe_adapter import MoEAdapter

from src.experiments.layer_shards import (
    is_shard_worker,
    launch_layer_shards,
    merge_shard_metrics,
    save_shard_metrics,
    source_model_n_layers,
)
from src.utils.benchmark import benchmark_and_save_parallel
from src.utils.logger import logger

ALPACA = False

tmpdir = os.environ.get("TMPDIR")


## Following forces torch to print exact parameter name where crash.
def _debug_load_param(*args, **kwargs):
    try:
        return _orig_load_param(*args, **kwargs)
    except Exception as e:
        print("\n" + "!" * 60)
        print("🚨 CRASH DURING PARAMETER LOADING 🚨")

        param_name = kwargs.get("param_name", kwargs.get("tensor_name", None))
        if not param_name and len(args) > 1:
            param_name = args[1]

        print(f"Failed on global parameter name: {param_name}")
        print(f"Original Error: {e}")
        print("!" * 60 + "\n")
        raise


_orig_load_param = tm._load_parameter_into_model
tm._load_parameter_into_model = _debug_load_param


def source_model(config: CompressionConfig) -> str:
    return config.model


def load_shard_config(config: CompressionConfig) -> CompressionConfig:
    with open(config.shard_config, "r") as f:
        parent = json.load(f)

    init_fields = {f.name for f in fields(CompressionConfig) if f.init}
    merged = {k: v for k, v in parent.items() if k in init_fields}
    merged.update(
        shards=config.shards,
        shard_rank=config.shard_rank,
        shard_start_layer=config.shard_start_layer,
        shard_end_layer=config.shard_end_layer,
        shard_config=config.shard_config,
    )

    return CompressionConfig(**merged)


@torch.no_grad()
def run_shard(config: CompressionConfig):
    rank, start, end = config.shard_rank, config.shard_start_layer, config.shard_end_layer

    from src.experiments.layer_shards import shard_dir

    if rank == 1:
        import time

        time.sleep(5)

    s = shard_dir()
    other_shard_rank = 0 if rank == 1 else 1
    shard_complete_path = f"{s}/shard{rank}_complete"
    other_shard_complete_path = f"{s}/shard{other_shard_rank}_complete"
    if os.path.exists(shard_complete_path):
        os.remove(shard_complete_path)
    # if os.path.exists(other_shard_complete_path):
    #     os.remove(other_shard_complete_path)

    start_memory_usage_worker()

    logger.info(
        f"Shard {rank}: compressing layers [{start}, {end}) on "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    model, tokenizer = reload_compressed_model(
        source_model(config),
        tokenizer_source=config.tokenizer_src,
    )
    adapter = MoEAdapter.from_model(model, tokenizer, config=config)

    adapter.calibs = load_calibration_texts(
        calib_size=config.calib_size,
        model=adapter.model,
        tokenizer=adapter.tokenizer,
        batch_size=config.calibs_batch_size,
        dataset=config.dataset,
    )

    adapter.metrics["args"] = config.to_dict()
    adapter.metrics["note"] = config.note

    # if rank == 0:
    # create_teacher_logits(adapter, dataset="c4")

    from src.experiments.combined import combined_compression

    combined_compression(
        adapter,
        compression_ratio=config.compression_ratio,
        trial=None,
        save_dir=os.path.expandvars(os.path.join(config.output_dir, "model")),
        start_layer=start,
        end_layer=end,
    )

    with open(shard_complete_path, "w") as f:
        f.write("done")

    other_metrics = os.path.join(tmpdir, "_tmp_metrics.json")
    if rank == 1:
        adapter.save_metrics(path=other_metrics)

    if rank == 0 and adapter.config.save:
        while not os.path.exists(other_shard_complete_path) and adapter.config.shards > 1:
            pass

        adapter.convert_model(os.path.join(tmpdir, "compressed_layers"))
        adapter.patch(nystrom=False, flipped_down=adapter.config.flipped_down)

        metrics_path = os.path.join(config.output_dir, "metrics.json")
        save_dir = os.path.expandvars(os.path.join(config.output_dir, "model"))
        os.makedirs(save_dir, exist_ok=True)
        save_compressed_model(
            adapter,
            None,
            save_dir=save_dir,
            source_model_name=config.model,
        )

        adapter.save_metrics(path=metrics_path)

        if os.path.exists(other_metrics):
            out = metrics_path.split(".json")[0] + "_other.json"
            print(out)
            shutil.copy(other_metrics, out)

        gc.collect()
        torch.cuda.empty_cache()
    elif rank == 0:
        while not os.path.exists(other_shard_complete_path) and adapter.config.shards > 1:
            pass
        adapter.convert_model(os.path.join(tmpdir, "compressed_layers"))
        adapter.patch(nystrom=False, flipped_down=adapter.config.flipped_down)

        from src.utils.model import rebuild_in_memory_meta

        model = rebuild_in_memory_meta(adapter, adapter.get_ModelForCausalLM())
        adapter.model = model

        benchmark_and_save_parallel(adapter)

        # overall = compute_forward_kl(
        #     adapter,
        #     ignore_index=-100,
        #     rows_per_chunk=None,
        #     subtract_teacher_entropy=False,
        #     dataset="c4",
        # )
        # c4_ppl = compute_perplexity(model, tokenizer, 2, dataset="c4", adapter=adapter)
        # wt_ppl = compute_perplexity(model, tokenizer, 2, dataset="wikitext", adapter=adapter)
        # score = (c4_ppl + wt_ppl) / 2.0
        # with open("c4-ppl-vs-kl.csv", "a") as f:
        #     f.writelines([f"{adapter.config.nys_to_fact_ratio},{c4_ppl},{overall}\n"])

        # print(f"wt_ppl = {wt_ppl}")
        # print("c4_ppl = ", c4_ppl)
        # print("wt_ppl = ", wt_ppl)
        # print("ave_ppl = ", score)

        # with open("./opt-result", "a") as f:
        # f.write(
        # f"({wt_ppl:.5f} - wt) | ({c4_ppl:.5f} - c4) | ({score:.5f} - avg) | (N:F - {adapter.config.nys_to_fact_ratio}) | (n_shared_projs = {adapter.config.n_shared_projs}) | (comp_ratio = {adapter.config.compression_ratio})\n"
        # )

    logger.info(f"Shard {rank} finished layers [{start}, {end})")

    del adapter.model, adapter.tokenizer, adapter.calibs
    del model, tokenizer, adapter

    gc.collect()
    torch.cuda.empty_cache()


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

    if config.shard_config:
        config = load_shard_config(config)

    if is_shard_worker(config):
        return run_shard(config)

    # start_memory_usage_worker()

    print(config.to_dict())

    metrics_path = os.path.join(config.output_dir, "metrics.json")
    save_dir = os.path.expandvars(os.path.join(config.output_dir, "model"))
    os.makedirs(save_dir, exist_ok=True)
    print(f"save_dir: {save_dir}")

    method = config.order
    comp_ratio = config.compression_ratio

    n_layers = source_model_n_layers(source_model(config))

    # Every shard is its own process holding its own copy of the model
    launch_layer_shards(config, n_layers)

    run_id = "ID_NA"
    if config.save:
        if adapter:
            del adapter.model, adapter.tokenizer
        del model, tokenizer, adapter

        torch.cuda.empty_cache()
        gc.collect()

        # save_dir = config.model

        model, tokenizer = reload_compressed_model(
            save_dir,
            device="cuda",
            tokenizer_source=config.tokenizer_src,
        )

        adapter = MoEAdapter.from_model(model, tokenizer, config=config)

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                adapter.metrics = json.load(f)

        run_id = benchmark_and_save_parallel(adapter, 0.0, model2_path=save_dir)

        adapter.save_metrics(path=metrics_path)

    archive = config.archive
    if archive:
        # root = "/orange/sgao1/cc22bc.fsu/compressed_models/LatentMoE/"
        root = archive
        method = adapter.config.order
        if adapter.config.nys_to_fact_ratio == 0.0:
            method = "pure_fact"
        elif adapter.config.nys_to_fact_ratio == 1.0:
            method = "nystrom"
        r = f"run{run_id}--{adapter.arch}_{adapter.config.compression_ratio}--{method}--{adapter.config.dataset}--{adapter.metrics['RunName']}"
        dest = os.path.join(root, r)
        src = config.output_dir
        os.makedirs(root, exist_ok=True)
        shutil.move(src, dest)

    # adapter.save_metrics()

    # if adapter:
    #     del adapter.model, adapter.tokenizer, model, tokenizer
    #     del adapter

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    if config.optuna:
        with open("./opt-result", "r") as f:
            c4_ppl = f.readline()
            return float(c4_ppl)


if __name__ == "__main__":
    main()
