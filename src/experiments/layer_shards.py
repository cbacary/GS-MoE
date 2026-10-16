import glob
import json
import os
import subprocess
import sys
import threading

import torch

from src.adapters.compression_config import CompressionConfig

from src.utils.logger import logger

CONFIG_NAME = "config.json"


def is_shard_worker(config) -> bool:
    return getattr(config, "shard_rank", -1) >= 0


def shard_dir() -> str:
    d = os.path.expandvars("$TMPDIR/compression_shards")
    os.makedirs(d, exist_ok=True)
    return d


def shard_metrics_path(rank: int) -> str:
    return os.path.join(shard_dir(), f"shard_{rank}_metrics.pt")


def shard_layer_ranges(n_layers: int, n_shards: int) -> list[tuple[int, int]]:
    """
    Split [0, n_layers) into n_shards contiguous ranges, remainder spread over the
    first shards. 32 layers / 2 shards -> [(0, 16), (16, 32)].
    """
    n_shards = max(1, min(n_shards, n_layers))
    per, remainder = divmod(n_layers, n_shards)

    ranges, start = [], 0
    for i in range(n_shards):
        end = start + per + (1 if i < remainder else 0)
        ranges.append((start, end))
        start = end

    return ranges


def source_model_n_layers(model_dir: str) -> int:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    n_layers = (
        getattr(cfg, "n_layer", None)
        or getattr(cfg, "num_hidden_layers", None)
        or getattr(cfg, "num_layers", None)
    )
    if not n_layers:
        raise ValueError(f"Could not determine the layer count of {model_dir}")

    return int(n_layers)


def visible_devices() -> list[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        devices = [d.strip() for d in env.split(",") if d.strip()]
    else:
        devices = [str(i) for i in range(torch.cuda.device_count())]

    return devices or ["0"]


def _clear_previous_run(n_shards: int):
    stale = [shard_metrics_path(rank) for rank in range(n_shards)]
    stale += [os.path.join(shard_dir(), f"shard{rank}_complete") for rank in range(n_shards)]
    stale += glob.glob(os.path.expandvars("$TMPDIR/compressed_layers/layer_*_fact"))

    for path in stale:
        if os.path.exists(path):
            os.remove(path)


def _pump_output(proc: subprocess.Popen, rank: int, log_path: str):
    with open(log_path, "w") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(f"[shard {rank}] {line}", end="", flush=True)


def launch_layer_shards(config: CompressionConfig, n_layers: int) -> list[tuple[int, int]]:
    devices = visible_devices()
    ranges = shard_layer_ranges(n_layers, config.shards)

    if len(devices) < len(ranges):
        logger.warning(
            f"{len(ranges)} shards but only {len(devices)} visible GPU(s); "
            "shards will share devices and may OOM."
        )

    _clear_previous_run(len(ranges))

    config_path = os.path.join(shard_dir(), CONFIG_NAME)
    with open(config_path, "w") as f:
        json.dump({k: v for k, v in config.to_dict().items() if not k.startswith("_")}, f)

    procs: list[subprocess.Popen] = []
    pumps: list[threading.Thread] = []

    try:
        for rank, (start, end) in enumerate(ranges):
            device = devices[rank % len(devices)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            env["PYTHONUNBUFFERED"] = "1"

            cmd = [
                sys.executable,
                "-m",
                "src.experiments.run_compression_parallel",
                "--shard_config",
                config_path,
                "--shards",
                str(len(ranges)),
                "--shard_rank",
                str(rank),
                "--shard_start_layer",
                str(start),
                "--shard_end_layer",
                str(end),
            ]

            logger.info(f"Launching shard {rank}: layers [{start}, {end}) on GPU {device}")
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            procs.append(proc)

            pump = threading.Thread(
                target=_pump_output,
                args=(proc, rank, os.path.join(shard_dir(), f"shard_{rank}.log")),
                daemon=True,
            )
            pump.start()
            pumps.append(pump)

        codes = [proc.wait() for proc in procs]
    except BaseException:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
        raise

    for pump in pumps:
        pump.join(timeout=30)

    failed = [rank for rank, code in enumerate(codes) if code != 0]
    if failed:
        raise RuntimeError(
            f"Compression shard(s) {failed} failed (exit codes {codes}). "
            f"Logs are in {shard_dir()}"
        )

    logger.info(f"All {len(ranges)} compression shards finished: {ranges}")
    return ranges


def save_shard_metrics(metrics: dict, rank: int):
    torch.save(metrics, shard_metrics_path(rank))


def merge_shard_metrics(adapter, n_shards: int):
    for rank in range(n_shards):
        path = shard_metrics_path(rank)
        if not os.path.exists(path):
            logger.warning(f"No metrics from shard {rank} at {path}")
            continue

        shard_metrics = torch.load(path, map_location="cpu", weights_only=False)

        adapter.metrics.setdefault("latent_moe_metrics", {}).update(
            shard_metrics.get("latent_moe_metrics", {})
        )

        for key in ("tokens_per_exp", "exp_routing_frequency"):
            rows = shard_metrics.get(key)
            if rows is None:
                continue
            merged = adapter.metrics.setdefault(key, [[0 for _ in row] for row in rows])
            for layer_idx, row in enumerate(rows):
                if any(row):
                    merged[layer_idx] = row

        for key in ("smoothing", "nys_keep_ratios", "fact_keep_ratios"):
            if key in shard_metrics and key not in adapter.metrics:
                adapter.metrics[key] = shard_metrics[key]
