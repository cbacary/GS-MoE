import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from src.utils.logger import logger

"""
precision to use for nearly all operations
"""
dtype_p = torch.float64
"""
The final type to cast all the weights back down to
"""
dtype_f = torch.bfloat16

"""
True if we will be using multiple gpu's to compress
 - calibs / activations corrs. will be on calib_device.
 - Weights will be compressed on the 2nd gpu, while the first GPU holds the entire model
"""
# parallel = False
parallel = torch.cuda.device_count() > 1
conservative = True
d1 = "cuda:0"
d2 = "cuda:1" if parallel else "cuda:0"
# calib_device = "cpu"
calib_device = "cuda:1" if parallel else "cuda:0"


def start_memory_usage_worker():

    import psutil
    import time
    import threading

    def print_memory_usage():
        process = psutil.Process(os.getpid())
        while True:
            mem_gb = process.memory_info().rss / (1024**3)
            sys_mem = psutil.virtual_memory()

            with open(".mem-usage", "w") as f:
                f.write(
                    f"[Monitor] Process RAM: {mem_gb:.2f} GB\nSystem RAM: {sys_mem.percent}% used"
                )
                if mem_gb > 60:
                    f.write(
                        "\n\nCRITICAL WARNING: Process nearing 64GB RAM limit! Crash imminent.\n"
                    )

            time.sleep(1)

    monitor_thread = threading.Thread(target=print_memory_usage, daemon=True)
    monitor_thread.start()

    return monitor_thread


def load_model(model_name: str, device: int = 0):
    """
    Loads the official model.
    """
    logger.info(f"Loading model from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True, torch_dtype="auto"
    )

    logger.info(f"params.dtype = {next(model.parameters()).dtype}")
    logger.info(f"model.config.dtype = {model.config.torch_dtype}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token_id
        logger.info("No pad_token found. Set pad_token = eos_token_id.")

    return model, tokenizer, model.config


def save_compressed_model(
    adapter,
    rotary_masks: torch.Tensor | None,
    save_dir: str,
    source_model_name: str,
):
    import shutil

    model, tokenizer = adapter.model, adapter.tokenizer

    arch = adapter.arch
    if arch == "opt":
        rebuild_path = "./src/patchers/opt_rebuild.py"
    elif arch == "llama":
        rebuild_path = "./src/patchers/llama_rebuild.py"
    elif arch == "mixtral":
        rebuild_path = "./src/patchers/mixtral_rebuild.py"
    elif arch == "qwen2_moe":
        rebuild_path = "./src/patchers/qwen2_moe_rebuild.py"
    elif arch == "deepseek":
        rebuild_path = "./src/patchers/deepseek_rebuild.py"
        if model.config.model_type == "deepseek_v2":
            rebuild_path = "./src/patchers/deepseekv2_rebuild.py"
    else:
        raise Exception("Cannot save compressed model ... no compressed model definition")

    os.makedirs(save_dir, exist_ok=True)

    if rotary_masks is not None:
        mask_path = os.path.abspath(os.path.join(save_dir, "rotary_masks.pt"))
        model.config.mask_path = mask_path
    else:
        model.config.mask_path = None

    model.config.torch_dtype = "bfloat16"
    model.config.dtype = "bfloat16"

    logger.info(f"params.dtype = {next(model.parameters()).dtype}")
    logger.info(f"model.config.dtype = {model.config.torch_dtype}")

    logger.info(f"Saving compressed model to {save_dir}")
    model.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)

    import json

    if rotary_masks is not None:
        torch.save(rotary_masks, mask_path)
    shutil.copy(rebuild_path, save_dir)
    shutil.copy("./src/patchers/latent_linear.py", save_dir)
    with open(os.path.join(save_dir, "tokenizer_source.txt"), "w") as f:
        f.write(source_model_name.strip())

    logger.info(f"✔ Model, tokenizer, and tokenizer_source.txt saved to {save_dir}")

    cfg_path = os.path.join(save_dir, "config.json")
    with open(cfg_path, "r") as f:
        written_config = json.load(f)
        written_config["auto_map"] = model.config.auto_map

    if arch == "deepseek":
        written_config["auto_map"] = {}
        if model.config.model_type == "deepseek_v2":
            written_config["auto_map"][
                "AutoModelForCausalLM"
            ] = "deepseekv2_rebuild.DeepseekV2ForCausalLM"
            written_config["auto_map"]["AutoConfig"] = "deepseekv2_rebuild.DeepseekV2Config"
            written_config["auto_map"]["AutoModel"] = "deepseekv2_rebuild.DeepseekV2Model"

        else:
            written_config["auto_map"][
                "AutoModelForCausalLM"
            ] = "deepseek_rebuild.DeepseekForCausalLM"
            written_config["auto_map"]["AutoConfig"] = "deepseek_rebuild.DeepseekConfig"
            written_config["auto_map"]["AutoModel"] = "deepseek_rebuild.DeepseekModel"
        # written_config["auto_map"] = {}
        # written_config["auto_map"]["AutoModelForCausalLM"] = "deepseek_rebuild.DeepseekForCausalLM"
        # written_config["auto_map"]["AutoConfig"] = "deepseek_rebuild.DeepseekConfig"
        # written_config["auto_map"]["AutoModel"] = "deepseek_rebuild.DeepseekModel"

    with open(cfg_path, "w") as f:
        json.dump(written_config, f, indent=2)


def reload_compressed_model(
    model_dir: str,
    device="cuda:0",
    tokenizer_source: str = "",
):
    """
    Better just to use this always. As long as you pass tokenizer_source (which is just the name of the model)
    it will always work compressed or not.
    """
    logger.info(f"Reloading model from: {model_dir}")
    if not tokenizer_source:
        tokenizer_source_path = os.path.join(model_dir, "tokenizer_source.txt")

        if not os.path.exists(tokenizer_source_path):
            raise FileNotFoundError("Missing tokenizer_source. Cannot reload tokenizer.")

        with open(tokenizer_source_path, "r") as f:
            tokenizer_source = f.read().strip()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    target_device_map = device if device else "auto"

    print(f"model loaded on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map=target_device_map,
        torch_dtype="auto",
    )
    # model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("No pad_token found. Set pad_token = eos_token_id.")

    logger.info(f"✔ Loaded model with device_map={target_device_map}.")
    logger.info(f"✔ Reloaded compressed model successfully.")

    model.eval()
    return model, tokenizer


def rebuild_in_memory_meta(adapter, model_cls, config=None):
    src = adapter.model
    cfg = adapter.model.config

    with torch.device("meta"):
        new_model = model_cls(cfg)

    missing, unexpected = new_model.load_state_dict(src.state_dict(), strict=False, assign=True)
    assert not unexpected, f"unexpected keys: {unexpected}"

    new_model.tie_weights()

    src_buffers = dict(src.named_buffers())
    still_meta = []
    for name, buf in list(new_model.named_buffers()):
        if buf.is_meta:
            if name in src_buffers and not src_buffers[name].is_meta:
                *path, attr = name.split(".")
                mod = new_model.get_submodule(".".join(path)) if path else new_model
                mod._buffers[attr] = src_buffers[name].clone()
            else:
                still_meta.append(name)

    leftover = [n for n, t in new_model.named_parameters() if t.is_meta] + still_meta
    assert not leftover, f"still on meta: {leftover}"

    new_model.eval()
    return new_model
