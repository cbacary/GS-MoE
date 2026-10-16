import math
import time

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm


from src.adapters.model_adapter import MLPComponents, ModelAdapter

from src.utils.model import d1, dtype_p
import random

from src.utils.logger import logger

DATASET_PATH = "allenai/c4"
DATASET_NAME = "en"

random.seed(1234)
np.random.seed(1234)

ALL_ALPACA_CHUNKS_TRAIN = None
ALL_ALPACA_CHUNKS_TEST = None


def load_calibration_texts(
    calib_size,
    model,
    tokenizer,
    batch_size: int,
    alpaca=False,
    dataset="wikitext",
    split="train",
    seed=1234,
):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if dataset == "alpaca":
        # joined_texts = load_alpaca_texts(tokenizer)
        return load_alpaca_kl_div_calibration(
            tokenizer, calib_size, batch_size, device=d1, split=split
        )
    elif dataset == "wikitext":
        dataset_obj = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        joined_texts = "\n\n".join(dataset_obj["text"])
    elif dataset == "c4":
        dataset_obj = load_dataset(
            "json",
            data_files={
                "train": "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"
            },
        )
        texts = [t for t in dataset_obj["train"]["text"] if len(t.strip()) > 0]
        # chunked = load_c4(tokenizer, texts, calib_size)
        # batches = [
        #     torch.cat(chunked[i : i + batch_size], dim=0).to(device="cuda")
        #     for i in range(0, calib_size, batch_size)
        # ]
        # return batches
        joined_texts = "\n\n".join(texts[:10000])  # Limit to avoid memory issues
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'wikitext' or 'c4'")

    chunked = chunk_text(
        model=model, tokenizer=tokenizer, long_texts=joined_texts, min_threshold=2048
    )

    num_chunks = chunked.shape[0]
    indices = np.random.choice(num_chunks, size=min(int(calib_size), num_chunks), replace=False)

    batches = []
    for i in range(0, len(indices), batch_size):
        batch_ind = indices[i : i + batch_size]
        batch_tensor = chunked[batch_ind]
        batches.append(batch_tensor.to("cuda"))

    return batches


def load_alpaca_kl_div_calibration(
    tokenizer, calib_size, batch_size, device="cuda", split="train", ignore_index=-100
):

    global ALL_ALPACA_CHUNKS_TRAIN, ALL_ALPACA_CHUNKS_TEST

    if split == "train" and ALL_ALPACA_CHUNKS_TRAIN is not None:
        return ALL_ALPACA_CHUNKS_TRAIN
    elif split == "test" and ALL_ALPACA_CHUNKS_TEST is not None:
        return ALL_ALPACA_CHUNKS_TEST

    PROMPT_INPUT = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    )
    PROMPT_NO_INPUT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    )
    seq_len = 2048
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    eos = tokenizer.eos_token_id
    assert eos is not None, "tokenizer has no eos_token_id"

    stream_ids = []
    stream_resp = []
    n_boundary_fallback = 0
    total_prompt_tokens = 0

    itered = ds if split == "train" else reversed(ds)
    # itered = ds
    for ex in itered:
        if ex.get("input", "").strip():
            prompt = PROMPT_INPUT.format(instruction=ex["instruction"], input=ex["input"])
        else:
            prompt = PROMPT_NO_INPUT.format(instruction=ex["instruction"])
        text = prompt + ex["output"]

        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        n_prompt = next(
            (i for i, (a, b) in enumerate(zip(ids, prompt_ids)) if a != b), len(prompt_ids)
        )
        total_prompt_tokens += n_prompt

        if n_prompt != len(prompt_ids):
            n_boundary_fallback += n_prompt != len(prompt_ids)

        ids.append(eos)
        stream_ids.extend(ids)
        stream_resp.extend([False] * n_prompt + [True] * (len(ids) - n_prompt))

    stream_ids = torch.tensor(stream_ids, dtype=torch.long)
    stream_resp = torch.tensor(stream_resp, dtype=torch.bool)

    response_only = split == "test"
    # response_only = False
    if response_only:
        keep = torch.zeros_like(stream_resp)
        keep[:-1] = stream_resp[1:]
    else:
        keep = stream_ids != eos
    labels_stream = stream_ids.masked_fill(~keep, ignore_index)

    seq_len = seq_len - 1

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        seq_len = 2048
        # bos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    n_chunks = stream_ids.numel() // seq_len
    ids_chunks = stream_ids[: n_chunks * seq_len].view(n_chunks, seq_len)
    lab_chunks = labels_stream[: n_chunks * seq_len].view(n_chunks, seq_len)  # [n_chunks, 2048]

    if bos_id is not None:
        bos = torch.full((n_chunks, 1), bos_id, dtype=ids_chunks.dtype)
        ign_bos = torch.full((n_chunks, 1), ignore_index, dtype=lab_chunks.dtype)
        ids_chunks = torch.cat([bos, ids_chunks], dim=1)
        lab_chunks = torch.cat([ign_bos, lab_chunks], dim=1)

    rng = np.random.default_rng(1234)
    take = min(calib_size, n_chunks)
    idx = np.sort(rng.choice(n_chunks, size=take, replace=False))
    ids_chunks = ids_chunks[idx]
    lab_chunks = lab_chunks[idx]

    # batch: 256 / 32 -> 8 batches, matching your wikitext budget
    if split == "train":
        ALL_ALPACA_CHUNKS_TRAIN = [
            (
                ids_chunks[i : i + batch_size].to(device)
                # ids_chunks[i : i + batch_size].to(device),
                # lab_chunks[i : i + batch_size].to(device),
            )
            for i in range(0, take, batch_size)
        ]

        return ALL_ALPACA_CHUNKS_TRAIN
    else:
        ALL_ALPACA_CHUNKS_TEST = [
            (
                ids_chunks[i : i + batch_size].to(device),
                lab_chunks[i : i + batch_size].to(device),
            )
            for i in range(0, take, batch_size)
        ]
        return ALL_ALPACA_CHUNKS_TEST


def chunk_text(model, tokenizer, long_texts: str, min_threshold):
    # following version has only a single BOS token
    # input_ids = tokenizer(long_texts, truncation=False, return_tensors="pt")["input_ids"]

    # max_length = min(2048, model.config.max_position_embeddings)

    # n_chunks = input_ids[0].size(0) // max_length
    # chunked = input_ids[0][: n_chunks * max_length].view(-1, max_length)
    # return chunked

    # BOS token at every chunk
    input_ids = tokenizer(
        long_texts, truncation=False, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    max_length = min(min_threshold, model.config.max_position_embeddings)
    bos = False
    if bos:
        body = max_length - 1
    else:
        body = max_length
    n_chunks = input_ids.size(0) // body
    chunked = input_ids[: n_chunks * body].view(-1, body)

    if not bos:
        return chunked

    bos_id = tokenizer.bos_token_id
    # bos_id = None
    if bos_id is None:
        bos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    bos = torch.full((n_chunks, 1), bos_id, dtype=chunked.dtype)
    return torch.cat([bos, chunked], dim=1)


@torch.no_grad()
def compute_perplexity(
    model, tokenizer, _, bs=16, device="cuda", dataset="wikitext", adapter: ModelAdapter = None
):
    # we hardcode a number of weird things here to make perplexiy comparable to other papers
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    bs = 16
    if dataset == "wikitext":
        if not hasattr(adapter, "wikitext_text"):
            dataset_obj = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n\n".join(dataset_obj["text"])
            adapter.wikitext_text = text
        text = adapter.wikitext_text
    elif dataset == "c4":
        if not hasattr(adapter, "c4_text"):
            dataset_obj = load_dataset(
                "json",
                data_files="data_files/c4-validation.json",
            )
            texts = [t for t in dataset_obj["train"]["text"]]
            text = "\n\n".join(texts[:2000])
            adapter.c4_text = text
        text = adapter.c4_text
    elif dataset == "ptb":
        if not hasattr(adapter, "ptb_text"):
            dataset_obj = load_dataset(
                "ptb_text_only", "penn_treebank", split="test", trust_remote_code=True
            )
            texts = [t for t in dataset_obj["sentence"]]
            text = "\n\n".join(texts)
            adapter.ptb_text = text
        text = adapter.ptb_text
    elif dataset == "alpaca":
        # text = load_alpaca_texts(tokenizer)
        return compute_perplexity_alpaca(
            adapter, model, tokenizer, batch_size=bs, ignore_index=-100
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'wikitext' or 'c4'")

    testenc = tokenizer(text, return_tensors="pt")
    testenc = testenc.input_ids

    seqlen = 2048
    nsamples = testenc.numel() // seqlen
    n_batches = (nsamples + bs - 1) // bs
    batch_starts = [0]
    # nsamples = min(testenc.numel() // seqlen, 128)

    nlls = []
    total_tokens_processed = 0

    # torch.cuda.synchronize()
    # t_start = time.perf_counter()
    # for i in range(0, nsamples, bs):
    for i in batch_starts:
        # elapsed_so_far = time.perf_counter() - t_start
        if nlls:
            pass
            # running_ppl = torch.exp(torch.stack(nlls).sum() / (total_tokens_processed)).item()
            # tps = total_tokens_processed / elapsed_so_far if elapsed_so_far > 0 else 0
            # print(
            #     f"\rsample {i}/{nsamples} | ppl: {running_ppl:.2f} | {tps:,.0f} tok/s | {elapsed_so_far:.1f}s    ",
            #     end="",
            #     flush=True,
            # )
        else:
            pass
            # print(f"\rsample {i}/{nsamples}", end="", flush=True)

        j = min(i + bs, nsamples)

        inputs = testenc[:, (i * seqlen) : (j * seqlen)].to(device)
        inputs = inputs.reshape(j - i, seqlen)

        lm_logits = model(inputs).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = inputs[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        neg_log_likelihood = loss.float() * seqlen
        # shift_logits = lm_logits[:, :-1, :].contiguous()
        # shift_labels = inputs[:, 1:]

        # loss_fct = torch.nn.CrossEntropyLoss()
        # loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        nlls.append(neg_log_likelihood)

        total_tokens_processed += (j - i) * seqlen
        del shift_labels, shift_logits

    # torch.cuda.synchronize()
    # t_end = time.perf_counter()
    # elapsed = t_end - t_start
    # tokens_per_sec = total_tokens_processed / elapsed
    # print(f"\n── Throughput ──")
    # print(f"  Total tokens : {total_tokens_processed:,}")
    # print(f"  Wall time    : {elapsed:.2f} s")
    # print(f"  Throughput   : {tokens_per_sec:,.0f} tok/s ({tokens_per_sec / 1000:.1f} ktok/s)")

    # if adapter:
    #     adapter.metrics["throughput_tok/s"] = tokens_per_sec
    #     adapter.metrics["throughput_ktok/s"] = tokens_per_sec / 1000

    # Compute perplexity
    # ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

    # ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (seqlen - 1)))

    nlls = nlls * n_batches
    ppl = float(torch.exp(torch.stack(nlls).sum() / (n_batches * seqlen)))

    # Empty CUDA cache to save memory
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    return ppl


def get_alpaca_eval_data():
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.select(range(len(dataset) - 500, len(dataset)))

    formatted_texts = []
    for example in dataset:
        if example.get("input", ""):
            text = (
                f"Below is an instruction that describes a task, paired with an input that provides further context. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                f"### Response:\n{example['output']}"
            )
        else:
            text = (
                f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Response:\n{example['output']}"
            )

        formatted_texts.append(text)

    return formatted_texts


@torch.no_grad()
def compute_perplexity_alpaca(
    adapter: ModelAdapter,
    model,
    tokenizer,
    batch_size=16,
    device="cuda",
    ignore_index=-100,
    fast=False,
):
    batches = load_alpaca_kl_div_calibration(
        tokenizer,
        # adapter.config.calib_size,
        64,
        batch_size,
        device=device,
        split="test",
        # split="train",
        ignore_index=ignore_index,
    )

    model.eval()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")

    total_loss = torch.tensor(0.0, dtype=dtype_p, device=d1)
    total_tokens = 0

    for input_ids, labels in batches:

        if fast:
            hidden = model.model(input_ids)[0]
            shift_labels = labels[:, 1:]
            mask = shift_labels != ignore_index
            h = hidden[:, :-1][mask]
            logits = model.lm_head(h)
            loss = loss_fct(logits, labels[:, 1:][mask])
            total_loss += loss
            total_tokens += int((shift_labels != ignore_index).sum())
            continue

        lm_logits = model(input_ids).logits

        # Prompt tokens are already masked to -100 in `labels`, so only
        # response tokens contribute to the loss below.
        shift_logits = lm_logits[:, :-1, :].contiguous().to(device=d1)
        shift_labels = labels[:, 1:].contiguous().to(device=d1)

        # n_tokens = int(shift_labels.sum())
        n_tokens = int((shift_labels != ignore_index).sum())
        if n_tokens == 0:
            continue

        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )

        total_loss += loss
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    ppl = math.exp(total_loss.item() / total_tokens)

    if adapter:
        adapter.metrics["alpaca_perplexity"] = ppl

    print(f"Alpaca Perplexity: {ppl:.2f}")
    return ppl


"""
eff_eval and get_test_data taken directly from D2-MoE for comparison
"""

import itertools
from torch.utils.data.dataset import Dataset

from torch.utils.flop_counter import FlopCounterMode
from collections import defaultdict
import itertools


def get_test_data(name, tokenizer, seq_len=2048, batch_size=4):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)

    ####
    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors="pt").input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len) : ((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    ####
    if "wikitext2" in name:
        test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        test_dataset = process_data(test_data, tokenizer, seq_len, "text")
    if "ptb" in name:
        test_data = load_dataset("ptb_text_only", "penn_treebank", split="test")
        test_dataset = process_data(test_data, tokenizer, seq_len, "sentence")
    elif "c4" in name:
        test_data = load_dataset("json", data_files="utils/c4-validation.json")["train"]
        test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, "text")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


@torch.no_grad()
def eff_eval(
    adapter,
    dataset="wikitext2",
    original_len=512,
    generated_len=128,
    batch_size=64,
    device="gpu",
    max_time=600,
):
    """
    eff_eval taken from https://github.com/lliai/D2MoE
    """

    model, tokenizer = adapter.model, adapter.tokenizer

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model.eval()
    test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size=batch_size)

    is_cuda = device == "gpu"
    if is_cuda:
        devices = [d for d in range(torch.cuda.device_count())]
        weight_memory = sum(p.element_size() * p.nelement() for p in model.parameters()) / (1024**3)
    else:
        devices = []
        weight_memory = sum(p.element_size() * p.nelement() for p in model.parameters()) / (1024**3)

    # -------------------------------
    # Part 1: Measure throughput
    # -------------------------------
    print("Measuring throughput...", flush=True)
    num_batches_to_fetch = 5 if device == "gpu" else 2
    throughput_time = 0
    token_num = 0
    completed_batches = 0

    # Synchronize devices and clear cache before evaluation starts
    if is_cuda:
        for d in devices:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(d)

    start_time_total = time.perf_counter()

    # stats = ActiveSubspaceStats(model)
    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        input_device = next(model.parameters()).device if is_cuda else torch.device("cpu")
        batch = batch_data.to(input_device)

        # Start timing
        start_time = time.perf_counter()

        # Generate output
        generation_output = model.generate(
            input_ids=batch,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            use_cache=True,
            top_k=50,
            max_length=original_len + generated_len,
            top_p=0.95,
            temperature=1,
        )

        # Synchronize devices to make sure all computation has finished
        if is_cuda:
            for d in devices:
                torch.cuda.synchronize(d)

        # Stop timing
        end_time = time.perf_counter()

        batch_time = end_time - start_time
        throughput_time += batch_time
        token_num += batch.shape[0] * generated_len
        completed_batches += 1

        print(
            f"Batch {batch_idx + 1}/{num_batches_to_fetch} - Time: {batch_time:.4f}s - Generated tokens: {batch.shape[0] * generated_len}",
            flush=True,
        )

        # Check whether the max time limit has been exceeded
        if end_time - start_time_total > max_time:
            print(f"Reached max time limit of {max_time}s, stopping evaluation.", flush=True)
            break

    total_time = time.perf_counter() - start_time_total

    # stats.remove()
    # summary = stats.summary()
    # print(summary)

    # save_summary(summary, "figures/summary.json")
    # plot_active_subspaces(summary, "active_subspaces.png")

    if is_cuda:
        # Synchronize devices after evaluation and get memory usage
        for d in devices:
            torch.cuda.synchronize(d)
        current_memory = sum(torch.cuda.max_memory_allocated(d) for d in devices)
        activation_memory = (current_memory) / (1024**3)
        memory_info = (
            f"Total memory usage: {current_memory / (1024 ** 3):.2f} GB\n"
            f"Weight memory: {weight_memory:.2f} GB\n"
            f"Activation memory: {activation_memory - weight_memory:.2f} GB\n"
        )
    else:
        memory_info = "Running on CPU, memory measurements unavailable.\n"

    avg_throughput = token_num / throughput_time if throughput_time > 0 else 0
    throughput_info = (
        f"Throughput: {avg_throughput:.2f} tokens/sec\n"
        f"Completed batches: {completed_batches}/{num_batches_to_fetch}\n"
        f"Total evaluation time: {total_time:.2f}s\n"
        f"Average time per batch: {(throughput_time / completed_batches):.2f}s\n"
        f"Generation length: {generated_len}\n"
    )

    # Flop measurement is incorrect:
    # -------------------------------
    # Part 2: Measure FLOPs
    # -------------------------------
    # print("Measuring FLOPs...", flush=True)
    # total_flops = 0
    # flops_completed_batches = 0
    # num_batches_to_fetch_flops = 5  # Use fewer batches for FLOPs measurement to save time

    # # Reload data
    # test_loader_flops = get_test_data(
    #     dataset, tokenizer, seq_len=original_len, batch_size=batch_size
    # )

    # # Synchronize devices
    # if is_cuda:
    #     for d in devices:
    #         torch.cuda.empty_cache()
    #         torch.cuda.synchronize(d)

    # start_time_total_flops = time.perf_counter()

    # aten = torch.ops.aten

    # def gqa_sdpa_flop(q_shape, k_shape, v_shape, *args, out_shape=None, **kwargs):
    #     b, h, s_q, d_q = q_shape  # K/V may have fewer heads than Q (GQA)
    #     return 2 * b * h * s_q * k_shape[2] * (d_q + v_shape[3])

    # SDPA_OPS = (
    #     "_scaled_dot_product_flash_attention",
    #     "_scaled_dot_product_efficient_attention",
    #     "_scaled_dot_product_cudnn_attention",
    # )
    # GQA_SDPA_FLOPS = {getattr(aten, n): gqa_sdpa_flop for n in SDPA_OPS if hasattr(aten, n)}

    # for batch_idx, batch_data in enumerate(
    #     itertools.islice(test_loader_flops, num_batches_to_fetch_flops)
    # ):
    #     input_device = next(model.parameters()).device if is_cuda else torch.device("cpu")
    #     batch = batch_data.to(input_device)

    #     # Initialize FLOPs counter
    #     # flop_counter = FlopCounterMode(model, display=False)
    #     flop_counter = FlopCounterMode(model, display=False, custom_mapping=GQA_SDPA_FLOPS)

    #     # Reset FLOPs count
    #     flop_counter.flop_counts = defaultdict(lambda: defaultdict(int))

    #     # Start timing
    #     start_time = time.perf_counter()

    #     with flop_counter:
    #         generation_output = model.generate(
    #             input_ids=batch,
    #             pad_token_id=tokenizer.eos_token_id,
    #             do_sample=True,
    #             use_cache=True,
    #             top_k=50,
    #             max_length=original_len + generated_len,
    #             top_p=0.95,
    #             temperature=1,
    #         )

    #     # Synchronize devices
    #     if is_cuda:
    #         for d in devices:
    #             torch.cuda.synchronize(d)

    #     # Stop timing
    #     end_time = time.perf_counter()

    #     batch_flops = flop_counter.get_total_flops()
    #     total_flops += batch_flops
    #     flops_completed_batches += 1

    #     batch_time = end_time - start_time
    #     print(
    #         f"Batch {batch_idx + 1}/{num_batches_to_fetch_flops} - FLOPs: {batch_flops:.2e} - Time: {batch_time:.4f}s",
    #         flush=True,
    #     )
    #     # print(f"Batch {batch_idx + 1}/{num_batches_to_fetch_flops} - FLOPs: {batch_flops:.2e} - Time: {batch_time:.4f}s")

    #     # Check whether the max time limit has been exceeded
    #     if end_time - start_time_total_flops > max_time:
    #         print(f"Reached max time limit of {max_time}s, stopping FLOPs evaluation.", flush=True)
    #         break

    # total_time_flops = time.perf_counter() - start_time_total_flops

    # avg_flops = total_flops / flops_completed_batches if flops_completed_batches > 0 else 0
    # flops_info = (
    #     f"Average FLOPs per batch: {avg_flops:.2e}\n"
    #     f"Completed batches: {flops_completed_batches}/{num_batches_to_fetch_flops}\n"
    #     f"Total evaluation time: {total_time_flops:.2f}s\n"
    #     f"Generation length: {generated_len}\n"
    # )

    # Return results
    # result = (
    #     memory_info
    #     + "\n---- Throughput Results ----\n"
    #     + throughput_info
    #     + "\n---- FLOPs Results ----\n"
    #     + flops_info
    # )

    # Return results
    result = memory_info + "\n---- Throughput Results ----\n" + throughput_info

    return result


import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


class ActiveSubspaceStats:
    def __init__(self, model, num_random_groupings=16, seed=0):
        self.num_random_groupings = num_random_groupings
        self._layers = {}
        self._handles = []
        rng = np.random.default_rng(seed)

        for name, module in model.named_modules():
            groups = getattr(module, "expert_group_indices", None)
            gate = getattr(module, "gate", None)
            if groups is None or gate is None:
                continue
            groups = np.asarray(groups, dtype=np.int64)
            k = getattr(module, "num_experts_per_tok", None) or module.top_k
            labelings = [groups] + [rng.permutation(groups) for _ in range(num_random_groupings)]
            match = re.search(r"layers\.(\d+)\.", name + ".")
            layer = int(match.group(1)) if match else len(self._layers)
            self._layers[layer] = {
                "k": k,
                "num_groups": int(groups.max()) + 1,
                "labelings": torch.as_tensor(np.stack(labelings)),  # [1 + P, E]; row 0 = actual
                "hist": None,  # [1 + P, k + 1] counts of tokens by active subspaces
            }
            self._handles.append(gate.register_forward_hook(self._make_hook(layer)))

    def _make_hook(self, layer):
        state = self._layers[layer]

        @torch.no_grad()
        def hook(gate, inputs, output):
            k = state["k"]
            if isinstance(output, tuple):
                # DeepSeek MoEGate returns (topk_idx, topk_weight, aux_loss)
                topk_idx = output[0]
            else:
                # Mixtral's gate returns router logits; softmax preserves their order
                topk_idx = output.topk(k, dim=-1).indices
            topk_idx = topk_idx.reshape(-1, k)
            device = topk_idx.device

            if state["labelings"].device != device:
                state["labelings"] = state["labelings"].to(device)
            labelings = state["labelings"]
            if state["hist"] is None:
                state["hist"] = torch.zeros(
                    labelings.shape[0], k + 1, dtype=torch.long, device=device
                )

            groups = labelings[:, topk_idx]  # [1 + P, T, k]: group of each routed expert
            hit = torch.zeros(
                *groups.shape[:2], state["num_groups"], dtype=torch.bool, device=device
            )
            hit.scatter_(2, groups, True)
            active = hit.sum(dim=2)  # [1 + P, T], values in 1..min(k, G)
            state["hist"].scatter_add_(1, active, torch.ones_like(active))

        return hook

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def summary(self):
        layers = sorted(l for l, s in self._layers.items() if s["hist"] is not None)
        if not layers:
            raise RuntimeError("No routing was recorded; run the model while hooks are attached.")
        max_active = max(min(self._layers[l]["k"], self._layers[l]["num_groups"]) for l in layers)

        actual, random_grouping, tokens = [], [], []
        for l in layers:
            hist = self._layers[l]["hist"].double().cpu()[:, 1 : max_active + 1]
            frac = hist / hist.sum(dim=1, keepdim=True)
            actual.append(frac[0].tolist())
            if self.num_random_groupings > 0:
                random_grouping.append(frac[1:].mean(dim=0).tolist())
            tokens.append(int(hist[0].sum()))

        return {
            "layers": layers,
            "active_subspaces": list(range(1, max_active + 1)),
            "actual": actual,  # [L][M] fractions
            "random_grouping": random_grouping or None,  # [L][M] fractions
            "num_random_groupings": self.num_random_groupings,
            "tokens_per_layer": tokens,
        }


def save_summary(summary, path):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def load_summary(path):
    with open(path) as f:
        return json.load(f)
