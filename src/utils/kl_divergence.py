import os
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from src.utils.model import d1, d2, calib_device

from src.adapters.moe_adapter import MoEAdapter

from src.eval import load_alpaca_kl_div_calibration, load_calibration_texts

from torch import Tensor


def softmax_fp32(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.softmax(x, dim=dim, dtype=torch.float32)


def log_softmax_fp32(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.log_softmax(x, dim=dim, dtype=torch.float32)


class ForwardKLLoss(torch.nn.Module):

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self, student_logits: Tensor, teacher_top_vals: Tensor, teacher_top_idx: Tensor, labels
    ) -> torch.Tensor:
        # if we are taking topk sampling then we've already softmaxxed teacher logs
        # teacher_prob = softmax_fp32(teacher_logits, dim=-1).detach()

        # teacher_logits: [T, topk_sampled]

        teacher_prob = teacher_top_vals.to(dtype=torch.float32)
        student_logprob_all = log_softmax_fp32(student_logits, dim=-1)
        if teacher_top_idx is not None:
            student_logprob = torch.gather(
                student_logprob_all, dim=-1, index=teacher_top_idx.to(dtype=torch.long)
            )
        else:
            student_logprob = student_logprob_all

        prod_probs = teacher_prob * student_logprob
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (labels != self.ignore_index).int()
        return -torch.sum(x * mask.view(-1), dim=0), torch.sum(mask.view(-1), dim=0)


def teacher_logits_path(batch_number: int) -> str:
    return os.path.join(os.environ["TMPDIR"], "model_logits", f"batch_{batch_number}_logits.pt")


def load_teacher_logits(batch_number: int, device) -> torch.Tensor:
    path = teacher_logits_path(batch_number)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no teacher logits for batch {batch_number} at {path} -- dumped "
            f"batches and calibration batches must line up 1:1, in order"
        )
    return torch.load(path, map_location=device)


@torch.no_grad()
def compute_forward_kl(
    adapter: MoEAdapter,
    ignore_index: int = -100,
    rows_per_chunk: Optional[int] = None,
    subtract_teacher_entropy: bool = False,
    gen=0,
    use_labels: bool = False,
    dataset="alpaca",
) -> float:

    model = adapter.model

    device = next(model.parameters()).device
    kl_loss_fn = ForwardKLLoss(ignore_index=ignore_index)
    # calib_size = adapter.config.calib_size
    teacher_calibs_size = 64
    calibs_size = 64
    # print(
    #     f"in compute_Forward_kl: calib_size = {calibs_size}, batch_size = {adapter.config.calibs_batch_size}"
    # )
    # calibs = load_alpaca_kl_div_calibration(
    #     adapter.tokenizer, calibs_size, adapter.config.calibs_batch_size, split="test"
    # )
    logit_path = os.path.expandvars(f"$TMPDIR/model_logits/logits-{dataset}.pt")
    if not hasattr(adapter, "teacher_outputs"):
        adapter.teacher_outputs = torch.load(logit_path, map_location=device)
    teacher_outputs: list = adapter.teacher_outputs  # [n_batches, [vals, idx, input_ids, labels]]
    # teacher_vals/idx: [B*T, topk]

    take_k_batches = calibs_size // adapter.config.calibs_batch_size
    t_s_ratio = teacher_calibs_size // calibs_size
    import numpy as np
    import random

    # this hopefully reduces noise over many generations
    batch_start = gen % t_s_ratio
    start = batch_start * take_k_batches
    calibs = teacher_outputs[start : start + take_k_batches]

    # random.seed(seed)
    # calibs = random.sample(teacher_outputs, take_k_batches)

    total_sum = torch.zeros((), device=d1)
    total_tokens = 0
    model.eval()
    model.zero_grad()
    for count, batch in enumerate(calibs):
        teacher_top_vals, teacher_top_idx, input_ids, labels = batch
        # teacher_top_vals, input_ids, labels = batch

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        outputs = model(
            input_ids,
            output_hidden_states=False,
            # output_router_logits=False,
        )
        # outputs = model(input_ids)
        student_logits: Tensor = outputs.logits

        student_logits = student_logits.view(-1, student_logits.size(-1))

        # if teacher_top_vals.shape[0] != student_logits.shape[0]:
        #     raise AttributeError(
        #         f"wrong shape/bsz: {teacher_top_vals.shape} != {student_logits.shape}"
        #     )

        labels_flat = labels.view(-1) if use_labels else input_ids.view(-1)
        # labels_flat = batch.view(-1)

        val_sum, n_masked = kl_loss_fn(
            student_logits, teacher_top_vals, teacher_top_idx, labels_flat
        )
        # val_sum, n_masked = kl_loss_fn(student_logits, teacher_top_vals, None, labels_flat)
        total_sum += val_sum
        total_tokens += n_masked
        if subtract_teacher_entropy:
            mask = labels_flat != ignore_index
            p = teacher_top_vals.to(dtype=torch.float32)
            ent = torch.special.entr(p).sum(dim=-1)
            val_sum -= ent[mask].sum() / int(mask.sum())

    overall = (total_sum / total_tokens).item()
    return overall


import time


@torch.no_grad()
def create_teacher_logits(adapter: MoEAdapter, topk_sampled=32, dataset="alpaca"):

    logit_dir = os.path.expandvars(f"$TMPDIR/model_logits/")
    logit_path = os.path.join(logit_dir, f"logits-{dataset}.pt")
    if topk_sampled is None:
        topk_sampled = adapter.model.config.vocab_size

    if os.path.exists(logit_path):
        return

    model = adapter.model
    tokenizer, model = adapter.tokenizer, adapter.model
    config = adapter.config
    # calibs_size = adapter.config.calib_size
    calibs_size = 64

    calibs = load_calibration_texts(
        calibs_size,
        model,
        tokenizer,
        adapter.config.calibs_batch_size,
        dataset=dataset,
        split="train",
    )

    # calibs = load_alpaca_kl_div_calibration(
    #     tokenizer, calibs_size, config.calibs_batch_size, split="test"
    # )

    calibs = [
        calibs[i]
        for i in torch.randperm(len(calibs), generator=torch.Generator().manual_seed(1234))
    ]
    model.eval()

    all_probs = []

    for count, batch in enumerate(calibs):

        if dataset == "alpaca":
            input_ids, labels = batch
        else:
            input_ids = batch
            labels = input_ids

        # do it like this because mixtral and deepseek have different options
        kwargs = {}
        if adapter.arch == "mixtral":
            kwargs["output_router_logits"] = False
        kwargs["output_hidden_states"] = False
        outputs = model(input_ids, **kwargs)

        logits = outputs.logits  # [T, 32k] for mixtral8x7b (maybe [B, T, 32k])

        logits = logits.view(-1, logits.size(-1))
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        vals, indx = probs.topk(topk_sampled, dim=-1)  # []
        # all_probs.append([probs, input_ids, labels])
        all_probs.append([vals, indx, input_ids, labels])

        del logits, outputs

    os.makedirs(logit_dir, exist_ok=True)
    torch.save(all_probs, logit_path)
    # adapter.teacher_outputs = all_probs


if __name__ == "__main__":
    # from wherever import build_student
    # model = build_student().to("cuda")
    # print(compute_forward_kl(model))
    raise NotImplementedError("wire up your student module and call compute_forward_kl(model)")
