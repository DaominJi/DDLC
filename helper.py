"""Utility functions for training and evaluation of DDLC models."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the cosine similarity matrix between two batches of vectors."""

    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.t()


def contrastive_loss(
    chart_embeddings: torch.Tensor,
    table_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute the symmetric InfoNCE loss used during training."""

    logits = cosine_similarity(chart_embeddings, table_embeddings) / temperature
    targets = torch.arange(len(chart_embeddings), device=logits.device)
    loss_chart = F.cross_entropy(logits, targets)
    loss_table = F.cross_entropy(logits.t(), targets)
    return (loss_chart + loss_table) * 0.5


def dcg_at_k(relevance: Sequence[float], k: int) -> float:
    """Discounted cumulative gain for the first ``k`` positions."""

    if k <= 0:
        return 0.0
    relevance = relevance[:k]
    return sum((2 ** rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevance))


def ndcg_at_k(ranked_items: Sequence[int], ground_truth: Sequence[int], k: int) -> float:
    """Compute NDCG@k for ranked items against a set of relevant ids."""

    if k <= 0:
        return 0.0

    gains = [1.0 if item in ground_truth else 0.0 for item in ranked_items[:k]]
    actual_dcg = dcg_at_k(gains, k)
    ideal_gains = sorted(gains, reverse=True)
    ideal_dcg = dcg_at_k(ideal_gains, k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def precision_at_k(ranked_items: Sequence[int], ground_truth: Sequence[int], k: int) -> float:
    """Compute Precision@k for ranked items against a set of relevant ids."""

    if k <= 0:
        return 0.0
    topk = ranked_items[:k]
    hits = sum(1 for item in topk if item in ground_truth)
    return hits / k


@dataclass
class AverageMeter:
    """Track the average value of streaming metrics."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def topk_indices(scores: Iterable[float], k: int) -> List[int]:
    """Return the indices of the ``k`` largest scores."""

    scores = list(scores)
    k = min(k, len(scores))
    return sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
