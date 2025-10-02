"""Evaluation script for chart-to-table retrieval."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from tqdm.auto import tqdm

from data import default_image_transform, load_repository_tables
from helper import ndcg_at_k, precision_at_k, seed_everything
from model import DDLCModel, DDLCModelConfig


def parse_ground_truth(path: Path) -> Dict[str, List[str]]:
    data = json.loads(path.read_text())
    ground_truth: Dict[str, List[str]] = defaultdict(list)
    for query_id, table_ids in data.items():
        if isinstance(table_ids, list):
            ground_truth[str(query_id)] = [str(tid) for tid in table_ids]
        else:
            ground_truth[str(query_id)] = [str(table_ids)]
    return ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DDLC model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained checkpoint")
    parser.add_argument("--query_dir", type=Path, required=True, help="Directory with query charts")
    parser.add_argument("--repository_dir", type=Path, required=True, help="Directory with candidate tables")
    parser.add_argument("--ground_truth", type=Path, required=True, help="JSON mapping query ids to relevant table ids")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--topk", type=int, default=10, help="Evaluate metrics at top-k")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_model(checkpoint_path: Path, num_columns: int, device: torch.device) -> DDLCModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_data = checkpoint.get("config", {})
    config = DDLCModelConfig(
        num_columns=num_columns,
        embedding_dim=config_data.get("embedding_dim", 256),
        hidden_dim=config_data.get("hidden_dim", 256),
    )
    model = DDLCModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def embed_repository(
    model: DDLCModel,
    tables: torch.Tensor,
    row_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        tables = tables.to(device)
        row_mask = row_mask.to(device)
        embeddings = model.encode_tables(tables, row_mask)
    return embeddings


def evaluate() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device)

    table_ids, tables, row_mask = load_repository_tables(args.repository_dir)
    model = load_model(args.checkpoint, num_columns=tables.size(-1), device=device)

    repository_embeddings = embed_repository(model, tables, row_mask, device)

    transform = default_image_transform()
    ground_truth = parse_ground_truth(args.ground_truth)

    precision_scores: List[float] = []
    ndcg_scores: List[float] = []

    with torch.no_grad():
        for query_path in tqdm(sorted(args.query_dir.glob("*.png")), desc="Evaluation"):
            query_id = query_path.stem
            with Image.open(query_path) as image:
                chart = transform(image.convert("RGB")).unsqueeze(0).to(device)
            query_embedding = model.encode_charts(chart)
            scores = torch.mv(repository_embeddings, query_embedding.squeeze(0)).cpu().tolist()
            ranking = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
            relevant_tables = ground_truth.get(query_id, [])
            ranked_ids = [table_ids[idx] for idx in ranking]

            precision_scores.append(precision_at_k(ranked_ids, relevant_tables, args.topk))
            ndcg_scores.append(ndcg_at_k(ranked_ids, relevant_tables, args.topk))

    print(f"Precision@{args.topk}: {sum(precision_scores) / len(precision_scores):.4f}")
    print(f"NDCG@{args.topk}: {sum(ndcg_scores) / len(ndcg_scores):.4f}")


if __name__ == "__main__":
    evaluate()
