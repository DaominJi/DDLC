"""Training script for Dataset Discovery via Line Charts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import DDLCDataset, build_collate_fn
from helper import AverageMeter, contrastive_loss, seed_everything
from model import DDLCModel, DDLCModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DDLC contrastive model")
    parser.add_argument("--chart_dir", type=Path, default=Path("Sample_Data"), help="Directory with training charts")
    parser.add_argument("--table_dir", type=Path, default=Path("Sample_Data"), help="Directory with training tables")
    parser.add_argument("--batch_size", type=int, default=4, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ddlc_model.pt"), help="Path to save the checkpoint")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Size of the shared embedding space")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden size for table encoder")
    return parser.parse_args()


def save_checkpoint(path: Path, state: Dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    dataset = DDLCDataset(chart_dir=args.chart_dir, table_dir=args.table_dir)
    collate_fn = build_collate_fn(dataset.max_columns)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    device = torch.device(args.device)

    model = DDLCModel(
        DDLCModelConfig(
            num_columns=dataset.max_columns,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
        )
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_meter = AverageMeter()

        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for batch in progress:
            charts = batch["charts"].to(device)
            tables = batch["tables"].to(device)
            row_mask = batch["row_mask"].to(device)

            chart_embeddings = model.encode_charts(charts)
            table_embeddings = model.encode_tables(tables, row_mask)
            loss = contrastive_loss(chart_embeddings, table_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), charts.size(0))
            progress.set_postfix({"loss": loss_meter.average})

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                args.checkpoint,
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": {
                        "num_columns": dataset.max_columns,
                        "embedding_dim": args.embedding_dim,
                        "hidden_dim": args.hidden_dim,
                    },
                },
            )


if __name__ == "__main__":
    main()
