"""Neural network components for Dataset Discovery via Line Charts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChartEncoder(nn.Module):
    """Encode chart images into dense embeddings."""

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, charts: torch.Tensor) -> torch.Tensor:
        features = self.backbone(charts)
        embeddings = self.head(features)
        return F.normalize(embeddings, dim=-1)


class TableEncoder(nn.Module):
    """Encode numeric tables by aggregating row representations."""

    def __init__(self, num_columns: int, embedding_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.row_mlp = nn.Sequential(
            nn.Linear(num_columns, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, tables: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_rows, _ = tables.shape
        row_features = self.row_mlp(tables)
        row_mask = row_mask.unsqueeze(-1)
        row_features = row_features * row_mask
        sums = row_features.sum(dim=1)
        counts = row_mask.sum(dim=1).clamp(min=1.0)
        pooled = sums / counts
        pooled = self.layer_norm(pooled)
        return F.normalize(pooled, dim=-1)


@dataclass
class DDLCModelConfig:
    """Configuration values for :class:`DDLCModel`."""

    num_columns: int
    embedding_dim: int = 256
    hidden_dim: int = 256


class DDLCModel(nn.Module):
    """Dual-encoder architecture for chart-table matching."""

    def __init__(self, config: DDLCModelConfig) -> None:
        super().__init__()
        self.chart_encoder = ChartEncoder(embedding_dim=config.embedding_dim)
        self.table_encoder = TableEncoder(
            num_columns=config.num_columns,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
        )
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def encode_charts(self, charts: torch.Tensor) -> torch.Tensor:
        return self.chart_encoder(charts)

    def encode_tables(self, tables: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
        return self.table_encoder(tables, row_mask)

    def forward(
        self,
        charts: torch.Tensor,
        tables: torch.Tensor,
        row_mask: torch.Tensor,
    ) -> torch.Tensor:
        chart_embeddings = self.encode_charts(charts)
        table_embeddings = self.encode_tables(tables, row_mask)
        scale = self.logit_scale.exp().clamp(min=1e-3, max=100.0)
        return chart_embeddings @ table_embeddings.t() * scale
