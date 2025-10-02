"""Data loading utilities for the DDLC project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def default_image_transform(height: int = 224, width: int = 224) -> transforms.Compose:
    """Return the default preprocessing pipeline for chart images."""

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


@dataclass
class TableInfo:
    """Metadata about a table file."""

    path: Path
    num_rows: int
    num_cols: int


def _load_numeric_csv(path: Path) -> torch.Tensor:
    """Load a CSV file and return a float tensor of numeric columns only."""

    df = pd.read_csv(path)
    numeric_df = df.select_dtypes(include=[np.number]).fillna(0.0)
    if numeric_df.empty:
        # If a table does not have numeric data we create a single zero column.
        numeric_df = pd.DataFrame([[0.0]])
    return torch.tensor(numeric_df.to_numpy(), dtype=torch.float32)


def discover_tables(table_dir: Path) -> List[TableInfo]:
    """Scan ``table_dir`` and collect information about each table file."""

    table_infos: List[TableInfo] = []
    for csv_path in sorted(table_dir.glob("*.csv")):
        data = _load_numeric_csv(csv_path)
        table_infos.append(TableInfo(csv_path, data.shape[0], data.shape[1]))
    if not table_infos:
        raise ValueError(f"No CSV files found in {table_dir}.")
    return table_infos


class DDLCDataset(Dataset):
    """Dataset consisting of chart/table pairs used for contrastive training."""

    def __init__(
        self,
        chart_dir: Path | str,
        table_dir: Path | str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        preload_tables: bool = False,
    ) -> None:
        super().__init__()
        self.chart_dir = Path(chart_dir)
        self.table_dir = Path(table_dir)
        self.transform = transform or default_image_transform()

        self.samples = self._collect_samples()
        self.max_columns = max(info.num_cols for info in self.samples.values())
        self.max_rows = max(info.num_rows for info in self.samples.values())

        self._table_cache: Dict[str, torch.Tensor] = {}
        if preload_tables:
            for sample_id in self.ids:
                self._table_cache[sample_id] = self._load_table(sample_id)

    def _collect_samples(self) -> Dict[str, TableInfo]:
        charts = {path.stem: path for path in sorted(self.chart_dir.glob("*.png"))}
        tables = discover_tables(self.table_dir)
        samples: Dict[str, TableInfo] = {}
        for info in tables:
            sample_id = info.path.stem
            if sample_id in charts:
                samples[sample_id] = info
        if not samples:
            raise ValueError("No matching chart/table pairs were found.")
        return samples

    @property
    def ids(self) -> List[str]:
        return sorted(self.samples.keys())

    def _load_chart(self, sample_id: str) -> torch.Tensor:
        image_path = self.chart_dir / f"{sample_id}.png"
        with Image.open(image_path) as img:
            return self.transform(img.convert("RGB"))

    def _load_table(self, sample_id: str) -> torch.Tensor:
        if sample_id in self._table_cache:
            return self._table_cache[sample_id].clone()
        tensor = _load_numeric_csv(self.table_dir / f"{sample_id}.csv")
        return tensor

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        sample_id = self.ids[index]
        chart = self._load_chart(sample_id)
        table = self._load_table(sample_id)
        return {"id": sample_id, "chart": chart, "table": table}


def build_collate_fn(max_columns: int) -> Callable[[Sequence[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    """Create a collate function that pads tables to ``max_columns``."""

    def collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        charts = torch.stack([item["chart"] for item in batch], dim=0)
        ids = [item["id"] for item in batch]

        row_lengths = [item["table"].shape[0] for item in batch]
        max_rows = max(row_lengths)

        tables = torch.zeros(len(batch), max_rows, max_columns, dtype=torch.float32)
        row_mask = torch.zeros(len(batch), max_rows, dtype=torch.float32)

        for idx, item in enumerate(batch):
            table = item["table"]
            rows, cols = table.shape
            tables[idx, :rows, :cols] = table
            row_mask[idx, :rows] = 1.0

        return {"ids": ids, "charts": charts, "tables": tables, "row_mask": row_mask}

    return collate


def load_repository_tables(table_dir: Path | str) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """Load all tables from ``table_dir`` and return padded tensors and masks."""

    table_dir = Path(table_dir)
    infos = discover_tables(table_dir)
    max_columns = max(info.num_cols for info in infos)
    max_rows = max(info.num_rows for info in infos)

    table_ids: List[str] = []
    tables = torch.zeros(len(infos), max_rows, max_columns, dtype=torch.float32)
    row_mask = torch.zeros(len(infos), max_rows, dtype=torch.float32)

    for idx, info in enumerate(infos):
        data = _load_numeric_csv(info.path)
        rows, cols = data.shape
        tables[idx, :rows, :cols] = data
        row_mask[idx, :rows] = 1.0
        table_ids.append(info.path.stem)

    return table_ids, tables, row_mask
