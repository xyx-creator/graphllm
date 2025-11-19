# train/dataset.py
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)


def _ensure_sample_index(df: pd.DataFrame) -> pd.DataFrame:
    if "sample_idx" not in df.columns:
        df = df.reset_index().rename(columns={"index": "sample_idx"})
    return df


def _format_context(row: pd.Series) -> str:
    return ""


class GraphTeacherDataset(Dataset):
    """Pairs GraphCLIP embeddings with teacher-generated targets."""

    def __init__(
        self,
        embeddings_path: str,
        teacher_csv: str,
        metadata_csv: str,
        instruction_template: str,
        *,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embeddings_path = Path(embeddings_path)
        self.teacher_csv = Path(teacher_csv)
        self.metadata_csv = Path(metadata_csv)
        self.instruction_template = instruction_template

        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_path}")
        if not self.teacher_csv.exists():
            raise FileNotFoundError(f"Teacher CSV not found: {self.teacher_csv}")
        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_csv}")

        LOGGER.info("Loading embeddings from %s", self.embeddings_path)
        self.embeddings: Dict[int, torch.Tensor] = torch.load(
            self.embeddings_path, map_location="cpu"
        )

        LOGGER.info("Reading metadata from %s", self.metadata_csv)
        metadata_df = _ensure_sample_index(pd.read_csv(self.metadata_csv))
        metadata_df["node_id"] = metadata_df["node_id"].astype(int)
        self.metadata = metadata_df.set_index("sample_idx")

        LOGGER.info("Reading teacher data from %s", self.teacher_csv)
        teacher_df = pd.read_csv(self.teacher_csv).sort_values("sample_idx")
        if max_samples is not None:
            teacher_df = teacher_df.head(max_samples)

        self.samples: List[Dict[str, object]] = []
        skipped = 0
        for _, row in teacher_df.iterrows():
            if pd.isna(row.get("sample_idx")):
                skipped += 1
                continue
            sample_idx = int(row["sample_idx"])
            if sample_idx not in self.metadata.index:
                skipped += 1
                continue
            meta = self.metadata.loc[sample_idx]
            node_id = int(meta["node_id"])
            embedding = self.embeddings.get(node_id)
            if embedding is None:
                skipped += 1
                continue
            title = str(meta.get("title", "")).strip()
            paper_summary = str(meta.get("paper_summary", "")).strip()
            neighbor_summary = str(meta.get("citepapers_summary", "")).strip()
            context = _format_context(meta)
            instruction = self.instruction_template.format(
                context=context,
                title="",
                paper_summary="",
                neighbor_summary="",
            ).strip()
            if pd.isna(row.get("teacher_text")):
                skipped += 1
                continue
            teacher_text = str(row["teacher_text"]).strip()
            if not teacher_text:
                skipped += 1
                continue
            self.samples.append(
                {
                    "node_id": node_id,
                    "sample_idx": sample_idx,
                    "embedding": embedding.float(),
                    "instruction": instruction,
                    "target": teacher_text,
                    "title": title,
                    "paper_summary": paper_summary,
                    "neighbor_summary": neighbor_summary,
                }
            )

        if not self.samples:
            raise RuntimeError("No usable samples found for training.")
        LOGGER.info(
            "Loaded %d samples for training (skipped %d problematic rows).",
            len(self.samples),
            skipped,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.samples[idx]


def collate_teacher_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    embeddings = torch.stack([item["embedding"] for item in batch])
    return {
        "graph_emb": embeddings,
        "instructions": [item["instruction"] for item in batch],
        "targets": [item["target"] for item in batch],
        "node_ids": [item["node_id"] for item in batch],
        "sample_idx": [item["sample_idx"] for item in batch],
    }
