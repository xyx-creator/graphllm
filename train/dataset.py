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
    # 保持原逻辑为空，或者填入你需要的 context 构建逻辑
    return ""


class GraphTeacherDataset(Dataset):
    """Pairs GraphCLIP sequence embeddings (pre-computed) with teacher-generated targets."""

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
        # 预期数据结构: {node_id: {'embedding': Tensor, 'mask': Tensor}}
        self.embeddings_data: Dict[int, Dict[str, torch.Tensor]] = torch.load(
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
            # 1. 基础索引校验
            if pd.isna(row.get("sample_idx")):
                skipped += 1
                continue
            
            sample_idx = int(row["sample_idx"])
            if sample_idx not in self.metadata.index:
                skipped += 1
                continue
            
            meta = self.metadata.loc[sample_idx]
            node_id = int(meta["node_id"])
            
            # 2. Graph 数据校验 (同时获取 embedding 和 mask)
            data_item = self.embeddings_data.get(node_id)
            if data_item is None:
                skipped += 1
                continue
            
            # 显式检查 mask 是否存在，防止 Collate 崩溃
            if "embedding" not in data_item or "mask" not in data_item:
                skipped += 1
                continue

            # 3. 文本有效性校验 (防止 "nan" 被转为字符串 "nan")
            raw_teacher_text = row.get("teacher_text")
            if pd.isna(raw_teacher_text): 
                skipped += 1
                continue
            
            teacher_text = str(raw_teacher_text).strip()
            if not teacher_text:
                skipped += 1
                continue

            context = _format_context(meta)
            instruction = self.instruction_template.format(
                context=context,
                title=str(meta.get("title", "")).strip(),
                paper_summary=str(meta.get("paper_summary", "")).strip(),
                neighbor_summary=str(meta.get("citepapers_summary", "")).strip(),
            ).strip()

            self.samples.append({
                "node_id": node_id,
                "sample_idx": sample_idx,
                "graph_emb": data_item["embedding"].float(), # Shape: (S, D)
                "graph_mask": data_item["mask"].long(),      # Shape: (S,)
                "instruction": instruction,
                "target": teacher_text,
            })

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
    # 1. Stack Embeddings: (B, S, D)
    embeddings = torch.stack([item["graph_emb"] for item in batch])
    
    # 2. Stack Masks: (B, S) - 此时已确保 mask 存在且为 Tensor
    masks = torch.stack([item["graph_mask"] for item in batch])
    
    # 3. Pack into dictionary for Model
    graph_batch = {
        "embedding": embeddings,
        "mask": masks
    }

    return {
        "graph_batch": graph_batch,
        "instructions": [item["instruction"] for item in batch],
        "targets": [item["target"] for item in batch],
        "node_ids": [item["node_id"] for item in batch],
        "sample_idx": [item["sample_idx"] for item in batch],
    }