"""
Generate GraphCLIP node-sequence embeddings (pre-mean-pooling) for random 10k arXiv nodes.

Output: graphclip_embeddings.seq.pt
Each entry maps node_id -> {"embedding": (S_max, D) padded tensor, "mask": (S_max,) 1/0}.
"""
# generate_graphclip_seq_embeddings.py
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_sparse import SparseTensor
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch.serialization import add_safe_globals

GRAPHCLIP_DIR = Path(__file__).resolve().parent / "GraphCLIP"
if str(GRAPHCLIP_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPHCLIP_DIR))

from GraphCLIP.data.load import load_data  # noqa: E402
from GraphCLIP.models import GraphCLIP  # noqa: E402

add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage, SparseTensor])
DATASET_NAME = "ogbn-arxiv"
CSV_PATH = "/home/xyx/GraphTranslator/data/arxiv/summary_embeddings_random10000.csv"
NUM_SAMPLES = 10000
NUM_HOPS = 1
OUTPUT_PATH = Path("graphclip_embeddings.seq.pt")


def load_arxiv_data(seed: int = 0):
    pt_path = GRAPHCLIP_DIR / "processed_data" / "ogbn-arxiv.pt"
    if pt_path.exists():
        data = torch.load(pt_path, map_location="cpu")
    else:
        data, _, _, _ = load_data(DATASET_NAME, seed=seed)
    edge_index = getattr(data, "edge_index", None)
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        data.edge_index = torch.stack([row, col], dim=0)
    elif edge_index is None and hasattr(data, "adj_t"):
        row, col, _ = data.adj_t.coo()
        data.edge_index = torch.stack([row, col], dim=0)
    return data


def build_khop_subgraph(data, node_id: int, num_hops: int):
    subset, edge_index, mapping, _ = k_hop_subgraph(
        node_id, num_hops, data.edge_index, relabel_nodes=True
    )
    x = data.x[subset]
    graph = Data(
        edge_index=edge_index,
        x=x,
        batch=torch.zeros(x.size(0), dtype=torch.long),
        root_n_index=torch.tensor(mapping, dtype=torch.long),
    )
    transform = T.AddRandomWalkPE(walk_length=32, attr_name="pe")
    return transform(graph)


def pad_sequences(
    records: List[Tuple[int, torch.Tensor]],
) -> Dict[int, Dict[str, torch.Tensor]]:
    max_len = max(t.size(0) for _, t in records)
    dim = records[0][1].size(-1)
    padded: Dict[int, Dict[str, torch.Tensor]] = {}
    for node_id, emb in records:
        length = emb.size(0)
        pad_len = max_len - length
        if pad_len > 0:
            emb = torch.cat([emb, emb.new_zeros((pad_len, dim))], dim=0)
        mask = torch.cat(
            [
                torch.ones(length, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ],
            dim=0,
        )
        padded[node_id] = {"embedding": emb.cpu(), "mask": mask.cpu()}
    return padded


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_arxiv_data(seed=0)
    attn_kwargs = {"dropout": 0.0}
    model = GraphCLIP(384, 1024, 12, attn_kwargs, text_model="tiny")
    ckpt_path = GRAPHCLIP_DIR / "checkpoints" / "pretrained_graphclip.pt"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    df = pd.read_csv(CSV_PATH, nrows=NUM_SAMPLES)
    records: List[Tuple[int, torch.Tensor]] = []
    with torch.no_grad():
        for idx, row in enumerate(df.itertuples(), start=1):
            node_id = int(row.node_id)
            graph = build_khop_subgraph(data, node_id, NUM_HOPS).to(device)
            # 直接从 GraphCLIP 的图编码器拿节点表示（未过池化/MLP）
            _, _, node_embs, node_batch = model.graph_model(
                graph.x, graph.pe, graph.edge_index, graph.batch, graph.root_n_index, return_node_embeddings=True
            )
            # node_batch should be all zeros (single graph), but keep only current graph nodes
            node_embs = node_embs[node_batch == 0].detach().cpu()
            records.append((node_id, node_embs))
            if idx % 50 == 0 or idx == len(df):
                print(f"Processed {idx}/{len(df)} nodes...")

    padded = pad_sequences(records)
    torch.save(padded, OUTPUT_PATH)
    print(f"Saved sequence embeddings for {len(padded)} nodes to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
