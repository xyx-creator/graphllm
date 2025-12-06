# test_embedding_inference.py
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import yaml

from graphllm.model import GraphLLM

LOGGER = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_dtype(name: str) -> torch.dtype:
    """Robust dtype parsing compatible with training script."""
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype value: {name}")
    return mapping[key]


def build_model(cfg: Dict[str, Any]) -> GraphLLM:
    model_cfg = cfg["model"]
    llm_kwargs = dict(model_cfg.get("llm_kwargs", {}))
    tokenizer_kwargs = dict(model_cfg.get("tokenizer_kwargs", {}))
    projector_kwargs = dict(model_cfg.get("projector", {}))

    # Robust dtype parsing
    dtype_name = model_cfg.get("llm_dtype")
    if dtype_name:
        llm_kwargs.setdefault("torch_dtype", parse_dtype(dtype_name))

    llm_kwargs.setdefault("trust_remote_code", True)
    tokenizer_kwargs.setdefault("use_fast", False)
    tokenizer_kwargs.setdefault("trust_remote_code", True)

    model = GraphLLM(
        graph_encoder=None,  # Inference uses pre-computed embeddings
        llm_path=model_cfg["llm_path"],
        llm_kwargs=llm_kwargs,
        tokenizer_path=model_cfg.get("tokenizer_path"),
        tokenizer_kwargs=tokenizer_kwargs,
        projector_kwargs=projector_kwargs,
        instruction_max_len=model_cfg.get("text_max_len", 512),
        graph_feature_type="sequence",  # [Critical] Ensure consistency with training
        device=model_cfg.get("device", "cpu"),
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on GraphCLIP embeddings.")
    parser.add_argument("--config", type=str, default="train/config.yaml")
    parser.add_argument("--projector", type=str, default="train/output/projector_stage1.pt")
    parser.add_argument("--embeddings", type=str, default="graphclip_embeddings.seq.pt")
    parser.add_argument(
        "--metadata",
        type=str,
        default="/home/xyx/GraphTranslator/data/arxiv/summary_embeddings_random100.csv",
    )
    parser.add_argument("--output", type=str, default="train/output/test_inference_outputs.jsonl")
    parser.add_argument("--batch-size", type=int, default=4)
    
    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=256)
    # [新增] 最小输出字数，强制模型多生成内容
    parser.add_argument("--min-new-tokens", type=int, default=32, help="Force model to generate at least this many tokens.")
    
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    
    return parser.parse_args()


def load_embeddings(path: Path) -> Dict[int, Dict[str, torch.Tensor]]:
    """Loads the {node_id: {'embedding':..., 'mask':...}} structure."""
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
        
    LOGGER.info(f"Loading embeddings from {path}...")
    data = torch.load(path, map_location="cpu")
    out = {}
    for k, v in data.items():
        out[int(k)] = {
            "embedding": v["embedding"].float(),
            "mask": v["mask"].long(),
        }
    return out


def main():
    args = parse_args()
    cfg = load_config(args.config)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 1. Load Metadata (Robustly)
    LOGGER.info(f"Loading metadata from {args.metadata}")
    metadata = pd.read_csv(
        args.metadata,
        usecols=["node_id", "title", "paper_summary"],
        engine="python",
        on_bad_lines="skip",
    )
    metadata["node_id"] = pd.to_numeric(metadata["node_id"], errors="coerce")
    metadata = metadata.dropna(subset=["node_id"]).reset_index(drop=True)
    metadata["sample_idx"] = metadata.index

    # 2. Load Embeddings
    embeddings = load_embeddings(Path(args.embeddings))

    # 3. Build Model & Load Projector
    LOGGER.info("Building model...")
    model = build_model(cfg)
    
    projector_path = Path(args.projector)
    if not projector_path.exists():
        raise FileNotFoundError(f"Projector checkpoint not found: {projector_path}")
        
    LOGGER.info(f"Loading projector weights from {projector_path}")
    state_dict = torch.load(projector_path, map_location="cpu")
    model.projector.load_state_dict(state_dict)
    
    model.projector.to(model.device).eval()
    model.llm.eval()

    # 4. Prompt Engineering
    # Removed "Answer: This paper" to allow natural generation.
    prompt_template = cfg["model"].get(
        "generate_prompt",
        "Question: We are trying to explore the paper titled <{title}>. "
        "Please summarize the topic and content of the paper and its citations.\n\nAnswer:",
    )
    if prompt_template.strip().endswith("This paper"):
        LOGGER.warning("Prompt template ends with 'This paper', which might cause sentence fragments. Consider removing it.")

    records = []
    missing_emb = 0
    for row in metadata.itertuples(index=False):
        node_id = int(row.node_id)
        emb_item = embeddings.get(node_id)
        if emb_item is None:
            missing_emb += 1
            continue

        try:
            prompt = prompt_template.format(title=str(row.title))
        except KeyError as e:
            LOGGER.error(f"Prompt template expects field {e} missing in metadata.")
            raise

        records.append(
            {
                "node_id": node_id,
                "sample_idx": int(row.sample_idx),
                "title": str(row.title),
                "paper_summary": str(row.paper_summary),
                "embedding": emb_item["embedding"],
                "mask": emb_item["mask"],
                "prompt": prompt,
            }
        )

    LOGGER.info(f"Prepared {len(records)} records (skipped {missing_emb} missing).")
    if not records:
        raise RuntimeError("No valid records found.")

    # 5. Inference Config
    batch_size = args.batch_size
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens, # [关键修改]
        "repetition_penalty": args.repetition_penalty,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": model.pad_token_id, 
    }
    if model.tokenizer.eos_token_id is not None:
        gen_kwargs["eos_token_id"] = model.tokenizer.eos_token_id

    # 6. Inference Loop with Debugging
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # IDs that resulted in empty outputs previously (for debugging)
    debug_ids = {18, 43, 46, 53, 68}

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]

            graph_emb = torch.stack([x["embedding"] for x in batch]).to(model.device)
            mask = torch.stack([x["mask"] for x in batch]).to(model.device)
            prompts = [x["prompt"] for x in batch]

            # [DEBUG PROBE] Check input stats for problematic samples
            for i, item in enumerate(batch):
                if item["sample_idx"] in debug_ids:
                    emb_mean = graph_emb[i].mean().item()
                    emb_std = graph_emb[i].std().item()
                    mask_sum = mask[i].sum().item()
                    LOGGER.info(f"--- DEBUG Sample {item['sample_idx']} ---")
                    LOGGER.info(f"    Emb Mean: {emb_mean:.4f}, Std: {emb_std:.4f}")
                    LOGGER.info(f"    Mask Sum: {mask_sum} / {mask[i].numel()}")

            with torch.no_grad():
                preds = model.generate(
                    {"embedding": graph_emb, "mask": mask},
                    prompts,
                    **gen_kwargs,
                )

            for item, pred in zip(batch, preds):
                entry = {
                    "node_id": item["node_id"],
                    "sample_idx": item["sample_idx"],
                    "title": item["title"],
                    "prompt": item["prompt"],
                    "model_output": pred.strip(),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1
            
            if start % (batch_size * 5) == 0:
                LOGGER.info(f"Processed {written}/{len(records)} samples...")

    LOGGER.info(f"Done. {written} predictions saved to {out_path}")


if __name__ == "__main__":
    main()