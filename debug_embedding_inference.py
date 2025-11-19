import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from graphllm.model import GraphLLM
from train.dataset import GraphTeacherDataset


LOGGER = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify GraphLLM projector by running inference on GraphCLIP embeddings."
    )
    parser.add_argument("--config", type=str, default="train/config.yaml", help="Path to training config.")
    parser.add_argument(
        "--projector",
        type=str,
        default="train/output/projector_stage1.pt",
        help="Checkpoint path for trained projector.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="How many samples from the teacher dataset to run.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling probability.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Penalty factor applied to repeated tokens.",
    )
    return parser.parse_args()


def build_model(cfg: Dict[str, Any]) -> GraphLLM:
    model_cfg = cfg["model"]
    llm_kwargs = dict(model_cfg.get("llm_kwargs", {}))
    tokenizer_kwargs = dict(model_cfg.get("tokenizer_kwargs", {}))
    projector_kwargs = dict(model_cfg.get("projector", {}))

    dtype_name = model_cfg.get("llm_dtype")
    if dtype_name:
        dtype = getattr(torch, dtype_name.lower(), None)
        if dtype is None:
            raise ValueError(f"Unsupported dtype '{dtype_name}' in config.")
        llm_kwargs.setdefault("torch_dtype", dtype)
    llm_kwargs.setdefault("trust_remote_code", True)
    tokenizer_kwargs.setdefault("use_fast", False)
    tokenizer_kwargs.setdefault("trust_remote_code", True)

    model = GraphLLM(
        graph_encoder=None,
        llm_path=model_cfg["llm_path"],
        llm_kwargs=llm_kwargs,
        tokenizer_path=model_cfg.get("tokenizer_path"),
        tokenizer_kwargs=tokenizer_kwargs,
        projector_kwargs=projector_kwargs,
        instruction_max_len=model_cfg.get("text_max_len", 512),
        device=model_cfg.get("device", "cpu"),
    )
    return model


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    LOGGER.info("Loading dataset and model...")

    dataset = GraphTeacherDataset(
        embeddings_path=cfg["data"]["embeddings_path"],
        teacher_csv=cfg["data"]["teacher_csv"],
        metadata_csv=cfg["data"]["metadata_csv"],
        instruction_template=cfg["model"]["instruction_template"],
        max_samples=cfg["data"].get("max_samples"),
    )

    model = build_model(cfg)
    projector_path = Path(args.projector)
    if not projector_path.exists():
        raise FileNotFoundError(f"Projector checkpoint not found: {projector_path}")
    state = torch.load(projector_path, map_location="cpu")
    model.projector.load_state_dict(state)
    model.projector.to(model.device)
    model.projector.eval()
    model.llm.eval()

    summary_prompt_template = cfg["model"].get(
        "generate_prompt",
        "Question: We are trying to explore the paper titled <{title}>. "
        "Please summarize the topic and content of the paper and its citations in English.\n\n"
        "Answer:",
    )

    num_samples = min(args.num_samples, len(dataset))
    samples = []
    graph_embs = []
    formatted_prompts = []
    targets = []
    for i in range(num_samples):
        item = dataset[i]
        samples.append(item)
        graph_embs.append(item["embedding"])
        targets.append(item["target"])

        title_text = item.get("title", "")
        if not title_text:
            try:
                sample_idx = item["sample_idx"]
                if "title" in dataset.metadata.columns:
                    title_text = str(dataset.metadata.loc[sample_idx, "title"])
                else:
                    LOGGER.warning("Column 'title' not found in metadata CSV.")
                    title_text = "Unknown Title"
            except Exception as exc:  # pragma: no cover - best effort logging
                LOGGER.warning("Failed to extract title for sample_idx %s: %s", item.get("sample_idx"), exc)
                title_text = "Unknown Title"

        prompt = summary_prompt_template.format(title=title_text).strip()
        formatted_prompts.append(prompt)

    graph_emb = torch.stack(graph_embs).to(model.device)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.temperature > 0.0:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p
        gen_kwargs["do_sample"] = True
    else:
        gen_kwargs["do_sample"] = False
    LOGGER.info(
        "Running generation for %d samples with temp=%.2f, repetition_penalty=%.2f",
        num_samples,
        args.temperature,
        args.repetition_penalty,
    )
    outputs = model.generate(graph_emb, formatted_prompts, **gen_kwargs)

    for idx, (sample, prompt, target, prediction) in enumerate(
        zip(samples, formatted_prompts, targets, outputs)
    ):
        print("=" * 80)
        print(f"Sample #{idx} | node_id={sample['node_id']} | sample_idx={sample['sample_idx']}")
        print("Instruction:")
        print(prompt.strip())
        print("-" * 20)
        print("Teacher Target (first 200 chars):")
        print(target.strip()[:200] + "...")
        print("-" * 20)
        print("Model Output:")
        print(prediction.strip())
        print("=" * 80)


if __name__ == "__main__":
    main()
