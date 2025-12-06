# train/train_graphllm.py
import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup  # [新增] 引入调度器

from graphllm.model import GraphLLM
from train.dataset import GraphTeacherDataset, collate_teacher_batch


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphLLM projector with teacher labels.")
    parser.add_argument("--config", type=str, default="train/config.yaml", help="Path to YAML configuration.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_dtype(name: str) -> torch.dtype:
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


def prepare_text_batch(
    tokenizer,
    instructions: Sequence[str],
    targets: Sequence[str],
    answer_prefix: str,
    add_eos: bool,
    max_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # -----------------------------------------------------------
    # [恢复原始逻辑] 保持与你提供的代码完全一致
    # -----------------------------------------------------------
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if add_eos else None

    seqs = []
    labels = []
    max_seq_len = 0
    for instruction, target in zip(instructions, targets):
        prompt_text = f"{instruction}{answer_prefix}" if answer_prefix else instruction
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        if eos_id is not None:
            target_ids = target_ids + [eos_id]
        
        # 原始截断逻辑：先截断 prompt，再计算剩余空间给 target
        if max_length is not None and max_length > 0:
            prompt_ids = prompt_ids[:max_length]
            remaining = max_length - len(prompt_ids)
            if remaining <= 0:
                target_ids = []
            elif remaining < len(target_ids):
                target_ids = target_ids[:remaining]
        
        seq = prompt_ids + target_ids
        label = [-100] * len(prompt_ids) + target_ids
        
        # 简单容错：防止全部为空
        if not seq:
            raise ValueError("Prompt + target truncated to zero tokens.")
            
        seqs.append(seq)
        labels.append(label)
        max_seq_len = max(max_seq_len, len(seq))

    attn_masks = []
    for idx in range(len(seqs)):
        seq = seqs[idx]
        label = labels[idx]
        seq_len = len(seq)
        pad_len = max_seq_len - seq_len
        attn_masks.append([1] * seq_len + [0] * pad_len)
        seqs[idx] = seq + [pad_id] * pad_len
        labels[idx] = label + [-100] * pad_len

    input_ids = torch.tensor(seqs, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attn_masks, dtype=torch.long, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    return input_ids, attention_mask, label_tensor


def build_model(cfg: Dict[str, Any]) -> GraphLLM:
    model_cfg = cfg["model"]
    llm_kwargs = dict(model_cfg.get("llm_kwargs", {}))
    tokenizer_kwargs = dict(model_cfg.get("tokenizer_kwargs", {}))
    projector_kwargs = dict(model_cfg.get("projector", {}))

    dtype_name = model_cfg.get("llm_dtype")
    if dtype_name:
        llm_kwargs.setdefault("torch_dtype", parse_dtype(dtype_name))
    llm_kwargs.setdefault("trust_remote_code", True)
    tokenizer_kwargs.setdefault("use_fast", False)
    tokenizer_kwargs.setdefault("trust_remote_code", True)

    model = GraphLLM(
        graph_encoder=None,  # 确保为 None，使用预计算 Embedding
        llm_path=model_cfg["llm_path"],
        llm_kwargs=llm_kwargs,
        tokenizer_path=model_cfg.get("tokenizer_path"),
        tokenizer_kwargs=tokenizer_kwargs,
        projector_kwargs=projector_kwargs,
        instruction_max_len=model_cfg.get("text_max_len", 1024),
        graph_feature_type="sequence",  # 必须设置为 sequence
        device=model_cfg.get("device", "cpu"),
    )
    if cfg["training"].get("freeze_llm", True):
        model.llm.train()
        for param in model.llm.parameters():
            param.requires_grad = False
    return model


def train(cfg: Dict[str, Any]) -> None:
    training_cfg = cfg["training"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(output_dir / "graphllm_training.log")

    set_seed(training_cfg.get("seed", 42))
    LOGGER.info("Configuration: %s", cfg)

    # 1. 加载 Dataset
    dataset = GraphTeacherDataset(
        embeddings_path=data_cfg["embeddings_path"],
        teacher_csv=data_cfg["teacher_csv"],
        metadata_csv=data_cfg["metadata_csv"],
        instruction_template=model_cfg["instruction_template"],
        max_samples=data_cfg.get("max_samples"),
    )
    
    # [新增] 关键数据对齐检查
    # 确保 CSV 中的 ID 在 .pt 文件中确实存在，否则训练是无效的
    if hasattr(dataset, "embeddings_data") and hasattr(dataset, "metadata"):
        pt_ids = set(dataset.embeddings_data.keys())
        csv_ids = set(dataset.metadata["node_id"].unique())
        overlap = pt_ids & csv_ids
        LOGGER.info(f"DATA CHECK: {len(overlap)} nodes match between CSV metadata and PT embeddings.")
        if len(overlap) == 0:
            raise RuntimeError("CRITICAL: No matching node_ids found between CSV and PT file! Training will fail.")

    dataloader = DataLoader(
        dataset,
        batch_size=training_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_teacher_batch, 
    )

    # 2. 构建模型
    model = build_model(cfg)
    device = model.device
    model.projector.train()

    # 3. 优化器设置 (建议在 Config 中将 lr 设为 1e-3)
    optimizer = AdamW(
        model.projector.parameters(),
        lr=float(training_cfg.get("lr", 1e-3)), # 默认值提高到 1e-3
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    num_epochs = training_cfg.get("num_epochs", 1)
    grad_accum = max(1, training_cfg.get("grad_accum_steps", 1))
    
    # [新增] Scheduler 设置
    # 计算总步数，用于 Scheduler
    total_steps = len(dataloader) * num_epochs // grad_accum
    warmup_steps = int(total_steps * 0.03) # 3% Warmup
    LOGGER.info(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    log_interval = max(1, training_cfg.get("log_interval", 10))
    max_txt_len = model_cfg.get("text_max_len", 1024)
    answer_prefix = model_cfg.get("answer_prefix", "\n\nAnswer: ")
    add_eos = training_cfg.get("add_eos_token", True)
    max_grad_norm = training_cfg.get("max_grad_norm")

    global_step = 0
    running_loss = 0.0

    for epoch in range(num_epochs):
        LOGGER.info("Starting epoch %d / %d", epoch + 1, num_epochs)
        for step, batch in enumerate(dataloader, start=1):
            
            # 处理字典类型的 graph_batch，移动到 device
            graph_batch = batch["graph_batch"]
            if isinstance(graph_batch, dict):
                for k, v in graph_batch.items():
                    if isinstance(v, torch.Tensor):
                        graph_batch[k] = v.to(device)
            else:
                graph_batch = graph_batch.to(device)

            input_ids, attention_mask, labels = prepare_text_batch(
                tokenizer=model.tokenizer,
                instructions=batch["instructions"],
                targets=batch["targets"],
                answer_prefix=answer_prefix,
                add_eos=add_eos,
                max_length=max_txt_len,
                device=device,
            )
            
            outputs = model(
                graph_batch=graph_batch,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            batch_loss = outputs.loss.detach().item()
            loss = outputs.loss / grad_accum
            loss.backward()
            running_loss += batch_loss

            # 梯度累积步
            if (global_step + 1) % grad_accum == 0:
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.projector.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step() # [新增] 更新学习率
                optimizer.zero_grad()
            
            LOGGER.info(
                "Epoch %d | Step %d/%d | Global Step %d | Loss %.4f | LR %.2e",
                epoch + 1,
                step,
                len(dataloader),
                global_step + 1,
                batch_loss,
                optimizer.param_groups[0]["lr"] # 打印当前学习率
            )

            global_step += 1

            if global_step % log_interval == 0:
                avg_loss = running_loss / log_interval
                running_loss = 0.0
                LOGGER.info(
                    "Averaged loss over last %d steps: %.4f",
                    log_interval,
                    avg_loss,
                )

        LOGGER.info("Completed epoch %d", epoch + 1)

    save_path = Path(training_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_projector(save_path)
    LOGGER.info("Saved projector checkpoint to %s", save_path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()