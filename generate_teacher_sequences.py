import argparse
import csv
from pathlib import Path
import sys

import torch
import yaml
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate teacher sequences using summary+instruction")
    parser.add_argument("--config", default="train/config.yaml")
    parser.add_argument("--output", default="train/output/teachers1000.csv")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--start-offset", type=int, default=0, help="Number of rows to skip before generating teachers")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer_kwargs = {
        "use_fast": False,
    }
    if cfg["model"].get("trust_remote_code"):
        tokenizer_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["llm_path"], **tokenizer_kwargs
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, cfg["model"].get("llm_dtype", "float32"))
    model_kwargs = {
        "torch_dtype": dtype,
    }
    if cfg["model"].get("trust_remote_code"):
        model_kwargs["trust_remote_code"] = True
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["llm_path"], **model_kwargs
    ).to(cfg["model"]["device"])
    model.eval()

    csv_path = cfg["data"].get("csv_path") or cfg["data"].get("metadata_csv")
    if not csv_path:
        raise KeyError("Config missing data.csv_path or data.metadata_csv")
    df = pd.read_csv(csv_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    def clean_teacher(text: str) -> str:
        markers = ["###", "USER:", "User:", "Human:", "Assistant:", "Please write in English language."]
        end = len(text)
        for marker in markers:
            pos = text.find(marker)
            if pos != -1:
                end = min(end, pos)
        cleaned = text[:end].strip()
        return cleaned

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "teacher_text"])
        start = min(max(args.start_offset, 0), len(df))
        end = min(start + args.max_samples, len(df))
        subset = df.iloc[start:end]
        for local_idx, (_, row) in enumerate(subset.iterrows()):
            paper_summary = str(row.get("paper_summary", "")).strip()
            title = str(row.get("title", "")).strip()
            context = f"Title: {title}\n\nPaper summary: {paper_summary}"
            text_instruction = cfg["model"]["instruction_template"].format(
                context=context,
                title=title,
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant that rewrites paper summaries."},
                {"role": "user", "content": text_instruction},
            ]
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = ""
                for msg in messages:
                    role = msg["role"].capitalize()
                    prompt += f"{role}: {msg['content']}\n"
                prompt += "Assistant:"
            prompt_max_len = (
                cfg["model"].get("text_prompt_max_len")
                or cfg["model"].get("text_max_len")
                or cfg["model"].get("max_txt_len")
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=prompt_max_len,
            ).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                )
            prompt_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][prompt_len:]
            teacher_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            teacher_text = clean_teacher(teacher_text)
            sample_idx = row.get("sample_idx")
            if pd.isna(sample_idx):
                sample_idx = start + local_idx
            writer.writerow((int(sample_idx), teacher_text))
            f.flush()
            if (local_idx + 1) % 10 == 0:
                print(f"Generated {local_idx+1}/{len(subset)} teacher sequences (offset {start})")
    print(f"Saved teacher sequences to {output_path}")


if __name__ == "__main__":
    main()
