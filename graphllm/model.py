"""
GraphLLM architecture: GraphCLIP encoder + translator-style projector + LLM (e.g., Qwen3).
"""
# model.py
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from GraphCLIP.models import GraphCLIP as GraphCLIPEncoder
except ImportError:  # pragma: no cover
    GraphCLIPEncoder = None


def _as_list(value: Union[int, Sequence[int], None]) -> List[int]:
    if value is None:
        return []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return list(value)
    return [int(value)]


def _get_llm_hidden_size(config: Any) -> int:
    for attr in ("hidden_size", "n_embd", "d_model", "dim"):
        hidden = getattr(config, attr, None)
        if hidden is not None:
            return int(hidden)
    raise ValueError("Cannot infer LLM hidden size from config.")


class TranslatorProjector(nn.Module):
    """Cross-attention projector inspired by Translator stage-2."""

    def __init__(
        self,
        graph_dim: int,
        projector_dim: int,
        llm_hidden_size: int,
        num_query_tokens: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.num_query_tokens = int(num_query_tokens)
        self.projector_dim = int(projector_dim)
        ffn_dim = ffn_dim or self.projector_dim * 4

        self.graph_proj = nn.Linear(graph_dim, self.projector_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.projector_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.query_tokens = nn.Parameter(
            torch.randn(self.num_query_tokens, self.projector_dim)
        )
        self.out_proj = nn.Linear(self.projector_dim, llm_hidden_size)

    def forward(
        self,
        graph_emb: torch.Tensor,
        graph_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            graph_emb: (B, D) or (B, S, D) tensor.
            graph_attention_mask: optional (B, S) mask (1=keep, 0=pad).
        Returns:
            (B, num_query_tokens, llm_hidden_size)
        """
        if graph_emb.dim() == 2:
            graph_seq = graph_emb.unsqueeze(1)  # B, 1, D
        elif graph_emb.dim() == 3:
            graph_seq = graph_emb
        else:
            raise ValueError("graph_emb must be rank-2 or rank-3 tensor.")

        batch_size = graph_seq.size(0)
        memory = self.graph_proj(graph_seq).transpose(0, 1)  # S, B, d
        tgt = self.query_tokens.unsqueeze(1).expand(-1, batch_size, -1)  # T, B, d
        if graph_attention_mask is not None:
            # Transformer expects shape (B, S)
            key_padding_mask = graph_attention_mask == 0
        else:
            key_padding_mask = None
        decoded = self.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=key_padding_mask,
        )
        decoded = decoded.permute(1, 0, 2)  # B, T, d
        return self.out_proj(decoded)


class GraphLLM(nn.Module):
    """Full pipeline that couples GraphCLIP with an LLM via translator-style projector."""

    def __init__(
        self,
        *,
        llm_path: str,
        graph_encoder: Optional[nn.Module] = None,
        graph_encoder_cfg: Optional[Dict[str, Any]] = None,
        graph_encoder_ckpt: Optional[Union[str, Path]] = None,
        freeze_graph_encoder: bool = True,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_path: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        projector_kwargs: Optional[Dict[str, Any]] = None,
        instruction_max_len: int = 1024,
        graph_feature_type: str = "graph",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or "cpu")
        self.instruction_max_len = instruction_max_len
        self.graph_feature_type = graph_feature_type
        self.graph_encoder_frozen = freeze_graph_encoder

        llm_kwargs = llm_kwargs or {}
        tokenizer_kwargs = tokenizer_kwargs or {}
        projector_kwargs = projector_kwargs or {}

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            **llm_kwargs,
        )
        if llm_kwargs.get("device_map") is None:
            self.llm.to(self.device)
        tokenizer_path = tokenizer_path or llm_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            **tokenizer_kwargs,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.d_llm = _get_llm_hidden_size(self.llm.config)

        graph_dim = projector_kwargs.get("graph_dim")
        if graph_dim is None:
            if graph_encoder_cfg and "graph_hid_dim" in graph_encoder_cfg:
                graph_dim = graph_encoder_cfg["graph_hid_dim"]
            else:
                raise ValueError(
                    "projector_kwargs['graph_dim'] is required when graph encoder cfg does not "
                    "specify 'graph_hid_dim'."
                )

        proj_dim = projector_kwargs.get("projector_dim", self.d_llm)
        num_query_tokens = projector_kwargs.get("num_query_tokens", 32)
        projector = TranslatorProjector(
            graph_dim=graph_dim,
            projector_dim=proj_dim,
            llm_hidden_size=self.d_llm,
            num_query_tokens=num_query_tokens,
            num_layers=projector_kwargs.get("num_layers", 2),
            num_heads=projector_kwargs.get("num_heads", 8),
            ffn_dim=projector_kwargs.get("ffn_dim"),
            dropout=projector_kwargs.get("dropout", 0.1),
            activation=projector_kwargs.get("activation", "gelu"),
        )
        self.projector = projector
        self.num_prefix_tokens = self.projector.num_query_tokens

        self.graph_encoder = graph_encoder
        if self.graph_encoder is None and graph_encoder_cfg is not None:
            if GraphCLIPEncoder is None:
                raise ImportError("GraphCLIP package is required to build the graph encoder.")
            self.graph_encoder = GraphCLIPEncoder(**graph_encoder_cfg)
        if self.graph_encoder is not None:
            if graph_encoder_ckpt:
                state = torch.load(graph_encoder_ckpt, map_location="cpu")
                self.graph_encoder.load_state_dict(state, strict=False)
            self.graph_encoder.to(self.device)
            if freeze_graph_encoder:
                for param in self.graph_encoder.parameters():
                    param.requires_grad = False
                self.graph_encoder.eval()

        self.projector.to(self.device)

    # ------------------------------------------------------------------
    # Graph encoding helpers
    # ------------------------------------------------------------------
    def _select_graph_feature(
        self,
        features: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            return features
        if not isinstance(features, (tuple, list)) or not features:
            raise ValueError("Unexpected graph encoder output.")
        if self.graph_feature_type == "center" and len(features) > 1:
            return features[1]
        if self.graph_feature_type in {"sequence", "nodes"} and len(features) > 2:
            return features[2]
        return features[0]

    def _pad_node_embeddings(
        self,
        node_embs: torch.Tensor,
        batch_index: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_index.numel() == 0:
            raise ValueError("Empty batch_index for node embeddings.")
        node_embs = node_embs.to(device)
        batch_index = batch_index.to(device)
        num_graphs = int(batch_index.max().item()) + 1
        if num_graphs <= 0:
            raise ValueError("Invalid num_graphs derived from batch_index.")

        counts = [(batch_index == gid).sum().item() for gid in range(num_graphs)]
        max_nodes = int(max(counts))
        if max_nodes == 0:
            raise ValueError("All graphs appear to be empty.")

        padded = []
        masks = []
        for gid, count in enumerate(counts):
            nodes_g = node_embs[batch_index == gid]
            pad_len = max_nodes - int(count)
            if pad_len > 0:
                nodes_g = torch.cat(
                    [nodes_g, node_embs.new_zeros((pad_len, node_embs.size(-1)))],
                    dim=0,
                )
            padded.append(nodes_g)
            masks.append(
                torch.cat(
                    [
                        torch.ones(int(count), device=device, dtype=torch.long),
                        torch.zeros(pad_len, device=device, dtype=torch.long),
                    ],
                    dim=0,
                )
            )
        return torch.stack(padded, dim=0), torch.stack(masks, dim=0)

    def encode_graph(
        self,
        batch: Union[torch.Tensor, Any],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        use_node_sequence = self.graph_feature_type in {"sequence", "nodes"}
        if self.graph_encoder is None:
            # Accept raw tensors or (embedding, mask) / {"embedding","mask"} tuples when no encoder.
            if isinstance(batch, dict) and "embedding" in batch:
                graph_emb = batch["embedding"]
                attention_mask = batch.get("mask", attention_mask)
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                graph_emb, attention_mask = batch
            elif isinstance(batch, torch.Tensor):
                graph_emb = batch
            else:
                raise TypeError("Expect tensor embeddings or (embedding, mask) when graph_encoder is None.")
            graph_emb = graph_emb.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            return graph_emb, attention_mask

        data = batch
        if hasattr(data, "to"):
            data = data.to(self.device)

        forward_ctx = (
            torch.no_grad()
            if self.graph_encoder_frozen or not self.training
            else contextlib.nullcontext()
        )
        with forward_ctx:
            if use_node_sequence:
                if hasattr(self.graph_encoder, "encode_nodes_only"):
                    features = self.graph_encoder.encode_nodes_only(data)
                elif hasattr(self.graph_encoder, "encode_graph_with_nodes"):
                    features = self.graph_encoder.encode_graph_with_nodes(data)
                else:
                    raise ValueError(
                        "Graph encoder does not expose node-level embeddings. "
                        "Provide precomputed node sequences or use graph_feature_type='graph'."
                    )
            else:
                features = self.graph_encoder.encode_graph(data)

        # If we requested node sequences and got them, build padded outputs + mask.
        if use_node_sequence and isinstance(features, (tuple, list)):
            if len(features) >= 4:
                node_embs = features[2]
                node_batch = features[3]
                graph_emb, node_mask = self._pad_node_embeddings(
                    node_embs, node_batch, device=self.device
                )
                return graph_emb, node_mask
            if len(features) == 2 and isinstance(features[0], torch.Tensor) and isinstance(features[1], torch.Tensor):
                node_embs, node_batch = features
                graph_emb, node_mask = self._pad_node_embeddings(
                    node_embs, node_batch, device=self.device
                )
                return graph_emb, node_mask

        graph_emb = self._select_graph_feature(features).to(self.device)
        return graph_emb, attention_mask

    def project_graph_embedding(
        self,
        graph_emb: torch.Tensor,
        graph_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prefix = self.projector(
            graph_emb.to(self.projector.graph_proj.weight.device),
            graph_attention_mask=graph_attention_mask,
        )
        return prefix.to(self.device)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def tokenize_instructions(
        self,
        instructions: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized = self.tokenizer(
            list(instructions),
            padding=True,
            truncation=True,
            max_length=self.instruction_max_len,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        return input_ids, attention_mask

    def build_inputs(
        self,
        graph_emb: torch.Tensor,
        *,
        graph_attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        instructions: Optional[Sequence[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if instructions is not None:
            input_ids, attention_mask = self.tokenize_instructions(instructions)
        if input_ids is None or attention_mask is None:
            raise ValueError("Either tokenized inputs or instructions must be provided.")

        embed_layer = self.llm.get_input_embeddings()
        embed_device = embed_layer.weight.device
        input_ids = input_ids.to(embed_device)
        attention_mask = attention_mask.to(embed_device)
        token_embeds = embed_layer(input_ids)
        prefix_embeds = self.project_graph_embedding(
            graph_emb,
            graph_attention_mask=graph_attention_mask,
        ).to(embed_device, dtype=token_embeds.dtype)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        batch_size = attention_mask.size(0)
        prefix_mask = torch.ones(
            (batch_size, self.num_prefix_tokens),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention = torch.cat([prefix_mask, attention_mask], dim=1)
        return inputs_embeds, full_attention, self.num_prefix_tokens

    # ------------------------------------------------------------------
    # Training forward + generation
    # ------------------------------------------------------------------
    def forward(
        self,
        graph_batch: Union[torch.Tensor, Any],
        *,
        graph_attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        instructions: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> Any:
        graph_emb, graph_attention_mask = self.encode_graph(
            graph_batch,
            attention_mask=graph_attention_mask,
        )
        inputs_embeds, full_attention, prompt_len = self.build_inputs(
            graph_emb,
            graph_attention_mask=graph_attention_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            instructions=instructions,
        )
        if labels is not None:
            labels = labels.to(self.device)
            if labels.size(1) != full_attention.size(1) - prompt_len:
                raise ValueError("Labels must align with provided input_ids length.")
            prefix_ignore = torch.full(
                (labels.size(0), prompt_len),
                fill_value=-100,
                dtype=labels.dtype,
                device=self.device,
            )
            labels = torch.cat([prefix_ignore, labels], dim=1)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            labels=labels,
            **kwargs,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        graph_batch: Union[torch.Tensor, Any],
        instructions: Sequence[str],
        *,
        graph_attention_mask: Optional[torch.Tensor] = None,
        **gen_kwargs: Any,
    ) -> List[str]:
        self.eval()
        graph_emb, graph_attention_mask = self.encode_graph(
            graph_batch,
            attention_mask=graph_attention_mask,
        )
        inputs_embeds, attention_mask, prompt_len = self.build_inputs(
            graph_emb,
            graph_attention_mask=graph_attention_mask,
            instructions=instructions,
        )
        gen_kwargs.setdefault("pad_token_id", self.pad_token_id)
        if "eos_token_id" not in gen_kwargs and self.tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        sequences = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
        # Some HF models return only the generated tokens when inputs_embeds are provided.
        if sequences.size(1) <= prompt_len:
            new_tokens = sequences
        else:
            new_tokens = sequences[:, prompt_len:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def save_projector(self, path: Union[str, Path]) -> None:
        torch.save(self.projector.state_dict(), path)

    def load_projector(self, path: Union[str, Path], strict: bool = True) -> None:
        state = torch.load(path, map_location="cpu")
        self.projector.load_state_dict(state, strict=strict)
