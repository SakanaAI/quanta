from __future__ import annotations

import torch

from quanta.experiments.scaling_laws.model import CNANDTransformerModel


MASK_MODES = ("none", "other_bits", "only_parent_outs")


class MaskedCNANDTransformer(CNANDTransformerModel):
    """cNAND transformer with graph-structured communication constraints."""

    def __init__(
        self,
        *,
        config,
        n_slots: int,
        tokens_per_node: int,
        mask_mode: str,
        parent_indices: torch.Tensor,
        parent_masks: torch.Tensor,
    ):
        super().__init__(config=config, n_slots=n_slots)
        self.n_heads = int(config.n_heads)
        self.tokens_per_node = int(tokens_per_node)
        self.mask_mode = str(mask_mode or "none")
        self.register_buffer("graph_parent_indices", parent_indices.detach().clone(), persistent=False)
        self.register_buffer("graph_parent_masks", parent_masks.detach().clone(), persistent=False)

    def residual_stream(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mask_mode == "none":
            return super().residual_stream(batch)
        hidden = self.token_embedding(batch["input_ids"]) + self.slot_embedding(batch["slot_ids"])
        attention_mask = structured_attention_mask(
            batch["slot_ids"],
            tokens_per_node=self.tokens_per_node,
            n_heads=self.n_heads,
            mask_mode=self.mask_mode,
            parent_indices=self.graph_parent_indices,
            parent_masks=self.graph_parent_masks,
        )
        return self.encoder(
            hidden,
            mask=attention_mask,
            src_key_padding_mask=~batch["active_mask"],
            is_causal=False,
        )


def structured_attention_mask(
    slot_ids: torch.Tensor,
    *,
    tokens_per_node: int,
    n_heads: int,
    mask_mode: str,
    parent_indices: torch.Tensor | None = None,
    parent_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return True for query-key pairs disallowed by the selected mask."""
    if mask_mode not in MASK_MODES or mask_mode == "none":
        raise ValueError("structured_attention_mask requires a structured mask mode.")
    if slot_ids.ndim != 2:
        raise ValueError("slot_ids must have shape [batch, sequence].")
    if tokens_per_node < 2:
        raise ValueError("tokens_per_node must include at least one bit and OUT.")

    shared_layout = bool(
        slot_ids.shape[0] > 0
        and torch.equal(slot_ids, slot_ids[:1].expand_as(slot_ids))
    )
    if shared_layout:
        slot_ids = slot_ids[:1]

    node_ids = torch.div(slot_ids, tokens_per_node, rounding_mode="floor")
    is_out = slot_ids.remainder(tokens_per_node).eq(tokens_per_node - 1)
    query_is_out = is_out.unsqueeze(2)
    key_is_out = is_out.unsqueeze(1)
    same_node = node_ids.unsqueeze(2).eq(node_ids.unsqueeze(1))
    bit_query_allowed = ~query_is_out & key_is_out & same_node

    if mask_mode == "other_bits":
        out_query_allowed = query_is_out & (key_is_out | same_node)
    else:
        if parent_indices is None or parent_masks is None:
            raise ValueError("only_parent_outs requires parent indices and masks.")
        query_nodes = node_ids.unsqueeze(2)
        key_nodes = node_ids.unsqueeze(1)
        query_parents = parent_indices.clamp_min(0)[query_nodes]
        query_parent_masks = parent_masks[query_nodes]
        key_is_parent = (
            query_parents.eq(key_nodes.unsqueeze(-1)) & query_parent_masks
        ).any(dim=-1)
        out_query_allowed = query_is_out & (
            same_node | (key_is_out & (same_node | key_is_parent))
        )

    disallowed = ~(bit_query_allowed | out_query_allowed)
    if shared_layout:
        return disallowed[0]
    return disallowed.repeat_interleave(n_heads, dim=0)
