from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from quanta.qprogram.neural import ParameterAudit, audit_homogeneous_parameters
from quanta.qprogram.types import CompiledQProgram, canonical_key


RoutingMode = Literal["oracle", "predicted"]


@dataclass(frozen=True)
class ParentIntervention:
    """Evaluation-only intervention on compiled parent communication."""

    kind: Literal["none", "zero", "shuffle", "ablate"] = "none"
    node: str | None = None
    parent: str | None = None
    permutation: torch.Tensor | None = None


@dataclass(frozen=True)
class QCoreTrace:
    source_query_state: torch.Tensor
    read_keys: torch.Tensor
    read_values: torch.Tensor
    ancestor_aggregates: torch.Tensor
    read_vectors: torch.Tensor
    gate_logits: torch.Tensor
    local_gate_probabilities: torch.Tensor
    effective_activity: torch.Tensor
    deltas: torch.Tensor
    messages: torch.Tensor
    read_attention: torch.Tensor
    depth_updates: torch.Tensor
    final_state: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True)
class ActivityLossResult:
    loss: torch.Tensor
    observed_nodes: int
    single_class_nodes: tuple[str, ...]
    per_node: dict[str, torch.Tensor]


@dataclass(frozen=True)
class SemanticLossResult:
    loss: torch.Tensor
    observed_nodes: int
    supervised_examples: int
    per_node: dict[str, torch.Tensor]


class SemanticClassifierSuite(nn.Module):
    """Training-only linear decoders for small compiler semantic domains."""

    def __init__(
        self,
        *,
        compiled: CompiledQProgram,
        d_quantum: int,
        max_supervised_classes: int,
    ) -> None:
        super().__init__()
        if (
            not isinstance(max_supervised_classes, int)
            or isinstance(max_supervised_classes, bool)
            or max_supervised_classes < 0
        ):
            raise ValueError("max_supervised_classes must be a non-negative integer.")
        if int(d_quantum) <= 0:
            raise ValueError("d_quantum must be positive.")
        self.nodes = tuple(compiled.nodes)
        self.node_to_index = {node: index for index, node in enumerate(self.nodes)}
        self.max_supervised_classes = int(max_supervised_classes)
        self.outcome_keys: dict[str, tuple[str, ...]] = {}
        self.class_indices: dict[str, dict[str, int]] = {}
        self.module_keys: dict[str, str] = {}
        self.excluded_by_cardinality: dict[str, int] = {}
        self.constant_nodes: tuple[str, ...]

        classifiers = {}
        constant_nodes = []
        for node_index, node in enumerate(self.nodes):
            keys = tuple(sorted({canonical_key(value) for value in compiled.semantic_outcomes[node]}))
            class_count = len(keys)
            if class_count < 2:
                constant_nodes.append(node)
                continue
            if class_count > self.max_supervised_classes:
                self.excluded_by_cardinality[node] = class_count
                continue
            module_key = f"node_{node_index}"
            self.outcome_keys[node] = keys
            self.class_indices[node] = {key: index for index, key in enumerate(keys)}
            self.module_keys[node] = module_key
            classifiers[module_key] = nn.Linear(int(d_quantum), class_count)
        self.constant_nodes = tuple(constant_nodes)
        self.classifiers = nn.ModuleDict(classifiers)
        self.supervised_nodes = tuple(node for node in self.nodes if node in self.module_keys)

    def encode_targets(
        self,
        semantics: Sequence[Sequence[tuple[object | None, ...] | None]],
        activity_targets: torch.Tensor,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Encode active semantic values; inactive and unsupervised entries are -100."""
        if activity_targets.ndim != 3 or activity_targets.shape[-1] != len(self.nodes):
            raise ValueError("activity_targets must have shape [batch, query_length, quantum].")
        batch, query_length, _ = activity_targets.shape
        if len(semantics) != batch or any(len(row) != query_length for row in semantics):
            raise ValueError("semantic targets must match activity target batch and query dimensions.")
        targets = torch.full(
            (batch, query_length, len(self.nodes)),
            -100,
            dtype=torch.long,
            device=device,
        )
        active = activity_targets.detach().to(device="cpu", dtype=torch.bool).tolist()
        for row_index, semantic_row in enumerate(semantics):
            for position, values in enumerate(semantic_row):
                if values is None:
                    continue
                if len(values) != len(self.nodes):
                    raise ValueError("each semantic target row must contain one value per quantum.")
                for node in self.supervised_nodes:
                    node_index = self.node_to_index[node]
                    if not active[row_index][position][node_index]:
                        continue
                    key = canonical_key(values[node_index])
                    try:
                        target = self.class_indices[node][key]
                    except KeyError as exc:
                        raise ValueError(
                            f"semantic target for {node!r} is absent from the compiler outcome catalog: {key}"
                        ) from exc
                    targets[row_index, position, node_index] = target
        return targets

    def logits(self, node: str, deltas: torch.Tensor) -> torch.Tensor:
        return self.classifiers[self.module_keys[node]](deltas[..., self.node_to_index[node], :])

    def catalog(self) -> dict[str, Any]:
        return {
            "max_supervised_classes": self.max_supervised_classes,
            "supervised_nodes": {
                node: {
                    "class_count": len(self.outcome_keys[node]),
                    "outcome_keys": list(self.outcome_keys[node]),
                }
                for node in self.supervised_nodes
            },
            "excluded_by_cardinality": dict(self.excluded_by_cardinality),
            "constant_nodes": list(self.constant_nodes),
        }


class _QuantumOperator(nn.Module):
    def __init__(self, d_quantum: int, mlp_ratio: float, activation: str) -> None:
        super().__init__()
        hidden = max(1, int(round(int(d_quantum) * float(mlp_ratio))))
        self.normalization = nn.LayerNorm(int(d_quantum))
        self.mlp_input = nn.Linear(int(d_quantum), hidden)
        self.mlp_output = nn.Linear(hidden, int(d_quantum))
        self.gate = nn.Linear(int(d_quantum), 1)
        self.read_bias = nn.Parameter(torch.zeros(int(d_quantum)))
        self.activation = str(activation)
        self.hidden_normalization = nn.LayerNorm(hidden) if self.activation == "layernorm_gelu" else None

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_state = self.normalization(value)
        hidden = self.mlp_input(local_state)
        if self.activation == "gelu":
            hidden = F.gelu(hidden)
        elif self.activation == "layernorm_gelu":
            assert self.hidden_normalization is not None
            hidden = F.gelu(self.hidden_normalization(hidden))
        elif self.activation == "relu":
            hidden = F.relu(hidden)
        else:
            raise ValueError("activation must be 'gelu', 'relu', or 'layernorm_gelu'.")
        return local_state, self.gate(local_state).squeeze(-1), self.mlp_output(hidden)


class CompiledQCore(nn.Module):
    """Neural execution of a compiler-generated quanta DAG."""

    def __init__(
        self,
        *,
        compiled: CompiledQProgram,
        d_source: int,
        d_quantum: int,
        vocab_size: int,
        read_heads: int = 1,
        mlp_ratio: float = 1.0,
        activation: str = "gelu",
        add_initial_state: bool = True,
        all_quanta_output: bool = True,
    ) -> None:
        super().__init__()
        if not compiled.nodes:
            raise ValueError("compiled Q-program must contain at least one quantum.")
        if int(d_quantum) <= 0 or int(d_source) <= 0:
            raise ValueError("d_source and d_quantum must be positive.")
        if int(read_heads) <= 0 or int(d_quantum) % int(read_heads) != 0:
            raise ValueError("read_heads must be positive and divide d_quantum.")
        if float(mlp_ratio) <= 0.0:
            raise ValueError("mlp_ratio must be positive.")
        if activation not in {"gelu", "relu", "layernorm_gelu"}:
            raise ValueError("activation must be 'gelu', 'relu', or 'layernorm_gelu'.")
        if not isinstance(add_initial_state, bool):
            raise ValueError("add_initial_state must be true or false.")
        if not isinstance(all_quanta_output, bool):
            raise ValueError("all_quanta_output must be true or false.")

        self.compiled = compiled
        self.nodes = tuple(compiled.nodes)
        self.node_to_index = {node: index for index, node in enumerate(self.nodes)}
        self.topological_order = tuple(compiled.structure.topological_order)
        self.topological_indices = tuple(self.node_to_index[node] for node in self.topological_order)
        self.parents = {
            node: tuple(compiled.structure.parents[node])
            for node in self.nodes
        }
        self.parent_indices = {
            node: tuple(self.node_to_index[parent] for parent in self.parents[node])
            for node in self.nodes
        }
        self.levels = tuple(tuple(level) for level in compiled.structure.levels)
        self.d_source = int(d_source)
        self.d_quantum = int(d_quantum)
        self.read_heads = int(read_heads)
        self.head_dim = self.d_quantum // self.read_heads
        self.activation = str(activation)
        self.add_initial_state = add_initial_state
        self.all_quanta_output = all_quanta_output
        self.output_nodes = self.nodes if all_quanta_output else tuple(compiled.structure.leaves)

        self.source_projection = nn.Linear(self.d_source, self.d_quantum)
        self.key_projection = nn.Linear(self.d_source, self.d_quantum)
        self.value_projection = nn.Linear(self.d_source, self.d_quantum)
        self.query_projection = nn.Linear(self.d_quantum, self.d_quantum)
        self.attention_output = nn.Linear(self.d_quantum, self.d_quantum)
        self.read_normalization = nn.LayerNorm(self.d_quantum)
        self.node_modules = nn.ModuleDict(
            {
                node: _QuantumOperator(self.d_quantum, float(mlp_ratio), activation)
                for node in self.nodes
            }
        )
        self.output_head = nn.Linear(self.d_quantum, int(vocab_size))

        self._validate_parent_signatures()
        self.parameter_audit()

    @property
    def depth(self) -> int:
        return len(self.levels)

    def forward(
        self,
        *,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        query_positions: torch.Tensor,
        routing: RoutingMode,
        activity_targets: torch.Tensor | None = None,
        parent_intervention: ParentIntervention | None = None,
        topology_override: Mapping[str, tuple[str, ...]] | None = None,
    ) -> QCoreTrace:
        if routing not in {"oracle", "predicted"}:
            raise ValueError("routing must be 'oracle' or 'predicted'.")
        if memory.ndim != 3:
            raise ValueError("memory must have shape [batch, memory_length, d_source].")
        batch, memory_length, _ = memory.shape
        if tuple(memory_mask.shape) != (batch, memory_length):
            raise ValueError("memory_mask must have shape [batch, memory_length].")
        if tuple(segment_ids.shape) != (batch, memory_length):
            raise ValueError("segment_ids must have shape [batch, memory_length].")
        if query_positions.ndim != 2 or query_positions.shape[0] != batch:
            raise ValueError("query_positions must have shape [batch, query_length].")
        query_length = int(query_positions.shape[1])
        if bool(((query_positions < 0) | (query_positions >= memory_length)).any()):
            raise ValueError("query_positions must index the source memory.")
        if routing == "oracle":
            self._validate_oracle_targets(activity_targets, batch, query_length)
        elif activity_targets is not None:
            raise ValueError("predicted routing does not accept activity_targets.")

        parent_intervention = parent_intervention or ParentIntervention()
        parent_map = self._validated_parent_override(topology_override)
        ancestor_map = self._ancestor_map(parent_map)
        row = torch.arange(batch, device=memory.device).unsqueeze(1)
        current_memory = memory[row, query_positions]
        source_state = self.source_projection(current_memory)
        keys = self.key_projection(memory)
        values = self.value_projection(memory)
        query_segments = segment_ids[row, query_positions]
        visible = self._visibility_mask(
            memory_mask=memory_mask,
            segment_ids=segment_ids,
            query_segments=query_segments,
            query_positions=query_positions,
        )

        messages: dict[str, torch.Tensor] = {}
        effective: dict[str, torch.Tensor] = {}
        ancestor_values: dict[str, torch.Tensor] = {}
        reads: dict[str, torch.Tensor] = {}
        logits: dict[str, torch.Tensor] = {}
        gate_probabilities: dict[str, torch.Tensor] = {}
        deltas: dict[str, torch.Tensor] = {}
        attentions: dict[str, torch.Tensor] = {}

        for node in self.topological_order:
            parents = parent_map[node]
            ancestor_aggregate = self._ancestor_aggregate(
                node=node,
                ancestors=ancestor_map[node],
                messages=messages,
                intervention=parent_intervention,
                like=source_state,
            )
            read, attention = self._read(
                source_state + ancestor_aggregate,
                node=node,
                keys=keys,
                values=values,
                visible=visible,
            )
            _, gate_logit, delta = self.node_modules[node](source_state + ancestor_aggregate + read)
            gate_probability = torch.sigmoid(gate_logit)
            if routing == "oracle":
                assert activity_targets is not None
                activity = activity_targets[..., self.node_to_index[node]].to(delta.dtype)
            else:
                activity = gate_probability
                for parent in parents:
                    activity = torch.minimum(activity, effective[parent])
            message_activity = activity.detach() if routing == "predicted" else activity
            message = message_activity.unsqueeze(-1) * delta
            ancestor_values[node] = ancestor_aggregate
            reads[node] = read
            logits[node] = gate_logit
            gate_probabilities[node] = gate_probability
            effective[node] = activity
            deltas[node] = delta
            messages[node] = message
            attentions[node] = attention

        message_tensor = self._stack(messages)
        depth_updates = torch.stack(
            [sum((messages[node] for node in level), torch.zeros_like(source_state)) for level in self.levels],
            dim=-2,
        )
        final_state = torch.stack([messages[node] for node in self.output_nodes], dim=-2).sum(dim=-2)
        if self.add_initial_state:
            final_state = source_state + final_state
        output_logits = self.output_head(final_state)
        return QCoreTrace(
            source_query_state=source_state,
            read_keys=keys,
            read_values=values,
            ancestor_aggregates=self._stack(ancestor_values),
            read_vectors=self._stack(reads),
            gate_logits=self._stack_scalars(logits),
            local_gate_probabilities=self._stack_scalars(gate_probabilities),
            effective_activity=self._stack_scalars(effective),
            deltas=self._stack(deltas),
            messages=message_tensor,
            read_attention=torch.stack([attentions[node] for node in self.nodes], dim=-2),
            depth_updates=depth_updates,
            final_state=final_state,
            logits=output_logits,
        )

    def parameter_audit(
        self,
        *,
        extra_shared: Mapping[str, nn.Module | torch.Tensor | nn.Parameter] | None = None,
    ) -> ParameterAudit:
        shared: dict[str, nn.Module | torch.Tensor | nn.Parameter] = {
            "source_projection": self.source_projection,
            "key_projection": self.key_projection,
            "value_projection": self.value_projection,
            "query_projection": self.query_projection,
            "attention_output": self.attention_output,
            "read_normalization": self.read_normalization,
            "output_head": self.output_head,
        }
        shared.update(extra_shared or {})
        return audit_homogeneous_parameters(self.node_modules, shared)

    def _read(
        self,
        condition: torch.Tensor,
        *,
        node: str,
        keys: torch.Tensor,
        values: torch.Tensor,
        visible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_projection(self.read_normalization(condition))
        query = query + self.node_modules[node].read_bias
        query_heads = self._split_heads(query)
        key_heads = self._split_heads(keys)
        value_heads = self._split_heads(values)
        scores = torch.einsum("bhqd,bhkd->bhqk", query_heads, key_heads) / math.sqrt(float(self.head_dim))
        scores = scores.masked_fill(~visible.unsqueeze(1), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * visible.unsqueeze(1).to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        read = torch.einsum("bhqk,bhkd->bhqd", attention, value_heads)
        read = read.transpose(1, 2).contiguous().view(*query.shape[:-1], self.d_quantum)
        return self.attention_output(read), attention.mean(dim=1)

    def _visibility_mask(
        self,
        *,
        memory_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        query_segments: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        key_positions = torch.arange(memory_mask.shape[1], device=memory_mask.device).view(1, 1, -1)
        causal = key_positions <= query_positions.unsqueeze(-1)
        segment_visible = segment_ids.unsqueeze(1) <= query_segments.unsqueeze(-1)
        return causal & segment_visible & memory_mask.to(torch.bool).unsqueeze(1)

    def _ancestor_aggregate(
        self,
        *,
        node: str,
        ancestors: tuple[str, ...],
        messages: dict[str, torch.Tensor],
        intervention: ParentIntervention,
        like: torch.Tensor,
    ) -> torch.Tensor:
        if intervention.kind == "zero" and (intervention.node is None or intervention.node == node):
            return torch.zeros_like(like)
        values = []
        for ancestor in ancestors:
            value = messages[ancestor]
            if intervention.kind == "ablate" and intervention.node == node and intervention.parent == ancestor:
                value = torch.zeros_like(value)
            if intervention.kind == "shuffle" and (intervention.node is None or intervention.node == node):
                permutation = intervention.permutation
                if permutation is None:
                    permutation = torch.arange(value.shape[0] - 1, -1, -1, device=value.device)
                value = value.index_select(0, permutation.to(value.device))
            values.append(value)
        return sum(values, torch.zeros_like(like))

    def _ancestor_map(
        self,
        parents: Mapping[str, tuple[str, ...]],
    ) -> dict[str, tuple[str, ...]]:
        """Expand direct parents to unique ordered ancestors without path duplication."""
        result: dict[str, tuple[str, ...]] = {}
        position = {node: index for index, node in enumerate(self.topological_order)}
        for node in self.topological_order:
            ancestors = set(parents[node])
            for parent in parents[node]:
                ancestors.update(result[parent])
            result[node] = tuple(sorted(ancestors, key=position.__getitem__))
        return result

    def _validated_parent_override(
        self,
        override: Mapping[str, tuple[str, ...]] | None,
    ) -> dict[str, tuple[str, ...]]:
        if override is None:
            return dict(self.parents)
        if set(override) != set(self.nodes):
            raise ValueError("topology_override must define parents for every compiled quantum.")
        topo_position = {node: index for index, node in enumerate(self.topological_order)}
        result = {}
        for node in self.nodes:
            parents = tuple(override[node])
            if len(parents) != len(self.parents[node]):
                raise ValueError("topology-shuffled controls must preserve every node's indegree.")
            if any(parent not in topo_position or topo_position[parent] >= topo_position[node] for parent in parents):
                raise ValueError("topology_override must remain acyclic in the compiled topological order.")
            result[node] = parents
        return result

    def _validate_oracle_targets(
        self,
        targets: torch.Tensor | None,
        batch: int,
        query_length: int,
    ) -> None:
        if targets is None:
            raise ValueError("oracle routing requires compiler activity_targets.")
        if tuple(targets.shape) != (batch, query_length, len(self.nodes)):
            raise ValueError("activity_targets must have shape [batch, query_length, quantum].")
        if bool(((targets != 0) & (targets != 1)).any()):
            raise ValueError("activity_targets must be binary.")
        for node in self.nodes:
            child = targets[..., self.node_to_index[node]]
            for parent in self.parents[node]:
                parent_target = targets[..., self.node_to_index[parent]]
                if bool((child > parent_target).any()):
                    raise ValueError(f"activity_targets are not downward closed at {parent!r}->{node!r}.")

    def _validate_parent_signatures(self) -> None:
        for trace in self.compiled.validation_traces:
            active = set(trace.active_nodes)
            for node in active:
                if any(parent not in active for parent in self.parents[node]):
                    raise ValueError(
                        f"compiled trace {trace.event_id!r} violates conjunctive parent signature for {node!r}."
                    )

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.view(*value.shape[:-1], self.read_heads, self.head_dim).transpose(-3, -2)

    def _stack(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([values[node] for node in self.nodes], dim=-2)

    def _stack_scalars(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([values[node] for node in self.nodes], dim=-1)


class QComputer(nn.Module):
    """Minimal causal token memory plus one compiled Q-core."""

    def __init__(
        self,
        *,
        compiled: CompiledQProgram,
        vocab_size: int,
        max_seq_len: int,
        d_source: int,
        d_quantum: int,
        pad_id: int,
        sep_id: int,
        read_heads: int = 1,
        mlp_ratio: float = 1.0,
        activation: str = "gelu",
        add_initial_state: bool = True,
        all_quanta_output: bool = True,
    ) -> None:
        super().__init__()
        self.pad_id = int(pad_id)
        self.sep_id = int(sep_id)
        self.max_seq_len = int(max_seq_len)
        self.token_embedding = nn.Embedding(int(vocab_size), int(d_source), padding_idx=self.pad_id)
        self.position_embedding = nn.Embedding(self.max_seq_len, int(d_source))
        self.segment_embedding = nn.Embedding(2, int(d_source))
        self.core = CompiledQCore(
            compiled=compiled,
            d_source=int(d_source),
            d_quantum=int(d_quantum),
            vocab_size=int(vocab_size),
            read_heads=int(read_heads),
            mlp_ratio=float(mlp_ratio),
            activation=activation,
            add_initial_state=add_initial_state,
            all_quanta_output=all_quanta_output,
        )

    @property
    def compiled(self) -> CompiledQProgram:
        return self.core.compiled

    def token_memory(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length = input_ids.shape
        if length > self.max_seq_len:
            raise ValueError(f"sequence length {length} exceeds max_seq_len={self.max_seq_len}.")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        separator = (input_ids == self.sep_id).to(torch.int64).argmax(dim=1)
        segments = (positions > separator.unsqueeze(1)).to(torch.long) * attention_mask.to(torch.long)
        memory = self.token_embedding(input_ids) + self.position_embedding(positions) + self.segment_embedding(segments)
        return memory, segments

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        routing: RoutingMode,
        activity_targets: torch.Tensor | None = None,
        query_positions: torch.Tensor | None = None,
        parent_intervention: ParentIntervention | None = None,
        topology_override: Mapping[str, tuple[str, ...]] | None = None,
    ) -> QCoreTrace:
        memory, segments = self.token_memory(input_ids, attention_mask)
        if query_positions is None:
            query_positions = torch.arange(input_ids.shape[1] - 1, device=input_ids.device).unsqueeze(0).expand(
                input_ids.shape[0], -1
            )
        return self.core(
            memory=memory,
            memory_mask=attention_mask,
            segment_ids=segments,
            query_positions=query_positions,
            routing=routing,
            activity_targets=activity_targets,
            parent_intervention=parent_intervention,
            topology_override=topology_override,
        )

    def parameter_audit(self) -> ParameterAudit:
        return self.core.parameter_audit(
            extra_shared={
                "token_embedding": self.token_embedding,
                "position_embedding": self.position_embedding,
                "segment_embedding": self.segment_embedding,
            }
        )


def balanced_activity_loss(
    gate_logits: torch.Tensor,
    activity_targets: torch.Tensor,
    eligibility_mask: torch.Tensor,
    *,
    nodes: tuple[str, ...],
    warn: bool = True,
) -> ActivityLossResult:
    if gate_logits.shape != activity_targets.shape or gate_logits.shape != eligibility_mask.shape:
        raise ValueError("gate logits, activity targets, and eligibility mask must have identical shapes.")
    per_node = {}
    single_class = []
    losses = []
    for index, node in enumerate(nodes):
        eligible = eligibility_mask[..., index].to(torch.bool)
        if not bool(eligible.any()):
            continue
        logits = gate_logits[..., index][eligible]
        targets = activity_targets[..., index][eligible].to(logits.dtype)
        positive = targets > 0.5
        negative = ~positive
        if bool(positive.any()) and bool(negative.any()):
            positive_loss = F.binary_cross_entropy_with_logits(logits[positive], targets[positive])
            negative_loss = F.binary_cross_entropy_with_logits(logits[negative], targets[negative])
            node_loss = 0.5 * (positive_loss + negative_loss)
        else:
            node_loss = F.binary_cross_entropy_with_logits(logits, targets)
            single_class.append(node)
        per_node[node] = node_loss
        losses.append(node_loss)
    if single_class and warn:
        warnings.warn(
            "eligible activity supervision has one class for: " + ", ".join(single_class),
            stacklevel=2,
        )
    loss = torch.stack(losses).mean() if losses else gate_logits.sum() * 0.0
    return ActivityLossResult(
        loss=loss,
        observed_nodes=len(losses),
        single_class_nodes=tuple(single_class),
        per_node=per_node,
    )


def semantic_classification_loss(
    deltas: torch.Tensor,
    semantic_targets: torch.Tensor,
    classifiers: SemanticClassifierSuite,
) -> SemanticLossResult:
    if deltas.ndim != 4 or deltas.shape[-2] != len(classifiers.nodes):
        raise ValueError("deltas must have shape [batch, query_length, quantum, d_quantum].")
    if semantic_targets.shape != deltas.shape[:-1]:
        raise ValueError("semantic_targets must match the first three delta dimensions.")
    per_node = {}
    losses = []
    supervised_examples = 0
    for node in classifiers.supervised_nodes:
        node_index = classifiers.node_to_index[node]
        targets = semantic_targets[..., node_index]
        active = targets != -100
        if not bool(active.any()):
            continue
        logits = classifiers.logits(node, deltas)
        node_loss = F.cross_entropy(logits[active], targets[active])
        per_node[node] = node_loss
        losses.append(node_loss)
        supervised_examples += int(active.sum().item())
    loss = torch.stack(losses).mean() if losses else deltas.sum() * 0.0
    return SemanticLossResult(
        loss=loss,
        observed_nodes=len(losses),
        supervised_examples=supervised_examples,
        per_node=per_node,
    )


def qcore_training_loss(
    model: QComputer,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    activity_targets: torch.Tensor,
    eligibility_mask: torch.Tensor,
    activity_weight: float,
    semantic_weight: float,
    semantic_classifiers: SemanticClassifierSuite | None = None,
    semantic_targets: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | QCoreTrace | ActivityLossResult | SemanticLossResult]:
    trace = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        routing="predicted",
    )
    shift_labels = labels[:, 1:]
    supervised = shift_labels != -100
    ce = F.cross_entropy(trace.logits[supervised], shift_labels[supervised])
    activity = balanced_activity_loss(
        trace.gate_logits,
        activity_targets,
        eligibility_mask,
        nodes=model.core.nodes,
        # Exact distribution-level one-class warnings come from the compiler;
        # a sampled minibatch is not a valid coverage audit.
        warn=False,
    )
    if semantic_classifiers is None:
        if semantic_targets is not None:
            raise ValueError("semantic_targets require semantic_classifiers.")
        semantic = SemanticLossResult(
            loss=trace.deltas.sum() * 0.0,
            observed_nodes=0,
            supervised_examples=0,
            per_node={},
        )
    else:
        if semantic_targets is None:
            raise ValueError("semantic_classifiers require semantic_targets.")
        semantic = semantic_classification_loss(trace.deltas, semantic_targets, semantic_classifiers)
    return {
        "loss": ce + float(activity_weight) * activity.loss + float(semantic_weight) * semantic.loss,
        "ce_loss": ce,
        "activity_loss": activity.loss,
        "semantic_loss": semantic.loss,
        "trace": trace,
        "activity_result": activity,
        "semantic_result": semantic,
    }


def topology_shuffled_control(compiled: CompiledQProgram, *, seed: int) -> dict[str, tuple[str, ...]]:
    """Construct a deterministic, indegree-matched incorrect acyclic graph."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = tuple(compiled.structure.topological_order)
    positions = {node: index for index, node in enumerate(order)}
    result = {}
    for node in order:
        indegree = len(compiled.structure.parents[node])
        candidates = list(order[: positions[node]])
        if indegree == 0:
            result[node] = ()
            continue
        if len(candidates) < indegree:
            raise ValueError(f"cannot preserve indegree for topology-shuffled node {node!r}.")
        permutation = torch.randperm(len(candidates), generator=generator).tolist()
        result[node] = tuple(candidates[index] for index in permutation[:indegree])
    if all(result[node] == compiled.structure.parents[node] for node in order):
        for node in order:
            if len(result[node]) == 1 and len(order[: positions[node]]) > 1:
                candidates = [item for item in order[: positions[node]] if item != result[node][0]]
                result[node] = (candidates[0],)
                break
    return result


def parent_necessity_audit(
    model: QComputer,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    activity_targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compare oracle logits under parent zeroing, shuffling, and ablation."""
    with torch.no_grad():
        baseline = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            routing="oracle",
            activity_targets=activity_targets,
        ).logits
        results = {"baseline_logits": baseline}
        for kind in ("zero", "shuffle"):
            changed = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                routing="oracle",
                activity_targets=activity_targets,
                parent_intervention=ParentIntervention(kind=kind),
            ).logits
            results[f"{kind}_logit_delta"] = changed - baseline
        for node in model.core.nodes:
            for parent in model.core.parents[node]:
                changed = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    routing="oracle",
                    activity_targets=activity_targets,
                    parent_intervention=ParentIntervention(kind="ablate", node=node, parent=parent),
                ).logits
                results[f"ablate/{parent}->{node}"] = changed - baseline
        shuffled_topology = topology_shuffled_control(model.compiled, seed=0)
        changed = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            routing="oracle",
            activity_targets=activity_targets,
            topology_override=shuffled_topology,
        ).logits
        results["topology_shuffled_logit_delta"] = changed - baseline
    return results
