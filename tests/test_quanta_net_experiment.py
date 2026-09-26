from __future__ import annotations

from dataclasses import replace
import json
import logging
from types import SimpleNamespace

import pytest
import torch

from quanta.config import QuantaNetConfig, QuantaSteeringConfig, load_experiment_config
from quanta.experiments.quanta_net.experiment import (
    _EpochExampleSampler,
    QuantaSteeringExperiment,
    _alignment_config_payload,
    _batch_supervision,
    _default_q_save_dir,
    _log_q_evaluation,
    _log_alignment_evaluation,
    _load_teacher_run,
    _write_run_status,
    _q_greedy_decode,
    _save_named_checkpoint,
    _scheduled_full_evaluation_steps,
    _validate_q_teacher_metadata,
)


class _Tokenizer:
    pad_id = 0
    bos_id = 1
    sep_id = 2
    eos_id = 3
    token_to_id = {
        "<D0>": 10,
        "<D1>": 11,
        "<D2>": 12,
        "one": 4,
    }
    id_to_token = {0: "[PAD]", 1: "[BOS]", 2: "[SEP]", 3: "[EOS]", 4: "one"}

    def decode_words(self, token_ids: list[int]) -> list[str]:
        words = []
        for token_id in token_ids:
            if int(token_id) == self.eos_id:
                break
            if int(token_id) == 4:
                words.append("one")
        return words


def test_epoch_example_sampler_covers_each_example_once_per_epoch() -> None:
    left = _EpochExampleSampler(range(7), seed=4)
    right = _EpochExampleSampler(range(7), seed=4)

    first = left.take(7)
    second = left.take(7)
    assert sorted(first) == list(range(7))
    assert sorted(second) == list(range(7))
    assert first == right.take(7)
    assert second == right.take(7)

    crossed = _EpochExampleSampler(range(5), seed=2).take(7)
    assert sorted(crossed[:5]) == list(range(5))
    assert len(crossed) == 7


def test_saved_alignment_config_omits_inherited_q_architecture_fields() -> None:
    payload = _alignment_config_payload(QuantaSteeringConfig(q_checkpoint="q/model.pt"))

    assert payload["q_checkpoint"] == "q/model.pt"
    assert payload["transformer_layers"] == 7
    assert "compiled_program_path" not in payload
    assert "d_source" not in payload
    assert "d_quantum" not in payload
    assert "read_heads" not in payload


def test_alignment_teacher_run_derives_q_architecture_and_artifact_from_saved_config(
    tmp_path, monkeypatch
) -> None:
    run_dir = tmp_path / "q_run"
    run_dir.mkdir()
    (run_dir / "model.pt").touch()
    (run_dir / "config.json").write_text(
        '{"compiled_program_path": "/artifacts/q", "d_source": 24, '
        '"d_quantum": 16, "read_heads": 2, "activation": "relu"}'
    )
    compiled = object()
    monkeypatch.setattr(
        "quanta.experiments.quanta_net.experiment.CompiledQProgram.read",
        lambda path: compiled,
    )

    checkpoint, observed_run_dir, teacher_config, observed_compiled = _load_teacher_run(
        str(run_dir)
    )

    assert checkpoint == str(run_dir / "model.pt")
    assert observed_run_dir == str(run_dir)
    assert teacher_config.compiled_program_path == "/artifacts/q"
    assert (teacher_config.d_source, teacher_config.d_quantum, teacher_config.read_heads) == (
        24,
        16,
        2,
    )
    assert teacher_config.activation == "relu"
    assert observed_compiled is compiled


def test_alignment_teacher_rejects_precompiler_q_run(tmp_path) -> None:
    run_dir = tmp_path / "legacy_q_run"
    run_dir.mkdir()
    (run_dir / "model.pt").touch()
    (run_dir / "config.json").write_text('{"d_quantum": 32}')

    with pytest.raises(ValueError, match="predates compiled Q-program checkpoints"):
        _load_teacher_run(str(run_dir))


class _GreedyTask:
    tokenizer = _Tokenizer()
    config = SimpleNamespace(max_seq_len=8)

    @staticmethod
    def input_digit_string(number: int) -> str:
        return str(int(number))


class _BatchedGreedyModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def eval(self):
        return self

    def __call__(self, *, input_ids, attention_mask, query_positions, routing, activity_targets):
        del query_positions, routing, activity_targets
        self.batch_sizes.append(int(input_ids.shape[0]))
        logits = torch.full((input_ids.shape[0], 1, 13), -100.0, device=input_ids.device)
        for row in range(input_ids.shape[0]):
            separator = int(torch.nonzero(input_ids[row] == _Tokenizer.sep_id, as_tuple=False)[0])
            visible_length = int(attention_mask[row].sum())
            generated_length = visible_length - separator - 1
            token = 4 if generated_length == 0 else _Tokenizer.eos_id
            logits[row, 0, token] = 100.0
        return SimpleNamespace(logits=logits)


def test_q_greedy_decode_batches_active_examples_with_mixed_prompt_lengths() -> None:
    model = _BatchedGreedyModel()

    decoded = _q_greedy_decode(
        model,
        _GreedyTask(),
        [1, 12, 2],
        supervision=None,
        routing="predicted",
        device=torch.device("cpu"),
    )

    assert decoded == [(["one"], True), (["one"], True), (["one"], True)]
    assert model.batch_sizes == [3, 3]


class _SupervisionIndex:
    def __init__(self, target: str) -> None:
        self.target = target
        self.states = []

    def lookup(self, state):
        self.states.append(state)
        return SimpleNamespace(
            activity=(1,),
            eligibility=(1,),
            semantics=("semantic",),
            target=self.target,
        )


def test_batch_supervision_checks_compiler_target_at_shifted_label_position() -> None:
    batch = SimpleNamespace(
        labels=torch.tensor([[-100, -100, -100, 4]]),
        numbers=[1],
        texts=["one"],
    )
    compiled = SimpleNamespace(nodes=("EMIT",))
    matching = _SupervisionIndex("one")

    activity, eligibility, semantics = _batch_supervision(
        batch,
        matching,
        compiled,
        tokenizer=_Tokenizer(),
        device=torch.device("cpu"),
    )

    assert activity[0, 2].tolist() == [1.0]
    assert eligibility[0, 2].tolist() == [1.0]
    assert semantics[0][2] == ("semantic",)
    assert matching.states[0].digits == (1,)
    assert matching.states[0].prefix == ()
    assert matching.states[0].position == 0

    with pytest.raises(ValueError, match="compiled Q-program target disagrees"):
        _batch_supervision(
            batch,
            _SupervisionIndex("two"),
            compiled,
            tokenizer=_Tokenizer(),
            device=torch.device("cpu"),
        )


def test_full_evaluation_defaults_to_final_and_accepts_an_interval() -> None:
    assert _scheduled_full_evaluation_steps(100, None) == {100}
    assert _scheduled_full_evaluation_steps(100, 30) == {30, 60, 90, 100}


def test_q_training_log_contains_losses_and_deployment_accuracy_only(caplog) -> None:
    record = {
        "step": 20,
        "full_evaluation": True,
        "train_loss": 1.2,
        "train_task": 0.4,
        "train_gate": 0.3,
        "train_message": 0.2,
        "oracle_token_accuracy": 0.99,
        "oracle_exact_sequence_accuracy": 0.98,
        "predicted_token_accuracy": 0.91,
        "predicted_exact_sequence_accuracy": 0.82,
        "predicted_soft_trace_mae": 0.1,
    }

    with caplog.at_level(logging.INFO):
        _log_q_evaluation(
            record,
            best_token_accuracy=0.93,
            best_sequence_accuracy=0.87,
        )

    message = caplog.records[-1].message
    assert message == (
        "experiment=quanta_net event=evaluation step=20 "
        "token_accuracy=0.91 best_token_accuracy=0.93 "
        "sequence_accuracy=0.82 best_sequence_accuracy=0.87 "
        "train_loss=1.2 train_task=0.4 train_gate=0.3 train_message=0.2"
    )


def test_alignment_training_log_reports_progress_losses_and_accuracy(caplog) -> None:
    record = {
        "step": 20,
        "elapsed_seconds": 12.5,
        "evaluation_seconds": 3.5,
        "lr": 0.009,
        "total_loss": 1.4,
        "task_loss": 0.4,
        "alignment_loss": 1.0,
        "cumulative_state_mse": 2.0,
        "transformer_token_accuracy": 0.91,
        "transformer_exact_sequence_accuracy": 0.82,
        "transformer_exact_accuracy_without_eos": 0.84,
        "transformer_normalized_edit_distance": 0.07,
    }

    with caplog.at_level(logging.INFO):
        _log_alignment_evaluation(
            record,
            total_steps=100,
            best_sequence_accuracy=0.85,
        )

    assert caplog.records[-1].message == (
        "experiment=quanta_steering event=evaluation step=20 total_steps=100 "
        "elapsed_seconds=12.5 evaluation_seconds=3.5 learning_rate=0.009 "
        "train_loss=1.4 task_loss=0.4 alignment_loss=1 cumulative_state_mse=2 "
        "token_accuracy=0.91 sequence_accuracy=0.82 best_sequence_accuracy=0.85 "
        "normalized_edit_distance=0.07"
    )


def test_alignment_run_status_is_json_and_atomically_replaced(tmp_path) -> None:
    _write_run_status(
        str(tmp_path),
        status="evaluating",
        step=20,
        total_steps=100,
        output_dir=str(tmp_path),
    )

    payload = json.loads((tmp_path / "status.json").read_text())

    assert payload["status"] == "evaluating"
    assert payload["step"] == 20
    assert payload["output_dir"] == str(tmp_path)
    assert "updated_at_unix" in payload
    assert not (tmp_path / "status.json.tmp").exists()


def test_alignment_failure_is_recorded_before_reraising(tmp_path) -> None:
    experiment = object.__new__(QuantaSteeringExperiment)
    experiment.config = SimpleNamespace(save_dir=str(tmp_path))

    def fail() -> str:
        raise RuntimeError("teacher mismatch")

    experiment._run_alignment = fail

    with pytest.raises(RuntimeError, match="teacher mismatch"):
        experiment.run()

    payload = json.loads((tmp_path / "status.json").read_text())
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "teacher mismatch"


def test_named_q_checkpoint_persists_model_optimizer_and_training_only_semantic_state(tmp_path) -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

    model = Model()
    semantic_classifiers = torch.nn.Linear(1, 2)
    optimizer = torch.optim.Adam(model.parameters())
    _save_named_checkpoint(
        model,
        optimizer,
        str(tmp_path),
        "best",
        semantic_classifiers=semantic_classifiers,
    )

    state = torch.load(tmp_path / "best_model.pt", weights_only=True)
    optimizer_state = torch.load(tmp_path / "best_optimizer.pt", weights_only=True)
    semantic_state = torch.load(tmp_path / "best_semantic_classifiers.pt", weights_only=True)
    assert state["weight"].item() == 1.0
    assert "param_groups" in optimizer_state
    assert set(semantic_state) == {"weight", "bias"}


def test_steering_teacher_metadata_includes_activation_and_readout_contract() -> None:
    compiled = SimpleNamespace(
        metadata=SimpleNamespace(program_fingerprint="program"),
    )
    model = SimpleNamespace(
        core=SimpleNamespace(
            nodes=("A", "B"),
            depth=2,
            d_source=4,
            d_quantum=4,
            activation="layernorm_gelu",
            add_initial_state=False,
            all_quanta_output=False,
        )
    )
    metadata = {
        "compiled_program_fingerprint": "program",
        "nodes": ["A", "B"],
        "depth": 2,
        "d_source": 4,
        "d_quantum": 4,
        "activation": "layernorm_gelu",
        "add_initial_state": False,
        "all_quanta_output": False,
    }

    _validate_q_teacher_metadata(metadata, compiled, model)
    with pytest.raises(ValueError, match="add_initial_state"):
        _validate_q_teacher_metadata({**metadata, "add_initial_state": True}, compiled, model)
    with pytest.raises(ValueError, match="all_quanta_output"):
        _validate_q_teacher_metadata({**metadata, "all_quanta_output": True}, compiled, model)


def test_default_q_save_dir_changes_with_config_and_compiler_fingerprints() -> None:
    metadata = SimpleNamespace(
        compiler_version="1",
        program_fingerprint="a" * 64,
        primitive_cost_fingerprint="p" * 64,
        validation_domain_fingerprint="v" * 64,
        training_distribution_fingerprint="b" * 64,
        evaluation_distribution_fingerprint="c" * 64,
    )
    compiled = SimpleNamespace(id="number_naming", metadata=metadata)
    config = QuantaNetConfig(compiled_program_path=".artifacts/program-a", steps=100, epochs=None)

    first = _default_q_save_dir("quanta_net", config, compiled)
    assert first == _default_q_save_dir("quanta_net", config, compiled)
    assert first != _default_q_save_dir("quanta_net", replace(config, d_quantum=64), compiled)
    assert first != _default_q_save_dir("quanta_net", replace(config, add_initial_state=False), compiled)
    assert first != _default_q_save_dir("quanta_net", replace(config, all_quanta_output=False), compiled)
    assert first != _default_q_save_dir("quanta_net", replace(config, activation="layernorm_gelu"), compiled)

    changed_metadata = SimpleNamespace(
        compiler_version="1",
        program_fingerprint="d" * 64,
        primitive_cost_fingerprint="p" * 64,
        validation_domain_fingerprint="v" * 64,
        training_distribution_fingerprint="b" * 64,
        evaluation_distribution_fingerprint="c" * 64,
    )
    changed_compiled = SimpleNamespace(id="number_naming", metadata=changed_metadata)
    assert first != _default_q_save_dir("quanta_net", config, changed_compiled)


def test_q_config_validates_full_eval_and_parent_audit_fields(tmp_path) -> None:
    path = tmp_path / "q.yaml"
    path.write_text(
        "model:\n"
        "  compiled_program_path: /tmp/program.pkl\n"
        "training:\n"
        "  steps: 10\n"
        "  full_eval_steps: 5\n"
        "  parent_audit_size: 32\n"
    )
    config = load_experiment_config("quanta_net", path)
    assert config.full_eval_steps == 5
    assert config.parent_audit_size == 32

    path.write_text(
        "model:\n"
        "  compiled_program_path: /tmp/program.pkl\n"
        "training:\n"
        "  steps: 10\n"
        "  full_eval_steps: 0\n"
    )
    with pytest.raises(ValueError, match="full_eval_steps"):
        load_experiment_config("quanta_net", path)
