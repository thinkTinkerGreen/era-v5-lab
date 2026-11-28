"""Unit tests for rl-loop/minibench_hook.py and metrics_feed.py minibench/ forwarding."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RL_LOOP = Path(__file__).resolve().parent
if str(RL_LOOP) not in sys.path:
    sys.path.insert(0, str(RL_LOOP))

import metrics_feed  # noqa: E402
import minibench_hook  # noqa: E402


def test_run_minibench_mock_separates_cleared_and_diagnostic():
    def fake_runner(policy, profile):
        assert profile == "quick"
        return {
            "heldout_bpb": {"en": 1.7687, "hi": 1.8982},
            "tasks": {
                "mmlu_cloze": {"acc_norm": 0.42},
                "milu_hi": {"acc_norm": 0.51},
                "arc_challenge_cloze": {"acc_norm": 0.38},
                "hellaswag_cloze": {"acc_norm": 0.44},
                "humaneval_gold": {"gold_bpb": 2.1},
            },
        }

    res = minibench_hook.run_minibench(object(), step=10, profile="quick", runner=fake_runner)
    assert res["status"] == "ok"
    m = res["metrics"]
    assert m["minibench/profile"] == "quick"
    assert m["minibench/step"] == 10
    assert m["minibench/heldout_bpb/en"] == 1.7687
    assert m["minibench/mmlu_cloze/acc_norm"] == 0.42
    assert m["minibench/milu_hi/acc_norm"] == 0.51
    assert m["minibench/diagnostic/arc_challenge_cloze/acc_norm"] == 0.38
    assert m["minibench/diagnostic/hellaswag_cloze/acc_norm"] == 0.44
    assert m["minibench/diagnostic/humaneval_gold/gold_bpb"] == 2.1


def test_run_minibench_missing_pr929_skip_vs_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(minibench_hook, "MINIBENCH_LAB_PATH", tmp_path / "missing_minibench_lab.py")
    skipped = minibench_hook.run_minibench(object(), step=0, profile="quick", on_missing="skip")
    assert skipped["status"] == "skipped_missing_pr929"
    assert skipped["metrics"] == {}

    with pytest.raises(FileNotFoundError):
        minibench_hook.run_minibench(object(), step=0, profile="full", on_missing="raise")


def test_run_minibench_rejects_bad_profile():
    with pytest.raises(ValueError):
        minibench_hook.run_minibench(object(), step=0, profile="invalid")


def test_metrics_feed_forwards_minibench_keys(tmp_path):
    metrics_p = tmp_path / "metrics.jsonl"
    out_p = tmp_path / "rl_feed.jsonl"
    metrics_p.write_text(
        json.dumps({
            "step": 10,
            "train_reward": 0.5,
            "minibench/heldout_bpb/en": 1.75,
            "minibench/diagnostic/hellaswag_cloze/acc_norm": 0.41,
        }) + "\n",
        encoding="utf-8",
    )
    rows = metrics_feed.convert(metrics_p, out_p)
    assert len(rows) == 1
    assert rows[0]["minibench/heldout_bpb/en"] == 1.75
    assert rows[0]["minibench/diagnostic/hellaswag_cloze/acc_norm"] == 0.41


def test_metrics_feed_merges_minibench_from_evals_jsonl(tmp_path):
    metrics_p = tmp_path / "metrics.jsonl"
    evals_p = tmp_path / "evals.jsonl"
    out_p = tmp_path / "rl_feed.jsonl"
    metrics_p.write_text(json.dumps({"step": 20, "train_reward": 0.6}) + "\n", encoding="utf-8")
    evals_p.write_text(
        json.dumps({
            "name": "minibench",
            "step": 20,
            "profile": "quick",
            "status": "ok",
            "metrics": {"minibench/mmlu_cloze/acc_norm": 0.45},
        }) + "\n",
        encoding="utf-8",
    )
    rows = metrics_feed.convert(metrics_p, out_p)
    assert rows[0]["minibench/mmlu_cloze/acc_norm"] == 0.45


def test_hydrate_minibench_steps():
    past = [
        {"name": "heldout", "step": 0},
        {"name": "minibench", "step": 0},
        {"name": "minibench", "step": 10},
    ]
    assert minibench_hook.hydrate_minibench_steps(past) == {0, 10}


def test_run_via_minibench_lab_score_items_adapter(monkeypatch, tmp_path):
    dummy_path = tmp_path / "minibench_lab.py"
    dummy_path.write_text("# stub\n", encoding="utf-8")
    monkeypatch.setattr(minibench_hook, "MINIBENCH_LAB_PATH", dummy_path)

    class FakeMod:
        SUITE = "minibench_v1.yaml"

        @staticmethod
        def load_items(suite, profile):
            assert suite == "minibench_v1.yaml"
            assert profile == "quick"
            return {"mmlu_cloze": ["item1"], "arc_challenge_cloze": ["item2"]}

        @staticmethod
        def score_items(lm, tasks):
            assert hasattr(lm, "loglikelihood") and hasattr(lm, "loglikelihood_tokens")
            return {k: [{"gold_nll_nats": 1.0, "utf8_bytes": 10}] for k in tasks}

        @staticmethod
        def summarize(rows):
            return {"gold_bpb": 0.14427, "accuracy_points": 50.0}

    monkeypatch.setattr(minibench_hook, "_load_minibench_lab", lambda: FakeMod)
    res = minibench_hook.run_minibench(object(), step=5, profile="quick")
    assert res["status"] == "ok"
    assert res["metrics"]["minibench/mmlu_cloze/acc"] == 0.5
    assert res["metrics"]["minibench/diagnostic/arc_challenge_cloze/acc"] == 0.5
