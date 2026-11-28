"""Pluggable Team 9 minibench_v1 (#929) hook for rl-loop/train.py.

Calls teams/09-evaluation/harness_lab/minibench_lab.py when present on disk,
or a caller-supplied runner (for unit tests). Separates cleared metrics from
diagnostic/awaiting-approval/internal-only metrics under `minibench/diagnostic/`.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
MINIBENCH_LAB_PATH = REPO_ROOT / "teams" / "09-evaluation" / "harness_lab" / "minibench_lab.py"

VALID_PROFILES = ("quick", "full")
DIAGNOSTIC_TASKS = frozenset({
    "arc_easy_cloze",
    "arc_challenge_cloze",
    "hellaswag_cloze",
    "humaneval_gold",
    "mbpp_gold",
})


def _load_minibench_lab():
    if not MINIBENCH_LAB_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("team9_minibench_lab", MINIBENCH_LAB_PATH)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def format_minibench_metrics(raw: dict[str, Any], *, profile: str, step: int) -> dict[str, Any]:
    """Flatten raw minibench_v1 results into Team 8 / Team 10 metric keys."""
    flat: dict[str, Any] = {"minibench/profile": profile, "minibench/step": step}
    for key, value in (raw.get("heldout_bpb") or {}).items():
        if isinstance(value, (int, float)):
            flat[f"minibench/heldout_bpb/{key}"] = round(float(value), 6)
    for task_name, task_res in (raw.get("tasks") or {}).items():
        prefix = "minibench/diagnostic" if task_name in DIAGNOSTIC_TASKS else "minibench"
        if isinstance(task_res, (int, float)):
            flat[f"{prefix}/{task_name}"] = round(float(task_res), 6)
        elif isinstance(task_res, dict):
            for metric_name in ("acc_norm", "acc", "gold_bpb", "score"):
                if isinstance(task_res.get(metric_name), (int, float)):
                    flat[f"{prefix}/{task_name}/{metric_name}"] = round(float(task_res[metric_name]), 6)
    return flat


def hydrate_minibench_steps(past_evals: list[dict[str, Any]]) -> set[int]:
    """Collect already-measured minibench steps from evals.jsonl on resume."""
    return {int(e["step"]) for e in past_evals if e.get("name") == "minibench" and "step" in e}


class _PolicyLM:
    """Minimal lm-eval-style loglikelihood adapter over an rl-loop Policy."""

    def __init__(self, policy: Any):
        self.policy = policy
        self.art = getattr(policy, "tokenizer", None)

    def _score_pair(self, prompt_ids: list[int], cont_ids: list[int]) -> tuple[float, bool]:
        import torch

        if not cont_ids:
            return 0.0, False
        model = getattr(self.policy, "model", self.policy)
        device = next(model.parameters()).device
        full_ids = prompt_ids + cont_ids
        inp = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(inp)
            logits = out.logits if hasattr(out, "logits") else out
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            start = max(len(prompt_ids) - 1, 0)
            target = inp[:, 1 + start :]
            token_lp = log_probs[:, start:, :].gather(-1, target.unsqueeze(-1)).squeeze(-1)
            greedy = log_probs[:, start:, :].argmax(dim=-1) == target
            return float(token_lp.sum().item()), bool(greedy.all().item())

    def loglikelihood(self, requests):
        encode = self.art.encode if hasattr(self.art, "encode") else (lambda s: list(s.encode("utf-8")))
        for req in requests:
            prompt, choice = req.args
            p_ids = encode(prompt)
            c_ids = encode(prompt + choice)[len(p_ids) :]
            yield self._score_pair(p_ids, c_ids)

    def loglikelihood_tokens(self, tokens, allow_truncation: bool = False):
        del allow_truncation
        for p_ids, c_ids in tokens:
            yield self._score_pair(list(p_ids), list(c_ids))


def _run_via_minibench_lab(mod: Any, policy: Any, profile: str) -> dict[str, Any]:
    if hasattr(mod, "score_policy"):
        return mod.score_policy(policy, profile=profile)
    if not (hasattr(mod, "score_items") and hasattr(mod, "load_items") and hasattr(mod, "summarize")):
        raise AttributeError("minibench_lab.py must expose score_policy or (load_items, score_items, summarize)")
    suite_cfg = getattr(mod, "SUITE", None)
    tasks = mod.load_items(suite_cfg, profile)
    lm = _PolicyLM(policy)
    by_task = mod.score_items(lm, tasks)
    summary_tasks = {}
    for name, rows in by_task.items():
        stats = mod.summarize(rows)
        entry: dict[str, Any] = {"gold_bpb": stats["gold_bpb"]}
        if "accuracy_points" in stats:
            entry["acc"] = stats["accuracy_points"] / 100.0
        summary_tasks[name] = entry
    return {"tasks": summary_tasks}


def run_minibench(
    policy: Any,
    *,
    step: int,
    profile: str = "quick",
    on_missing: str = "skip",
    runner: Callable[[Any, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Team 9 minibench_v1 if available (or via mock runner) and return structured metrics."""
    if profile not in VALID_PROFILES:
        raise ValueError(f"Unsupported minibench profile {profile!r}; expected one of {VALID_PROFILES}")
    if on_missing not in ("skip", "raise"):
        raise ValueError(f"Unsupported on_missing {on_missing!r}; expected 'skip' or 'raise'")

    if runner is not None:
        raw = runner(policy, profile)
    else:
        mod = _load_minibench_lab()
        if mod is None:
            if on_missing == "raise":
                raise FileNotFoundError(f"Team 9 minibench_lab.py not found at {MINIBENCH_LAB_PATH} (awaiting #929)")
            return {
                "status": "skipped_missing_pr929",
                "profile": profile,
                "step": step,
                "metrics": {},
            }
        raw = _run_via_minibench_lab(mod, policy, profile)

    metrics = format_minibench_metrics(raw or {}, profile=profile, step=step)
    return {
        "status": "ok",
        "profile": profile,
        "step": step,
        "metrics": metrics,
    }
