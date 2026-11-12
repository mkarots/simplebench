"""Score an answer file and write the mean next to the run.

This module does not open a network connection. The caller passes ``complete``.
"""

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from simplebench.judge import Complete, Judgment, score_completion
from simplebench.prompts import Exam, Prompt, PromptError, load_answers, repo_root

JUDGMENTS_NAME = "judgments.jsonl"
METRICS_NAME = "metrics.json"
ANSWERS_NAME = "answers.jsonl"


def score_rows(
    prompts: Sequence[Prompt],
    answers: Mapping[str, str],
    rubric: str,
    complete: Complete,
) -> tuple[Judgment, ...]:
    """Score every prompt. The answer ids must be exactly the prompt ids."""
    expected = {item.id for item in prompts}
    got = set(answers)
    if got != expected:
        missing = sorted(expected - got)[:5]
        extra = sorted(got - expected)[:5]
        raise PromptError(f"answers must match the prompt ids; missing {missing} extra {extra}")
    return tuple(
        score_completion(item.id, item.prompt, answers[item.id], rubric, complete) for item in prompts
    )


def summarize(judgments: Sequence[Judgment], exam: Exam, model: str, checkpoint: str) -> dict:
    """The decision number is ``mean_score``. ``model`` wrote the answers. ``judge`` scored them."""
    counts = {str(score): 0 for score in (0, 1, 2)}
    for row in judgments:
        counts[str(row.score)] += 1
    total = len(judgments)
    return {
        "exam": exam.id,
        "exam_sha256": exam.sha256,
        "model": model,
        "checkpoint": checkpoint,
        "judge": {"provider": exam.judge.provider, "model": exam.judge.model},
        "rows": total,
        "mean_score": (sum(row.score for row in judgments) / total) if total else 0.0,
        "counts": counts,
        "pass_rate": (counts["2"] / total) if total else 0.0,
        "judged": sum(1 for row in judgments if not row.empty),
        "empty": sum(1 for row in judgments if row.empty),
    }


def run_dir(model: str, checkpoint: str, root: Path | None = None) -> Path:
    """``runs/<model>/<checkpoint>`` for the model that wrote the answers."""
    base = repo_root() if root is None else root
    return base / "runs" / _path_segment(model, "model") / _path_segment(checkpoint, "checkpoint")


def _path_segment(name: str, label: str) -> str:
    if name in {"", ".", ".."} or "/" in name or "\\" in name or name != name.strip():
        raise PromptError(f"{label} must be one path segment, got {name!r}")
    return name


def write_run(
    run: Path,
    judgments: Sequence[Judgment],
    exam: Exam,
    model: str,
    checkpoint: str,
    answers_path: Path,
) -> dict:
    """Write the score and a copy of the answers into ``run``. Returns the metrics object."""
    run.mkdir(parents=True, exist_ok=True)
    _keep_answers(answers_path, run)
    lines = [
        json.dumps(
            {
                "id": row.id,
                "score": row.score,
                "reason": row.reason,
                "empty": row.empty,
            },
            ensure_ascii=False,
        )
        for row in judgments
    ]
    (run / JUDGMENTS_NAME).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    metrics = summarize(judgments, exam, model, checkpoint)
    _write_json(run / METRICS_NAME, metrics)
    return metrics


def judge_answers(
    answers_path: Path,
    run: Path,
    complete: Complete,
    exam: Exam,
    model: str,
    checkpoint: str,
) -> dict:
    """Score the answer file with this exam and write the run."""
    judgments = score_rows(exam.prompts, load_answers(answers_path), exam.rubric, complete)
    return write_run(run, judgments, exam, model, checkpoint, answers_path)


def compare_runs(left: Path, right: Path) -> dict:
    """Compare two checkpoints of one exam. The ids are the prompts whose score moved."""
    left_metrics = _read_json(left / METRICS_NAME)
    right_metrics = _read_json(right / METRICS_NAME)
    left_exam = (left_metrics.get("exam"), left_metrics.get("exam_sha256"))
    right_exam = (right_metrics.get("exam"), right_metrics.get("exam_sha256"))
    if left_exam != right_exam:
        raise PromptError(f"exams differ: {left_exam[0]} {left_exam[1]} and {right_exam[0]} {right_exam[1]}")
    left_scores = _scores(left)
    right_scores = _scores(right)
    if set(left_scores) != set(right_scores):
        raise PromptError("judgment ids differ")
    changed = tuple(
        (prompt_id, left_scores[prompt_id], right_scores[prompt_id])
        for prompt_id in sorted(left_scores)
        if left_scores[prompt_id] != right_scores[prompt_id]
    )
    return {
        "exam": left_metrics["exam"],
        "left_model": left_metrics.get("model", ""),
        "right_model": right_metrics.get("model", ""),
        "left_checkpoint": left_metrics.get("checkpoint", left.name),
        "right_checkpoint": right_metrics.get("checkpoint", right.name),
        "left_mean": left_metrics["mean_score"],
        "right_mean": right_metrics["mean_score"],
        "changed": changed,
    }


def _keep_answers(answers_path: Path, run: Path) -> None:
    target = run / ANSWERS_NAME
    if not answers_path.is_file():
        raise PromptError(f"missing answers: {answers_path}")
    if answers_path.resolve() == target.resolve():
        return
    shutil.copyfile(answers_path, target)


def _scores(run: Path) -> dict[str, int]:
    path = run / JUDGMENTS_NAME
    if not path.is_file():
        raise PromptError(f"missing judgments: {path}")
    scores: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scores[row["id"]] = row["score"]
    return scores


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise PromptError(f"missing metrics: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PromptError(f"{path}: metrics must be an object")
    return data


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
