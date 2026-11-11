"""Score an answer file and write the mean next to the run.

This module does not open a network connection. The caller passes ``complete``.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from simple_completions.judge import Complete, Judgment, score_completion
from simple_completions.prompts import DATASET_ID, PromptError, Stem, load_answers, load_rubric, load_stems

JUDGMENTS_NAME = "judgments.jsonl"
METRICS_NAME = "metrics.json"


def score_rows(
    stems: Sequence[Stem],
    answers: Mapping[str, str],
    rubric: str,
    complete: Complete,
) -> tuple[Judgment, ...]:
    """Score every stem. The answer ids must be exactly the stem ids."""
    expected = {stem.id for stem in stems}
    got = set(answers)
    if got != expected:
        missing = sorted(expected - got)[:5]
        extra = sorted(got - expected)[:5]
        raise PromptError(f"answers must match the stem ids; missing {missing} extra {extra}")
    return tuple(
        score_completion(stem.id, stem.prompt, answers[stem.id], rubric, complete) for stem in stems
    )


def summarize(judgments: Sequence[Judgment], provider: str, model: str) -> dict:
    """The decision number is ``mean_score``. ``pass_rate`` is the share of 2s."""
    counts = {str(score): 0 for score in (0, 1, 2)}
    for row in judgments:
        counts[str(row.score)] += 1
    total = len(judgments)
    return {
        "dataset": DATASET_ID,
        "provider": provider,
        "model": model,
        "rows": total,
        "mean_score": (sum(row.score for row in judgments) / total) if total else 0.0,
        "counts": counts,
        "pass_rate": (counts["2"] / total) if total else 0.0,
        "judged": sum(1 for row in judgments if not row.empty),
        "empty": sum(1 for row in judgments if row.empty),
    }


def write_run(
    run: Path,
    judgments: Sequence[Judgment],
    provider: str,
    model: str,
) -> dict:
    """Write judgments and metrics beside the run. Returns the metrics object."""
    run.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "id": row.id,
                "score": row.score,
                "reason": row.reason,
                "truncated": row.truncated,
                "empty": row.empty,
            },
            ensure_ascii=False,
        )
        for row in judgments
    ]
    (run / JUDGMENTS_NAME).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    metrics = summarize(judgments, provider, model)
    _write_json(run / METRICS_NAME, metrics)
    return metrics


def judge_answers(
    answers_path: Path,
    run: Path,
    complete: Complete,
    provider: str,
    model: str,
) -> dict:
    """Load the frozen stems, score the answer file, and write the run."""
    judgments = score_rows(load_stems(), load_answers(answers_path), load_rubric(), complete)
    return write_run(run, judgments, provider, model)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
