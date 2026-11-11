"""Score an answer file and write the mean next to the run.

This module does not open a network connection. The caller passes ``complete``.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from simplebench.judge import Complete, Judgment, score_completion
from simplebench.prompts import DATASET_ID, Prompt, PromptError, exam_stamp, load_answers, load_prompts, load_rubric

JUDGMENTS_NAME = "judgments.jsonl"
METRICS_NAME = "metrics.json"
METADATA_NAME = "metadata.json"
JUDGES_DIR = "judges"


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


def summarize(judgments: Sequence[Judgment], provider: str, model: str) -> dict:
    """The decision number is ``mean_score``. ``pass_rate`` is the share of 2s."""
    counts = {str(score): 0 for score in (0, 1, 2)}
    for row in judgments:
        counts[str(row.score)] += 1
    total = len(judgments)
    stamp = exam_stamp()
    return {
        "dataset": DATASET_ID,
        "exam": stamp["exam"],
        "prompts_sha256": stamp["prompts_sha256"],
        "rubric_sha256": stamp["rubric_sha256"],
        "provider": provider,
        "model": model,
        "rows": total,
        "mean_score": (sum(row.score for row in judgments) / total) if total else 0.0,
        "counts": counts,
        "pass_rate": (counts["2"] / total) if total else 0.0,
        "judged": sum(1 for row in judgments if not row.empty),
        "empty": sum(1 for row in judgments if row.empty),
    }


def judge_dir(run: Path, model: str) -> Path:
    """One folder per judge, under the run that holds the answers."""
    return run / JUDGES_DIR / model


def write_metadata(run: Path, provider: str, model: str, answers: str = "answers.jsonl") -> dict:
    """Name the model that wrote the answers. This is not the judge."""
    payload = {
        "dataset": DATASET_ID,
        "model": model,
        "provider": provider,
        "answers": answers,
    }
    _write_json(run / METADATA_NAME, payload)
    return payload


def write_run(
    run: Path,
    judgments: Sequence[Judgment],
    provider: str,
    model: str,
) -> dict:
    """Write this judge's files under ``run/judges/<model>``. Returns the metrics object."""
    folder = judge_dir(run, model)
    folder.mkdir(parents=True, exist_ok=True)
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
    (folder / JUDGMENTS_NAME).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    metrics = summarize(judgments, provider, model)
    _write_json(folder / METRICS_NAME, metrics)
    return metrics


def judge_answers(
    answers_path: Path,
    run: Path,
    complete: Complete,
    provider: str,
    model: str,
) -> dict:
    """Load the frozen prompts, score the answer file, and write the run."""
    judgments = score_rows(load_prompts(), load_answers(answers_path), load_rubric(), complete)
    return write_run(run, judgments, provider, model)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
