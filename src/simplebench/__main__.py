"""Sentence-completion commands. Run them from the repository root.

    uv run python examples/answer_openai.py --out answers.jsonl
    uv run python -m simplebench judge --answers answers.jsonl --model their-model --checkpoint step-1000
    uv run python -m simplebench compare --left runs/their-model/step-1000 --right runs/their-model/step-2000

The score lands in ``runs/<model>/<checkpoint>/``.

``judge`` scores an answer file with the one exam folder under ``datasets/``.
``compare`` prints the two means and the prompts whose score moved.
"""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from simplebench.judge import JudgeError, openai_complete
from simplebench.prompts import PromptError, load_exam, repo_root
from simplebench.score import compare_runs, judge_answers, run_dir


def load_env(path: Path) -> None:
    """Fill missing environment variables from a .env file. Existing values stay."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip("'\"" )


def _judge(args: argparse.Namespace) -> None:
    load_env(repo_root() / ".env")
    exam = load_exam()
    run = run_dir(args.model, args.checkpoint)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise PromptError("set OPENAI_API_KEY to call the judge")

    def complete(system: str, user: str) -> str:
        return openai_complete(api_key, exam.judge.model, system, user)

    metrics = judge_answers(args.answers, run, complete, exam, args.model, args.checkpoint)
    written = run / "metrics.json"
    print(
        f"wrote {written} mean_score={metrics['mean_score']:.2f} "
        f"pass_rate={metrics['pass_rate']:.2f} exam={exam.id} model={args.model}"
    )


def _compare(args: argparse.Namespace) -> None:
    result = compare_runs(args.left, args.right)
    print(f"exam {result['exam']}")
    print(
        f"{result['left_model']} {result['left_checkpoint']} mean_score={result['left_mean']:.2f}"
    )
    print(
        f"{result['right_model']} {result['right_checkpoint']} mean_score={result['right_mean']:.2f}"
    )
    if not result["changed"]:
        print("no score changed")
        return
    for prompt_id, left_score, right_score in result["changed"]:
        print(f"{prompt_id} {left_score} -> {right_score}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="simplebench", description="Sentence-completion exam.")
    sub = parser.add_subparsers(dest="command", required=True)

    judge = sub.add_parser("judge", help="Score an answer file with the one exam folder under datasets/.")
    judge.add_argument("--answers", type=Path, required=True)
    judge.add_argument("--model", required=True, help="Model that wrote the answers.")
    judge.add_argument("--checkpoint", required=True, help="Checkpoint of that model.")
    judge.set_defaults(func=_judge)

    compare = sub.add_parser("compare", help="Compare two checkpoints of one exam.")
    compare.add_argument("--left", type=Path, required=True, help="A runs/<model>/<checkpoint> folder.")
    compare.add_argument("--right", type=Path, required=True, help="The other checkpoint folder.")
    compare.set_defaults(func=_compare)

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        args.func(args)
    except (PromptError, JudgeError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
