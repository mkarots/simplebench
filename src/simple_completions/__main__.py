"""Sentence-stem commands. Run them from the repository root.

    uv run python -m simple_completions prompts
    uv run python -m simple_completions judge --answers answers.jsonl --run runs/their-model

``prompts`` prints the 100 stems.
``judge`` scores an answer file. The default judge is gpt-4.1.
Pass ``--provider anthropic`` to score with claude-sonnet-4-5.
"""

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from simple_completions.judge import DEFAULT_MODEL, JudgeError, anthropic_complete, openai_complete
from simple_completions.prompts import PromptError, load_stems
from simple_completions.score import judge_answers

_KEY = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def _prompts(_args: argparse.Namespace) -> None:
    for stem in load_stems():
        print(json.dumps({"id": stem.id, "prompt": stem.prompt}, ensure_ascii=False))


def _judge(args: argparse.Namespace) -> None:
    provider = args.provider
    model = args.model or DEFAULT_MODEL[provider]
    key_name = _KEY[provider]
    api_key = os.environ.get(key_name, "").strip()
    if not api_key:
        raise PromptError(f"set {key_name} to call the judge")
    poster = openai_complete if provider == "openai" else anthropic_complete

    def complete(system: str, user: str) -> str:
        return poster(api_key, model, system, user)

    metrics = judge_answers(args.answers, args.run, complete, provider, model)
    print(
        f"wrote {args.run / 'metrics.json'} mean_score={metrics['mean_score']:.2f} "
        f"pass_rate={metrics['pass_rate']:.2f} model={model}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="simple_completions", description="Sentence-stem exam.")
    sub = parser.add_subparsers(dest="command", required=True)

    prompts = sub.add_parser("prompts", help="Print the 100 stems as JSON lines.")
    prompts.set_defaults(func=_prompts)

    judge = sub.add_parser("judge", help="Score an answer file with a judge model.")
    judge.add_argument("--answers", type=Path, required=True)
    judge.add_argument("--run", type=Path, required=True)
    judge.add_argument("--provider", choices=tuple(DEFAULT_MODEL), default="openai")
    judge.add_argument("--model", default=None, help="Replaces the provider default.")
    judge.set_defaults(func=_judge)

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        args.func(args)
    except (PromptError, JudgeError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
