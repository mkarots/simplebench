"""The 100 prompts and the rubric text. Scores never go in data/."""

import json
from dataclasses import dataclass
from pathlib import Path

DATASET_ID = "simplebench"
PROMPT_COUNT = 100


class PromptError(ValueError):
    """A prompt file or an answer file broke the format contract."""


@dataclass(frozen=True)
class Prompt:
    """One unfinished sentence. ``prompt`` is the text the model continues."""

    id: str
    prompt: str


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "data" / "prompts.jsonl").is_file():
            return parent
    raise PromptError("could not find the repository root (pyproject.toml and data/prompts.jsonl)")


def dataset_dir() -> Path:
    return repo_root() / "data"


def load_prompts(path: Path | None = None) -> tuple[Prompt, ...]:
    """Read the frozen prompt list. Ids are c001 through c100, in that order."""
    path = dataset_dir() / "prompts.jsonl" if path is None else path
    prompts: list[Prompt] = []
    seen: set[str] = set()
    for number, line in enumerate(_lines(path), start=1):
        row = _object(path, number, line)
        prompt_id = row.get("id")
        prompt = row.get("prompt")
        expected = f"c{number:03d}"
        if prompt_id != expected:
            raise PromptError(f"{path}:{number}: expected id {expected}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise PromptError(f"{path}:{number}: prompt must be non-empty text")
        if "\n" in prompt or prompt != prompt.strip() or prompt.endswith("."):
            raise PromptError(f"{path}:{number}: prompt must be one unfinished line")
        if prompt in seen:
            raise PromptError(f"{path}:{number}: repeated prompt")
        seen.add(prompt)
        prompts.append(Prompt(prompt_id, prompt))
    if len(prompts) != PROMPT_COUNT:
        raise PromptError(f"{path}: expected {PROMPT_COUNT} prompts, found {len(prompts)}")
    return tuple(prompts)


def load_rubric(path: Path | None = None) -> str:
    """The rubric file is the system text sent to the judge."""
    path = dataset_dir() / "rubric.txt" if path is None else path
    if not path.is_file():
        raise PromptError(f"missing rubric: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise PromptError(f"rubric is empty: {path}")
    return text + "\n"


def load_answers(path: Path) -> dict[str, str]:
    """Map prompt id to completion text. Every id must be a new string."""
    answers: dict[str, str] = {}
    for number, line in enumerate(_lines(path), start=1):
        row = _object(path, number, line)
        prompt_id = row.get("id")
        completion = row.get("completion")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise PromptError(f"{path}:{number}: id must be a string")
        if prompt_id in answers:
            raise PromptError(f"{path}:{number}: repeated id {prompt_id}")
        if not isinstance(completion, str):
            raise PromptError(f"{path}:{number}: completion must be a string")
        answers[prompt_id] = completion
    return answers


def _lines(path: Path) -> list[str]:
    if not path.is_file():
        raise PromptError(f"missing file: {path}")
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _object(path: Path, number: int, line: str) -> dict:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise PromptError(f"{path}:{number}: not JSON") from exc
    if not isinstance(row, dict):
        raise PromptError(f"{path}:{number}: not an object")
    return row
