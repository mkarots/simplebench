"""The prompt list and the exam files. Scores never go in dataset/."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from simplebench.judge import JUDGE_MODEL, JUDGE_PROVIDER

PROMPT_COUNT = 100


class PromptError(ValueError):
    """A prompt file or an answer file broke the format contract."""


@dataclass(frozen=True)
class Prompt:
    """One unfinished sentence. ``prompt`` is the text the model continues."""

    id: str
    prompt: str


@dataclass(frozen=True)
class Judge:
    """The model named in ``dataset/config.json``. The scorer calls this model."""

    provider: str
    model: str


@dataclass(frozen=True)
class Exam:
    """One scoring setup. ``sha256`` covers this file, its prompts, and its rubric."""

    id: str
    prompts: tuple[Prompt, ...]
    rubric: str
    judge: Judge
    sha256: str


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "dataset" / "prompts.jsonl").is_file():
            return parent
    raise PromptError("could not find the repository root (pyproject.toml and dataset/prompts.jsonl)")


def dataset_dir() -> Path:
    return repo_root() / "dataset"


def load_exam(root: Path | None = None) -> Exam:
    """Read ``dataset/config.json`` and the prompt and rubric files it names."""
    folder = dataset_dir() if root is None else root
    path = folder / "config.json"
    if not path.is_file():
        raise PromptError(f"missing exam: {path}")
    exam_bytes = path.read_bytes()
    data = json.loads(exam_bytes.decode("utf-8"))
    if not isinstance(data, dict):
        raise PromptError(f"{path}: exam must be an object")
    exam_id = data.get("exam")
    if not isinstance(exam_id, str) or _path_segment(exam_id) != exam_id:
        raise PromptError(f"{path}: exam id must be one path segment")
    judge = _judge(data.get("judge"), path)
    prompts_path = _named_file(folder, path, data.get("prompts"), "prompts")
    rubric_path = _named_file(folder, path, data.get("rubric"), "rubric")
    rubric = _rubric_text(rubric_path)
    prompts = load_prompts(prompts_path)
    digest = hashlib.sha256(exam_bytes + prompts_path.read_bytes() + rubric_path.read_bytes()).hexdigest()
    return Exam(exam_id, prompts, rubric, judge, digest)


def _judge(data: object, path: Path) -> Judge:
    if not isinstance(data, dict):
        raise PromptError(f"{path}: judge must be an object")
    provider = data.get("provider")
    model = data.get("model")
    if provider != JUDGE_PROVIDER or model != JUDGE_MODEL:
        raise PromptError(f"{path}: judge must be {JUDGE_PROVIDER}/{JUDGE_MODEL}")
    return Judge(provider, model)


def _named_file(folder: Path, exam_path: Path, name: object, field: str) -> Path:
    if not isinstance(name, str) or _path_segment(name) != name:
        raise PromptError(f"{exam_path}: {field} must name a file in the dataset folder")
    path = folder / name
    if not path.is_file():
        raise PromptError(f"missing {field}: {path}")
    return path


def _rubric_text(path: Path) -> str:
    rubric = path.read_text(encoding="utf-8")
    if not rubric.strip():
        raise PromptError(f"{path}: rubric must be non-empty text")
    return rubric if rubric.endswith("\n") else rubric + "\n"


def _path_segment(name: str) -> str:
    if name in {"", ".", ".."} or "/" in name or "\\" in name or name != name.strip():
        return ""
    return name


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
