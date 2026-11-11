"""One-off smoke run. Reads keys from the environment. Writes under runs/smoke."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from simplebench.judge import (
    JudgeError,
    anthropic_complete,
    openai_complete,
    parse_judgment,
    post_json,
    score_completion,
)
from simplebench.prompts import load_answers, load_rubric, load_stems
from simplebench.score import write_metadata, write_run

ROOT = Path(__file__).resolve().parents[2]
ANSWERS = ROOT / "runs" / "smoke" / "answers.jsonl"
WORKERS = 8


def _continue_stem(key: str, stem: str) -> str:
    """Send the stem alone. No system instruction."""
    data = post_json(
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        {
            "model": "gpt-4.1",
            "temperature": 0,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": stem}],
        },
    )
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise JudgeError("openai response has no message content") from exc
    if not isinstance(text, str):
        raise JudgeError("openai response content is not text")
    return text


def _retry(call):
    last = None
    for attempt in range(6):
        try:
            return call()
        except JudgeError as exc:
            last = exc
            text = str(exc)
            if not any(token in text for token in ("429", "500", "502", "503", "529", "overloaded", "no JSON")):
                raise
            time.sleep(2**attempt)
    raise last


def generate() -> None:
    stems = load_stems()
    key = os.environ["OPENAI_API_KEY"]

    def one(stem):
        text = _retry(lambda: _continue_stem(key, stem.prompt))
        return stem.id, " ".join(text.strip().split())

    rows = {}
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(one, stem) for stem in stems]
        for future in as_completed(futures):
            stem_id, completion = future.result()
            rows[stem_id] = completion
    lines = [json.dumps({"id": stem.id, "completion": rows[stem.id]}) for stem in stems]
    ANSWERS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {ANSWERS} rows={len(lines)}")


def score(provider: str, model: str) -> None:
    stems = load_stems()
    answers = load_answers(ANSWERS)
    rubric = load_rubric()
    key_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    key = os.environ[key_name]
    poster = openai_complete if provider == "openai" else anthropic_complete

    def complete(system: str, user: str) -> str:
        raw = ""

        def call() -> str:
            nonlocal raw
            raw = poster(key, model, system, user)
            try:
                parse_judgment(raw)
            except JudgeError as exc:
                raise JudgeError(f"{exc}: {raw[:160]!r}") from exc
            return raw

        return _retry(call)

    ordered = [None] * len(stems)

    def one(index, stem):
        return index, score_completion(stem.id, stem.prompt, answers[stem.id], rubric, complete)

    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(one, index, stem) for index, stem in enumerate(stems)]
        for future in as_completed(futures):
            index, row = future.result()
            ordered[index] = row
    run = ROOT / "runs" / "smoke"
    metrics = write_run(run, ordered, provider, model)
    print(
        f"{model} mean_score={metrics['mean_score']:.2f} "
        f"counts={metrics['counts']} pass_rate={metrics['pass_rate']:.2f}"
    )


if __name__ == "__main__":
    import sys

    opus = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-4-5"
    write_metadata(ROOT / "runs" / "smoke", "openai", "gpt-4.1")
    if os.environ.get("SMOKE_SCORE_ONLY") != "1":
        generate()
        score("openai", "gpt-4.1")
    score("openai", "gpt-5")
    score("anthropic", opus)
