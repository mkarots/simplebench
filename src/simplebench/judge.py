"""Score one completion against the rubric.

``complete(system, user)`` is the only call into a provider. Tests pass a
stand-in. ``openai_complete`` and ``anthropic_complete`` are the two
stand-ins that post JSON.
"""

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass

JUDGE_CHAR_LIMIT = 400
DEFAULT_MODEL = {
    "openai": "gpt-4.1",
    "anthropic": "claude-sonnet-4-5",
}
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
Complete = Callable[[str, str], str]


class JudgeError(ValueError):
    """The judge reply or the judge request could not be used."""


@dataclass(frozen=True)
class Judgment:
    """One row. ``empty`` rows were scored here and were not sent."""

    id: str
    score: int
    reason: str
    truncated: bool
    empty: bool


def user_message(prompt: str, completion: str, truncated: bool) -> str:
    lines = ["Prompt:", prompt, "", "Completion:", completion]
    if truncated:
        lines.extend(["", "The completion continues and was cut here."])
    return "\n".join(lines)


def parse_judgment(text: str) -> tuple[int, str]:
    """Read ``score`` and ``reason`` from the judge reply."""
    match = _OBJECT.search(text.strip())
    if match is None:
        raise JudgeError("judge reply has no JSON object")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeError("judge reply JSON did not parse") from exc
    if not isinstance(data, dict) or "score" not in data:
        raise JudgeError("judge reply has no score")
    score = data["score"]
    if isinstance(score, str) and score.strip() in {"0", "1", "2"}:
        score = int(score.strip())
    if isinstance(score, bool) or score not in (0, 1, 2):
        raise JudgeError(f"judge score must be 0, 1, or 2, got {score!r}")
    reason = data.get("reason", "")
    if not isinstance(reason, str):
        raise JudgeError("judge reason is not text")
    reason = " ".join(reason.split())
    return score, reason or "no reason"


def score_completion(
    prompt_id: str,
    prompt: str,
    completion: str,
    rubric: str,
    complete: Complete,
) -> Judgment:
    """Score one prompt. A blank completion is 0 and does not call ``complete``."""
    if not completion.strip():
        return Judgment(prompt_id, 0, "empty", False, True)
    judged = completion[:JUDGE_CHAR_LIMIT]
    truncated = len(completion) > JUDGE_CHAR_LIMIT
    raw = complete(rubric, user_message(prompt, judged, truncated))
    score, reason = parse_judgment(raw)
    return Judgment(prompt_id, score, reason, truncated, False)


def post_json(url: str, headers: Mapping[str, str], body: dict) -> dict:
    """POST JSON and return the object. The error text does not include headers."""
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise JudgeError(f"judge request failed: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise JudgeError(f"judge request failed: {exc.reason}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JudgeError("judge response was not JSON") from exc
    if not isinstance(data, dict):
        raise JudgeError("judge response was not an object")
    return data


def openai_complete(
    api_key: str,
    model: str,
    system: str,
    user: str,
    post: Callable[[str, Mapping[str, str], dict], dict] = post_json,
) -> str:
    data = post(
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        _openai_body(model, system, user),
    )
    return _openai_text(data)


def anthropic_complete(
    api_key: str,
    model: str,
    system: str,
    user: str,
    post: Callable[[str, Mapping[str, str], dict], dict] = post_json,
) -> str:
    data = post(
        "https://api.anthropic.com/v1/messages",
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        {
            "model": model,
            "max_tokens": 200,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
    )
    return _anthropic_text(data)


def _openai_body(model: str, system: str, user: str) -> dict:
    """gpt-4.1 takes temperature 0. gpt-5 only accepts the default temperature."""
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if model.startswith("gpt-5"):
        body["max_completion_tokens"] = 1000
        return body
    body["temperature"] = 0
    body["max_tokens"] = 200
    return body


def _openai_text(data: dict) -> str:
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise JudgeError("openai response has no message content") from exc
    if not isinstance(text, str):
        raise JudgeError("openai response content is not text")
    return text


def _anthropic_text(data: dict) -> str:
    content = data.get("content")
    if not isinstance(content, list):
        raise JudgeError("anthropic response has no content list")
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            return block["text"]
    raise JudgeError("anthropic response has no text block")
