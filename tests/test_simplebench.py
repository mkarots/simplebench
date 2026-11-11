"""Prompt list and judge checks. No network."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from simplebench.judge import (
    Judgment,
    JudgeError,
    anthropic_complete,
    openai_complete,
    parse_judgment,
    score_completion,
)
from simplebench.answers import render_answers
from simplebench.prompts import Prompt, PromptError, dataset_dir, exam_stamp, load_answers, load_prompts
from simplebench.score import score_rows, summarize, write_metadata, write_run


def test_prompt_file_has_one_hundred_unique_unfinished_lines():
    prompts = load_prompts()
    assert len(prompts) == 100
    assert prompts[0] == Prompt("c001", "education is important because")
    assert [item.id for item in prompts] == [f"c{number:03d}" for number in range(1, 101)]
    assert len({item.prompt for item in prompts}) == 100
    assert all(item.prompt == item.prompt.strip() and not item.prompt.endswith(".") for item in prompts)


def test_parse_plain_json_and_a_fenced_string_score():
    assert parse_judgment('{"score": 2, "reason": "it finishes the thought"}') == (
        2,
        "it finishes the thought",
    )
    fenced = '```json\n{"score": "1", "reason": "it wanders"}\n```'
    assert parse_judgment(fenced) == (1, "it wanders")


def test_parse_rejects_a_score_outside_the_rubric():
    with pytest.raises(JudgeError, match="0, 1, or 2"):
        parse_judgment('{"score": 3, "reason": "too high"}')
    with pytest.raises(JudgeError, match="no JSON"):
        parse_judgment("looks fine")
    with pytest.raises(JudgeError, match="0, 1, or 2"):
        parse_judgment('{"score": true, "reason": "bool"}')


def test_blank_completion_scores_zero_without_calling_the_judge():
    def complete(system: str, user: str) -> str:
        raise AssertionError("blank rows are not sent")

    row = score_completion("c001", "education is important because", "  ", "rubric", complete)
    assert row == Judgment("c001", 0, "empty", True)


def test_long_completion_is_sent_whole():
    seen: dict[str, str] = {}

    def complete(system: str, user: str) -> str:
        seen["user"] = user
        return '{"score": 2, "reason": "it continues"}'

    completion = "word " * 200
    row = score_completion("c002", "sleep is important because", completion, "rubric\n", complete)
    assert row == Judgment("c002", 2, "it continues", False)
    assert seen["user"].startswith("Prompt:")
    assert completion in seen["user"]
    assert "was cut here" not in seen["user"]


def test_answer_ids_must_match_the_prompts(tmp_path: Path):
    prompts = (Prompt("c001", "education is important because"), Prompt("c002", "sleep is important because"))
    with pytest.raises(PromptError, match="prompt ids"):
        score_rows(prompts, {"c001": "it helps"}, "rubric", lambda system, user: "{}")
    repeated = tmp_path / "repeated.jsonl"
    repeated.write_text(
        '{"id": "c001", "completion": "one"}\n{"id": "c001", "completion": "two"}\n',
        encoding="utf-8",
    )
    with pytest.raises(PromptError, match="repeated"):
        load_answers(repeated)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "c001", "completion": 1}\n', encoding="utf-8")
    with pytest.raises(PromptError, match="completion"):
        load_answers(bad)
    bad.write_text("not json\n", encoding="utf-8")
    with pytest.raises(PromptError, match="not JSON"):
        load_answers(bad)


def test_mean_score_is_the_decision_number(tmp_path: Path):
    prompts = (Prompt("c001", "education is important because"), Prompt("c002", "sleep is important because"))
    answers = {"c001": "it helps people learn", "c002": ""}

    def complete(system: str, user: str) -> str:
        assert "education is important because" in user
        return '{"score": 2, "reason": "it finishes the reason"}'

    rows = score_rows(prompts, answers, "rubric\n", complete)
    metrics = write_run(tmp_path, rows, "openai", "gpt-4.1")
    assert metrics["exam"] == "simplebench-1"
    assert metrics["prompts_sha256"] == exam_stamp()["prompts_sha256"]
    assert metrics["rubric_sha256"] == exam_stamp()["rubric_sha256"]
    assert metrics["mean_score"] == 1.0
    assert metrics["counts"] == {"0": 1, "1": 0, "2": 1}
    assert metrics["pass_rate"] == 0.5
    assert metrics["empty"] == 1
    assert metrics["judged"] == 1
    assert summarize(rows, "openai", "gpt-4.1") == metrics
    saved_path = tmp_path / "judges" / "gpt-4.1" / "judgments.jsonl"
    saved = [json.loads(line) for line in saved_path.read_text().splitlines()]
    assert [row["id"] for row in saved] == ["c001", "c002"]
    assert saved[1]["empty"] is True
    assert "truncated" not in saved[0]
    write_run(tmp_path, rows, "openai", "gpt-5")
    assert (tmp_path / "judges" / "gpt-5" / "metrics.json").is_file()
    assert (tmp_path / "judges" / "gpt-4.1" / "metrics.json").is_file()
    meta = write_metadata(tmp_path, "openai", "small-model")
    assert json.loads((tmp_path / "metadata.json").read_text()) == meta
    assert meta["model"] == "small-model"
    assert meta["answers"] == "answers.jsonl"


def test_exam_stamp_is_the_id_and_the_bytes_of_the_two_files():
    stamp = exam_stamp()
    folder = dataset_dir()
    assert stamp == {
        "exam": "simplebench-1",
        "prompts_sha256": hashlib.sha256((folder / "prompts.jsonl").read_bytes()).hexdigest(),
        "rubric_sha256": hashlib.sha256((folder / "rubric.txt").read_bytes()).hexdigest(),
    }
    assert stamp["prompts_sha256"] != stamp["rubric_sha256"]


def test_openai_and_anthropic_adapters_read_the_text_and_send_the_key():
    calls: list[tuple[str, dict, dict]] = []

    def post(url: str, headers: dict, body: dict) -> dict:
        calls.append((url, dict(headers), body))
        if "openai" in url:
            return {"choices": [{"message": {"content": '{"score": 2, "reason": "ok"}'}}]}
        return {"content": [{"type": "text", "text": '{"score": 1, "reason": "partial"}'}]}

    assert openai_complete("sk-test", "gpt-4.1", "rubric", "prompt", post) == '{"score": 2, "reason": "ok"}'
    assert anthropic_complete("ant-test", "claude-sonnet-4-5", "rubric", "prompt", post) == (
        '{"score": 1, "reason": "partial"}'
    )
    assert calls[0][1]["Authorization"] == "Bearer sk-test"
    assert calls[0][2]["model"] == "gpt-4.1"
    assert calls[0][2]["temperature"] == 0
    assert calls[1][1]["x-api-key"] == "ant-test"
    assert calls[1][2]["model"] == "claude-sonnet-4-5"


def test_judge_answers_scores_every_prompt(tmp_path: Path):
    from simplebench.score import judge_answers

    prompts = load_prompts()
    answers = tmp_path / "answers.jsonl"
    answers.write_text(
        "".join(
            json.dumps({"id": item.id, "completion": "it continues the thought"}) + "\n" for item in prompts
        ),
        encoding="utf-8",
    )

    def complete(system: str, user: str) -> str:
        assert system.startswith("Score the completion of the prompt")
        return '{"score": 2, "reason": "it finishes"}'

    run = tmp_path / "run"
    metrics = judge_answers(answers, run, complete, "openai", "gpt-4.1")
    assert metrics["rows"] == 100
    assert metrics["mean_score"] == 2.0
    assert metrics["judged"] == 100
    assert metrics["empty"] == 0
    assert (run / "judges" / "gpt-4.1" / "metrics.json").is_file()
    assert not (run / "metrics.json").exists()


def test_load_env_fills_a_missing_key_and_keeps_an_existing_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from simplebench.__main__ import load_env

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "already")
    path = tmp_path / ".env"
    path.write_text(
        "# comment\n\nOPENAI_API_KEY=sk-from-file\nANTHROPIC_API_KEY=from-file\n",
        encoding="utf-8",
    )
    load_env(path)
    load_env(tmp_path / "missing.env")
    assert os.environ["OPENAI_API_KEY"] == "sk-from-file"
    assert os.environ["ANTHROPIC_API_KEY"] == "already"


def test_judge_command_stops_when_the_key_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from simplebench.__main__ import main

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        main(["judge", "--answers", str(tmp_path / "answers.jsonl"), "--run", str(tmp_path / "run")])


def test_gpt5_omits_temperature_and_caps_completion_tokens():
    seen: dict = {}

    def post(url: str, headers: dict, body: dict) -> dict:
        seen.update(body)
        return {"choices": [{"message": {"content": '{"score": 2, "reason": "ok"}'}}]}

    openai_complete("sk", "gpt-5", "rubric", "prompt", post)
    assert "temperature" not in seen
    assert "max_tokens" not in seen
    assert seen["max_completion_tokens"] == 1000


def test_render_answers_keeps_prompt_order_and_collapses_whitespace():
    prompts = (
        Prompt("c001", "education is important because"),
        Prompt("c002", "sleep is important because"),
    )
    text = render_answers(prompts, {"c002": "  it\nrestores  ", "c001": "it helps"})
    rows = [json.loads(line) for line in text.splitlines()]
    assert rows == [
        {"id": "c001", "completion": "it helps"},
        {"id": "c002", "completion": "it restores"},
    ]
    with pytest.raises(PromptError, match="missing completion"):
        render_answers(prompts, {"c001": "it helps"})
    with pytest.raises(PromptError, match="completion must be a string"):
        render_answers(prompts, {"c001": "it helps", "c002": 1})


def test_reference_sends_the_prompt_as_the_only_message():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "examples" / "answer_openai.py"
    spec = importlib.util.spec_from_file_location("answer_openai", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    seen: dict = {}

    class _Message:
        content = "the reply"

    class _Reply:
        choices = [type("Choice", (), {"message": _Message()})()]

    class _Completions:
        def create(self, **kwargs):
            seen.update(kwargs)
            return _Reply()

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    assert module.complete(_Client(), "gpt-4.1", "education is important because") == "the reply"
    assert seen["model"] == "gpt-4.1"
    assert seen["messages"] == [{"role": "user", "content": "education is important because"}]
    assert seen["temperature"] == 0
    assert all(message["role"] == "user" for message in seen["messages"])


def test_adapter_rejects_a_response_with_no_text():
    with pytest.raises(JudgeError, match="message content"):
        openai_complete("sk", "gpt-4.1", "r", "u", lambda url, headers, body: {})
    with pytest.raises(JudgeError, match="text block"):
        anthropic_complete("ant", "claude", "r", "u", lambda url, headers, body: {"content": []})
