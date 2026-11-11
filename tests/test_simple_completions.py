"""Stem list and judge checks. No network."""

import json
from pathlib import Path

import pytest

from simple_completions.judge import (
    JUDGE_CHAR_LIMIT,
    Judgment,
    JudgeError,
    anthropic_complete,
    openai_complete,
    parse_judgment,
    score_completion,
)
from simple_completions.prompts import PromptError, Stem, load_answers, load_rubric, load_stems, repo_root
from simple_completions.score import score_rows, summarize, write_run


def test_stem_file_has_one_hundred_unique_unfinished_lines():
    stems = load_stems()
    assert len(stems) == 100
    assert stems[0] == Stem("c001", "education is important because")
    assert [stem.id for stem in stems] == [f"c{number:03d}" for number in range(1, 101)]
    assert len({stem.prompt for stem in stems}) == 100
    assert all(stem.prompt == stem.prompt.strip() and not stem.prompt.endswith(".") for stem in stems)


def test_readme_quotes_the_rubric_lines():
    root = repo_root()
    rubric = (root / "data" / "rubric.txt").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    for line in rubric.splitlines():
        if line.startswith(("2:", "1:", "0:")):
            assert line in readme
    assert load_rubric().strip() == rubric.strip()


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
    assert row == Judgment("c001", 0, "empty", False, True)


def test_long_completion_is_cut_before_the_judge_sees_it():
    seen: dict[str, str] = {}

    def complete(system: str, user: str) -> str:
        seen["user"] = user
        return '{"score": 2, "reason": "it continues"}'

    completion = "word " * 200
    row = score_completion("c002", "sleep is important because", completion, "rubric\n", complete)
    assert row.truncated is True
    assert row.score == 2
    assert "was cut here" in seen["user"]
    assert completion[:JUDGE_CHAR_LIMIT] in seen["user"]
    assert completion not in seen["user"]


def test_answer_ids_must_match_the_stems(tmp_path: Path):
    stems = (Stem("c001", "education is important because"), Stem("c002", "sleep is important because"))
    with pytest.raises(PromptError, match="missing"):
        score_rows(stems, {"c001": "it helps"}, "rubric", lambda system, user: "{}")
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
    stems = (Stem("c001", "education is important because"), Stem("c002", "sleep is important because"))
    answers = {"c001": "it helps people learn", "c002": ""}

    def complete(system: str, user: str) -> str:
        assert "education is important because" in user
        return '{"score": 2, "reason": "it finishes the reason"}'

    rows = score_rows(stems, answers, "rubric\n", complete)
    metrics = write_run(tmp_path, rows, "openai", "gpt-4.1")
    assert metrics["mean_score"] == 1.0
    assert metrics["counts"] == {"0": 1, "1": 0, "2": 1}
    assert metrics["pass_rate"] == 0.5
    assert metrics["empty"] == 1
    assert metrics["judged"] == 1
    assert summarize(rows, "openai", "gpt-4.1") == metrics
    saved = [json.loads(line) for line in (tmp_path / "judgments.jsonl").read_text().splitlines()]
    assert [row["id"] for row in saved] == ["c001", "c002"]
    assert saved[1]["empty"] is True


def test_openai_and_anthropic_adapters_read_the_text_and_send_the_key():
    calls: list[tuple[str, dict, dict]] = []

    def post(url: str, headers: dict, body: dict) -> dict:
        calls.append((url, dict(headers), body))
        if "openai" in url:
            return {"choices": [{"message": {"content": '{"score": 2, "reason": "ok"}'}}]}
        return {"content": [{"type": "text", "text": '{"score": 1, "reason": "partial"}'}]}

    assert openai_complete("sk-test", "gpt-4.1", "rubric", "stem", post) == '{"score": 2, "reason": "ok"}'
    assert anthropic_complete("ant-test", "claude-sonnet-4-5", "rubric", "stem", post) == (
        '{"score": 1, "reason": "partial"}'
    )
    assert calls[0][1]["Authorization"] == "Bearer sk-test"
    assert calls[0][2]["model"] == "gpt-4.1"
    assert calls[0][2]["temperature"] == 0
    assert calls[1][1]["x-api-key"] == "ant-test"
    assert calls[1][2]["model"] == "claude-sonnet-4-5"


def test_judge_answers_scores_every_frozen_stem(tmp_path: Path):
    from simple_completions.score import judge_answers

    stems = load_stems()
    answers = tmp_path / "answers.jsonl"
    answers.write_text(
        "".join(
            json.dumps({"id": stem.id, "completion": "it continues the thought"}) + "\n" for stem in stems
        ),
        encoding="utf-8",
    )

    def complete(system: str, user: str) -> str:
        assert system.startswith("Score the completion")
        return '{"score": 2, "reason": "it finishes"}'

    metrics = judge_answers(answers, tmp_path / "run", complete, "openai", "gpt-4.1")
    assert metrics["rows"] == 100
    assert metrics["mean_score"] == 2.0
    assert metrics["judged"] == 100
    assert metrics["empty"] == 0


def test_judge_command_stops_when_the_key_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from simple_completions.__main__ import main

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        main(["judge", "--answers", str(tmp_path / "answers.jsonl"), "--run", str(tmp_path / "run")])


def test_adapter_rejects_a_response_with_no_text():
    with pytest.raises(JudgeError, match="message content"):
        openai_complete("sk", "gpt-4.1", "r", "u", lambda url, headers, body: {})
    with pytest.raises(JudgeError, match="text block"):
        anthropic_complete("ant", "claude", "r", "u", lambda url, headers, body: {"content": []})
