"""Prompt list and judge checks. No network."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from simplebench.judge import (
    Judgment,
    JudgeError,
    openai_complete,
    parse_judgment,
    score_completion,
)
from simplebench.answers import render_answers
from simplebench.prompts import (
    Exam,
    Judge,
    Prompt,
    PromptError,
    dataset_dir,
    exam_folder,
    load_answers,
    load_exam,
    load_prompts,
    repo_root,
)
from simplebench.score import compare_runs, run_dir, score_rows, summarize, write_run


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
    exam = Exam("simplebench-1", prompts, "rubric\n", Judge("openai", "gpt-4.1"), "abc")
    answers_path = tmp_path / "answers.jsonl"
    answers_path.write_text('{"id": "c001", "completion": "it helps"}\n{"id": "c002", "completion": ""}\n')
    run = tmp_path / "tiny" / "step-1000"
    metrics = write_run(run, rows, exam, "tiny", "step-1000", answers_path)
    assert metrics["exam"] == "simplebench-1"
    assert metrics["exam_sha256"] == "abc"
    assert metrics["model"] == "tiny"
    assert metrics["checkpoint"] == "step-1000"
    assert metrics["judge"] == {"provider": "openai", "model": "gpt-4.1"}
    assert "provider" not in metrics
    assert metrics["mean_score"] == 1.0
    assert metrics["counts"] == {"0": 1, "1": 0, "2": 1}
    assert metrics["pass_rate"] == 0.5
    assert metrics["empty"] == 1
    assert metrics["judged"] == 1
    assert summarize(rows, exam, "tiny", "step-1000") == metrics
    assert (run / "answers.jsonl").read_text() == answers_path.read_text()
    write_run(run, rows, exam, "tiny", "step-1000", run / "answers.jsonl")
    assert (run / "answers.jsonl").read_text() == answers_path.read_text()
    saved_path = run / "judgments.jsonl"
    saved = [json.loads(line) for line in saved_path.read_text().splitlines()]
    assert [row["id"] for row in saved] == ["c001", "c002"]
    assert saved[1]["empty"] is True
    assert "truncated" not in saved[0]
    assert (run / "metrics.json").is_file()
    assert not (run / "judges").exists()


def test_exam_file_points_at_the_prompt_file_and_the_rubric_file(tmp_path: Path):
    exam = load_exam()
    folder = dataset_dir()
    raw = (
        (folder / "config.json").read_bytes()
        + (folder / "prompts.jsonl").read_bytes()
        + (folder / "rubric.txt").read_bytes()
    )
    assert folder == repo_root() / "datasets" / "simplebench-1"
    assert exam.id == "simplebench-1"
    assert exam.judge == Judge("openai", "gpt-4.1")
    assert exam.rubric == (folder / "rubric.txt").read_text(encoding="utf-8")
    assert json.loads((folder / "config.json").read_text())["rubric"] == "rubric.txt"
    assert len(exam.prompts) == 100
    assert exam.sha256 == hashlib.sha256(raw).hexdigest()
    published = json.loads((repo_root() / "runs" / "gpt-4.1" / "smoke" / "metrics.json").read_text())
    assert exam.sha256 == published["exam_sha256"]
    with pytest.raises(PromptError, match="missing exam"):
        load_exam(tmp_path)
    (tmp_path / "config.json").write_text(
        '{"exam": "simplebench/2", "prompts": "prompts.jsonl", "rubric": "rubric.txt"}\n',
        encoding="utf-8",
    )
    with pytest.raises(PromptError, match="exam id"):
        load_exam(tmp_path)
    other = tmp_path / "simplebench-1"
    other.mkdir()
    (other / "config.json").write_text(
        '{"exam": "simplebench-1", "prompts": "prompts.jsonl", "rubric": "rubric.txt", "judge": {"provider": "openai", "model": "gpt-5"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(PromptError, match="openai/gpt-4.1"):
        load_exam(other)


def test_datasets_folder_holds_one_exam(tmp_path: Path):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    with pytest.raises(PromptError, match="found: none"):
        exam_folder(datasets)
    first = datasets / "simplebench-1"
    first.mkdir()
    (first / "config.json").write_text("{}\n", encoding="utf-8")
    assert exam_folder(datasets) == first
    second = datasets / "simplebench-2"
    second.mkdir()
    (second / "config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(PromptError, match="simplebench-1, simplebench-2"):
        exam_folder(datasets)
    (datasets / "notes").mkdir()
    with pytest.raises(PromptError, match="simplebench-1, simplebench-2"):
        exam_folder(datasets)


def test_exam_id_must_match_its_folder(tmp_path: Path):
    folder = tmp_path / "other-name"
    folder.mkdir()
    (folder / "config.json").write_text(
        '{"exam": "simplebench-1", "prompts": "prompts.jsonl", "rubric": "rubric.txt", "judge": {"provider": "openai", "model": "gpt-4.1"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(PromptError, match="folder name"):
        load_exam(folder)


def test_exam_rejects_a_missing_empty_or_escaped_rubric(tmp_path: Path):
    folder = tmp_path / "simplebench-1"
    folder.mkdir()
    exam_path = folder / "config.json"
    exam_path.write_text(
        '{"exam": "simplebench-1", "prompts": "prompts.jsonl", "rubric": "rubric.txt", "judge": {"provider": "openai", "model": "gpt-4.1"}}\n',
        encoding="utf-8",
    )
    (folder / "prompts.jsonl").write_text('{"id": "c001", "prompt": "x"}\n', encoding="utf-8")
    with pytest.raises(PromptError, match="missing rubric"):
        load_exam(folder)
    (folder / "rubric.txt").write_text("   \n", encoding="utf-8")
    with pytest.raises(PromptError, match="non-empty"):
        load_exam(folder)
    exam_path.write_text(
        '{"exam": "simplebench-1", "prompts": "prompts.jsonl", "rubric": "../rubric.txt", "judge": {"provider": "openai", "model": "gpt-4.1"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(PromptError, match="rubric must name a file"):
        load_exam(folder)


def test_openai_adapter_calls_gpt_4_1():
    calls: list[tuple[str, dict, dict]] = []

    def post(url: str, headers: dict, body: dict) -> dict:
        calls.append((url, dict(headers), body))
        return {"choices": [{"message": {"content": '{"score": 2, "reason": "ok"}'}}]}

    assert openai_complete("sk-test", "gpt-4.1", "rubric", "prompt", post) == '{"score": 2, "reason": "ok"}'
    assert calls[0][1]["Authorization"] == "Bearer sk-test"
    assert calls[0][2]["model"] == "gpt-4.1"
    assert calls[0][2]["temperature"] == 0
    assert calls[0][2]["max_tokens"] == 200


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
    metrics = judge_answers(answers, run, complete, load_exam(), "tiny", "step-1000")
    assert metrics["rows"] == 100
    assert metrics["model"] == "tiny"
    assert metrics["judge"]["model"] == "gpt-4.1"
    assert metrics["mean_score"] == 2.0
    assert metrics["judged"] == 100
    assert metrics["empty"] == 0
    assert (run / "answers.jsonl").read_bytes() == answers.read_bytes()
    assert (run / "metrics.json").is_file()
    assert (run / "judgments.jsonl").is_file()


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


def test_run_dir_is_runs_model_checkpoint(tmp_path: Path):
    assert run_dir("tiny", "step-1000", tmp_path) == tmp_path / "runs" / "tiny" / "step-1000"
    with pytest.raises(PromptError, match="path segment"):
        run_dir("tiny/other", "step-1000", tmp_path)
    with pytest.raises(PromptError, match="path segment"):
        run_dir("tiny", "..", tmp_path)
    with pytest.raises(PromptError, match="path segment"):
        run_dir(" ", "step-1000", tmp_path)


def test_judge_command_scores_under_runs_model_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import simplebench.__main__ as cli
    import simplebench.score as score

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(score, "repo_root", lambda: tmp_path)
    seen: dict[str, Path] = {}

    def fake_judge(answers: Path, run: Path, complete, exam: Exam, model: str, checkpoint: str) -> dict:
        seen["run"] = run
        assert exam.id == "simplebench-1"
        assert model == "tiny"
        assert checkpoint == "step-1000"
        return {"mean_score": 1.0, "pass_rate": 0.5}

    monkeypatch.setattr(cli, "judge_answers", fake_judge)
    cli.main(
        [
            "judge",
            "--answers",
            str(tmp_path / "answers.jsonl"),
            "--model",
            "tiny",
            "--checkpoint",
            "step-1000",
        ]
    )
    assert seen["run"] == tmp_path / "runs" / "tiny" / "step-1000"


def test_judge_command_stops_when_the_key_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from simplebench.__main__ import main

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        main(
            [
                "judge",
                "--answers",
                str(tmp_path / "answers.jsonl"),
                "--model",
                "tiny",
                "--checkpoint",
                "step-1000",
            ]
        )


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
    with pytest.raises(JudgeError, match="gpt-4.1"):
        openai_complete("sk", "gpt-5", "r", "u", lambda url, headers, body: {})


def _checkpoint(folder: Path, model: str, checkpoint: str, scores: dict[str, int], exam_sha: str = "abc") -> None:
    folder.mkdir(parents=True)
    rows = [
        json.dumps({"id": prompt_id, "score": score, "reason": "because", "empty": False})
        for prompt_id, score in scores.items()
    ]
    (folder / "judgments.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (folder / "metrics.json").write_text(
        json.dumps(
            {
                "exam": "simplebench-1",
                "exam_sha256": exam_sha,
                "model": model,
                "checkpoint": checkpoint,
                "mean_score": sum(scores.values()) / len(scores),
            }
        ),
        encoding="utf-8",
    )


def test_compare_names_the_prompts_whose_score_moved(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from simplebench.__main__ import main

    left = tmp_path / "tiny" / "step-1000"
    right = tmp_path / "tiny" / "step-2000"
    _checkpoint(left, "tiny", "step-1000", {"c001": 2, "c002": 1})
    _checkpoint(right, "tiny", "step-2000", {"c001": 2, "c002": 2})
    result = compare_runs(left, right)
    assert result["exam"] == "simplebench-1"
    assert result["left_mean"] == 1.5
    assert result["right_mean"] == 2.0
    assert result["changed"] == (("c002", 1, 2),)
    main(["compare", "--left", str(left), "--right", str(right)])
    printed = capsys.readouterr().out
    assert "tiny step-1000 mean_score=1.50" in printed
    assert "tiny step-2000 mean_score=2.00" in printed
    assert "c002 1 -> 2" in printed


def test_compare_rejects_a_different_exam(tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _checkpoint(left, "tiny", "step-1000", {"c001": 2})
    _checkpoint(right, "tiny", "step-2000", {"c001": 2}, exam_sha="def")
    with pytest.raises(PromptError, match="exams differ"):
        compare_runs(left, right)


def test_compare_rejects_a_folder_with_no_metrics(tmp_path: Path):
    with pytest.raises(PromptError, match="missing metrics"):
        compare_runs(tmp_path / "left", tmp_path / "right")


def test_compare_says_when_no_score_moved(tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _checkpoint(left, "tiny", "step-1000", {"c001": 2})
    _checkpoint(right, "tiny", "step-2000", {"c001": 2})
    assert compare_runs(left, right)["changed"] == ()
