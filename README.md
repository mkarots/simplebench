![simplebench](docs/simplebench.png)

# simplebench

Simplebench is a first slice of a sentence-completion benchmark for anyone who wants to score a model on the same 100 prompts.

## Terms

| Term | In this repo |
| --- | --- |
| Judge | The model that scores each answer 0, 1, or 2. [dataset/config.json](dataset/config.json) names it: openai / gpt-4.1. |
| Scorer | The command `python -m simplebench judge`. It sends each answer to the judge and writes the result. |
| Rubric | [dataset/rubric.txt](dataset/rubric.txt). The scoring rules the judge reads. |
| Exam | [dataset/config.json](dataset/config.json). It names the prompt file, the rubric file, and the judge. |
| Mean | The average of the 100 scores. This is the result of a run. |

## Judged so far

gpt-4.1 wrote the 100 answers and the current exam scored them. The checkpoint is [runs/gpt-4.1/smoke/](runs/gpt-4.1/smoke/). `model` in the metrics file is the model that wrote the answers. `judge` is gpt-4.1 from [dataset/config.json](dataset/config.json). The whole completion is sent, at temperature 0, with a 200-token reply cap. Each score is 0, 1, or 2. The mean is the result. To see why a mean is under 2, see [1](#note-1).

| Answers | Checkpoint | Judge | Mean | 2 | 1 | 0 | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-4.1 | smoke | gpt-4.1 | 1.96 | 96 | 4 | 0 | [runs/gpt-4.1/smoke/metrics.json](runs/gpt-4.1/smoke/metrics.json) |

That metrics file carries `exam` (`simplebench-1`) and `exam_sha256` beside `mean_score`. Compare two means when both fields match. The answers are copied into the checkpoint folder. They are not part of the exam.

## What is included

| Piece | Where | Role |
| --- | --- | --- |
| Dataset | [dataset/prompts.jsonl](dataset/prompts.jsonl) | The 100 prompts a model completes |
| Rubric | [dataset/rubric.txt](dataset/rubric.txt) | The scoring rules the judge reads |
| Model | yours, or [examples/answer_openai.py](examples/answer_openai.py) | Reads the prompts and writes `answers.jsonl` |
| Exam | [dataset/config.json](dataset/config.json) | Which prompt file, which rubric file, and which judge. |
| Scorer | [src/simplebench/__main__.py](src/simplebench/__main__.py) (`python -m simplebench judge`) | Reads `answers.jsonl` and writes `runs/<model>/<checkpoint>/` |

[examples/answer_openai.py](examples/answer_openai.py) is the reference for producing answers. The finished run of that script, scored with the current exam, is [runs/gpt-4.1/smoke/](runs/gpt-4.1/smoke/).

## A slice of the dataset

The first three prompts in [dataset/prompts.jsonl](dataset/prompts.jsonl#L1-L3):

```json
{"id": "c001", "prompt": "education is important because"}
{"id": "c002", "prompt": "sleep is important because"}
{"id": "c003", "prompt": "water is important because"}
```

Ids run from `c001` to `c100`. The matching lines in [runs/gpt-4.1/smoke/answers.jsonl](runs/gpt-4.1/smoke/answers.jsonl#L1-L3) are the gpt-4.1 replies. Each reply restates the prompt and keeps going:

- [c001](runs/gpt-4.1/smoke/answers.jsonl#L1) starts `**Education is important because** it provides individuals with the knowledge, skills, and values necessary to navigate the world`.
- [c002](runs/gpt-4.1/smoke/answers.jsonl#L2) starts `Sleep is important because it plays a vital role in your physical health, mental well-being, and overall quality of life`.
- [c003](runs/gpt-4.1/smoke/answers.jsonl#L3) starts `Water is important because: 1. **Essential for Life:** All living organisms need water to survive`.

`completion` is a string. `""` is valid and scores 0. Any other field is ignored.

## How to use it

1. From this directory, install:

```text
uv sync
```

2. Copy [.env.example](.env.example) to `.env` and put `OPENAI_API_KEY` in it. The reference answer script and the judge, gpt-4.1, both use that key. A key already set in the shell is left as it is. Do not commit `.env`.

3. Write `answers.jsonl` by running a model once on each prompt.

    1. Open [examples/answer_openai.py](examples/answer_openai.py). `complete` is the function to replace with your model. It takes the prompt text and returns the completion text. The prompt is the only message.
    2. Run the reference as it stands to call gpt-4.1, or run it after you replace `complete`:

```text
uv run python examples/answer_openai.py --out answers.jsonl
```

The file has one JSON object per line. The ids are `c001` through `c100`, each once, in the same shape as [runs/gpt-4.1/smoke/answers.jsonl](runs/gpt-4.1/smoke/answers.jsonl). A blank completion is `""` and scores 0.

4. Score that file. `<model>` is the model that wrote the answers. `<checkpoint>` is which checkpoint of that model.

```text
uv run python -m simplebench judge --answers answers.jsonl --model <model> --checkpoint <checkpoint>
```

5. Read `mean_score` from the line the command prints. The folder `runs/<model>/<checkpoint>/` holds `answers.jsonl`, `judgments.jsonl`, and `metrics.json`. `metrics.json` has `model`, `checkpoint`, `judge`, `exam`, `exam_sha256`, and `mean_score`. `model` is the model that wrote the answers. `judge` is the model that scored them. Each judgment line has `id`, `score`, `reason`, and `empty`. The next checkpoint repeats the command with a new `--checkpoint`.

6. Compare two checkpoints after both exist:

```text
uv run python -m simplebench compare --left runs/<model>/step-1000 --right runs/<model>/step-2000
```

The command prints both means and the prompt ids whose score changed. It stops when `exam` or `exam_sha256` differs.

These replies stop the run:

| Reply | Cause |
| --- | --- |
| `set OPENAI_API_KEY to call the judge` | `.env` has no key and the shell has no key |
| `answers must match the prompt ids` | An id is missing, repeated, or unknown |
| `completion must be a string` | `completion` was missing or not text |
| `judge score must be 0, 1, or 2` | The judge reply was not a rubric score |

## For agents

Submit a result for the user's model. Do not edit [dataset/prompts.jsonl](dataset/prompts.jsonl) or [dataset/config.json](dataset/config.json). Do not commit `.env`. Do not invent `OPENAI_API_KEY`. If `.env` is missing, copy [.env.example](.env.example) and ask the user for the key.

1. Open [examples/answer_openai.py](examples/answer_openai.py). Replace `complete` with a call to the user's model. Pass the prompt text as the only message. Do not add a system message.
2. Run `uv run python examples/answer_openai.py --out answers.jsonl`. That writes one JSON object per line, ids `c001` to `c100`, in the shape of [runs/gpt-4.1/smoke/answers.jsonl](runs/gpt-4.1/smoke/answers.jsonl).
3. Run `uv run python -m simplebench judge --answers answers.jsonl --model <model> --checkpoint <checkpoint>` from [src/simplebench/__main__.py](src/simplebench/__main__.py).
4. Report `mean_score` from `runs/<model>/<checkpoint>/metrics.json`. Leave that folder in the repo as the submission. [runs/gpt-4.1/smoke/metrics.json](runs/gpt-4.1/smoke/metrics.json) is a finished example.

## Notes

<a id="note-1"></a>

**1. Why the mean is under 2.** [examples/answer_openai.py](examples/answer_openai.py) sends each prompt as the only message. gpt-4.1 answers in chat: it restates the prompt, writes a list, or asks the reader to finish the sentence. No judge gave a 0.

[runs/gpt-4.1/smoke/](runs/gpt-4.1/smoke/) sent the whole completion. gpt-4.1 scored [c050](runs/gpt-4.1/smoke/judgments.jsonl#L50), [c061](runs/gpt-4.1/smoke/judgments.jsonl#L61), [c062](runs/gpt-4.1/smoke/judgments.jsonl#L62), and [c063](runs/gpt-4.1/smoke/judgments.jsonl#L63) as 1. Those replies explain the prompt, list several endings, or ask a question.
