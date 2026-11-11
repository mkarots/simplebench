# simplebench

Simplebench is a first slice of a sentence-completion benchmark for anyone who wants to score a model on the same 100 prompts.

## Terms

| Term | In this repo |
| --- | --- |
| Judge | The model that scores each answer 0, 1, or 2. The default judge is gpt-4.1. |
| Scorer | The command `python -m simplebench judge`. It sends each answer to the judge and writes the result. |
| Rubric | The rules for those scores, in [data/rubric.txt](data/rubric.txt). |
| Exam | `simplebench-1` in [src/simplebench/prompts.py](src/simplebench/prompts.py). The prompts, the rubric, the judge, and the way the completion is sent. |
| Mean | The average of the 100 scores. This is the result of a run. |

## Judged so far

gpt-4.1 wrote the 100 answers in [runs/smoke/answers.jsonl](runs/smoke/answers.jsonl). [runs/smoke/metadata.json](runs/smoke/metadata.json) names that model. Three judges scored the same file, each in its own folder under [runs/smoke/judges/](runs/smoke/judges/).

| Answers | Judge | Mean | 2 | 1 | 0 | Result |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-4.1 | gpt-4.1 | 1.97 | 97 | 3 | 0 | [runs/smoke/judges/gpt-4.1/metrics.json](runs/smoke/judges/gpt-4.1/metrics.json) |
| gpt-4.1 | gpt-5 | 1.44 | 44 | 56 | 0 | [runs/smoke/judges/gpt-5/metrics.json](runs/smoke/judges/gpt-5/metrics.json) |
| gpt-4.1 | claude-opus-4-5 | 1.80 | 80 | 20 | 0 | [runs/smoke/judges/claude-opus-4-5/metrics.json](runs/smoke/judges/claude-opus-4-5/metrics.json) |

Each score is 0, 1, or 2. The mean in each metrics file is the result. To see why a mean is under 2, see [1](#note-1).

Those three files were written before the exam id. That run showed the judge the first 400 characters. A metrics file written now carries `exam`, `prompts_sha256`, and `rubric_sha256` beside `mean_score`. Compare two means when `exam`, both hashes, and `model` are the same. The id is `simplebench-1`: the whole completion goes to gpt-4.1 at temperature 0, with a 200-token reply cap. Bump the id when the prompt file, the rubric, the judge, the temperature, the token cap, or how much of the completion is sent changes, and tag the repo with that id. The answers are not part of the exam.

## What is included

| Piece | Where | Role |
| --- | --- | --- |
| Dataset | [data/prompts.jsonl](data/prompts.jsonl), [data/rubric.txt](data/rubric.txt) | 100 prompts, and the 0–2 rubric the judge reads |
| Model | yours, or [examples/answer_openai.py](examples/answer_openai.py) | Reads the prompts and writes `answers.jsonl` |
| Scorer | [src/simplebench/__main__.py](src/simplebench/__main__.py) (`python -m simplebench judge`) | Reads `answers.jsonl` and writes `runs/<name>/judges/<judge>/metrics.json` |

[examples/answer_openai.py](examples/answer_openai.py) is the reference for producing answers. [runs/smoke/run_smoke.py](runs/smoke/run_smoke.py) produced the table above: it sends each prompt to gpt-4.1 with no extra instruction, then scores the answers with the three judges. Running it again calls the APIs and replaces [runs/smoke/](runs/smoke/).

## A slice of the dataset

The first three prompts in [data/prompts.jsonl](data/prompts.jsonl#L1-L3):

```json
{"id": "c001", "prompt": "education is important because"}
{"id": "c002", "prompt": "sleep is important because"}
{"id": "c003", "prompt": "water is important because"}
```

Ids run from `c001` to `c100`. The matching lines in [runs/smoke/answers.jsonl](runs/smoke/answers.jsonl#L1-L3) are the gpt-4.1 replies. Each reply restates the prompt and keeps going:

- [c001](runs/smoke/answers.jsonl#L1) starts `**Education is important because** it provides individuals with the knowledge, skills, and values necessary to navigate the world`.
- [c002](runs/smoke/answers.jsonl#L2) starts `Sleep is important because it plays a vital role in your physical health, mental well-being, and overall quality of life`.
- [c003](runs/smoke/answers.jsonl#L3) starts `Water is important because: 1. **Essential for Life:** All living organisms need water to survive`.

`completion` is a string. `""` is valid and scores 0. Any other field is ignored.

## How to use it

1. From this directory, install:

```text
uv sync
```

2. Copy [.env.example](.env.example) to `.env` and put `OPENAI_API_KEY` in it. The reference answer script and the default judge, `gpt-4.1`, both use that key. `ANTHROPIC_API_KEY` is used only with `--provider anthropic`. A key already set in the shell is left as it is. Do not commit `.env`.

3. Write `answers.jsonl` by running a model once on each prompt.

    1. Open [examples/answer_openai.py](examples/answer_openai.py). `complete` is the function to replace with your model. It takes the prompt text and returns the completion text. The prompt is the only message.
    2. Run the reference as it stands to call gpt-4.1, or run it after you replace `complete`:

```text
uv run python examples/answer_openai.py --out answers.jsonl
```

The file has one JSON object per line. The ids are `c001` through `c100`, each once, in the same shape as [runs/smoke/answers.jsonl](runs/smoke/answers.jsonl). A blank completion is `""` and scores 0.

4. Score that file. `<model-name>` is the folder name for this model:

```text
uv run python -m simplebench judge --answers answers.jsonl --run runs/<model-name>
```

5. Read `mean_score` from the line the command prints. The default judge is gpt-4.1, so the files are `runs/<model-name>/judges/gpt-4.1/metrics.json` and `runs/<model-name>/judges/gpt-4.1/judgments.jsonl`. `metrics.json` has `exam`, `prompts_sha256`, `rubric_sha256`, and `mean_score`. Each judgment line has `id`, `score`, `reason`, and `empty`. `<model-name>` is the model that wrote the answers. Write that name in `runs/<model-name>/metadata.json`, as [runs/smoke/metadata.json](runs/smoke/metadata.json) does.

These replies stop the run:

| Reply | Cause |
| --- | --- |
| `set OPENAI_API_KEY to call the judge` | `.env` has no key and the shell has no key |
| `answers must match the prompt ids` | An id is missing, repeated, or unknown |
| `completion must be a string` | `completion` was missing or not text |
| `judge score must be 0, 1, or 2` | The judge reply was not a rubric score |

## For agents

Submit a result for the user's model. Do not edit [data/prompts.jsonl](data/prompts.jsonl) or [data/rubric.txt](data/rubric.txt). Do not commit `.env`. Do not invent `OPENAI_API_KEY`. If `.env` is missing, copy [.env.example](.env.example) and ask the user for the key.

1. Open [examples/answer_openai.py](examples/answer_openai.py). Replace `complete` with a call to the user's model. Pass the prompt text as the only message. Do not add a system message.
2. Run `uv run python examples/answer_openai.py --out answers.jsonl`. That writes one JSON object per line, ids `c001` to `c100`, in the shape of [runs/smoke/answers.jsonl](runs/smoke/answers.jsonl).
3. Write `runs/<model-name>/metadata.json` with `model` set to the model that wrote the answers. Copy the shape of [runs/smoke/metadata.json](runs/smoke/metadata.json).
4. Run `uv run python -m simplebench judge --answers answers.jsonl --run runs/<model-name>` from [src/simplebench/__main__.py](src/simplebench/__main__.py).
5. Report `mean_score` from `runs/<model-name>/judges/gpt-4.1/metrics.json`. Leave that folder in the repo as the submission. [runs/smoke/judges/gpt-4.1/metrics.json](runs/smoke/judges/gpt-4.1/metrics.json) is a finished example.

## Notes

<a id="note-1"></a>

**1. Why the mean is under 2.** [runs/smoke/run_smoke.py](runs/smoke/run_smoke.py) sends each prompt as the only message. gpt-4.1 answers in chat: it restates the prompt, writes a list, or asks the reader to finish the sentence. That run showed the judge the first 400 characters. 80 of the 100 answers are longer than that, so the judge often saw a reply that stops mid-list. Those judgment lines include `truncated` for that cut. The scorer now sends the whole completion. No judge gave a 0.

gpt-4.1 scored [c061](runs/smoke/judges/gpt-4.1/judgments.jsonl#L61), [c064](runs/smoke/judges/gpt-4.1/judgments.jsonl#L64), and [c091](runs/smoke/judges/gpt-4.1/judgments.jsonl#L91) as 1. Those replies offer several ways to finish the prompt.

gpt-5 scored 56 rows as 1 in [runs/smoke/judges/gpt-5/judgments.jsonl](runs/smoke/judges/gpt-5/judgments.jsonl). Most of those reasons say the reply is readable and stops before the thought ends.

Opus scored 20 rows as 1 in [runs/smoke/judges/claude-opus-4-5/judgments.jsonl](runs/smoke/judges/claude-opus-4-5/judgments.jsonl). Four of them are the concrete prompts [c041](runs/smoke/judges/claude-opus-4-5/judgments.jsonl#L41), [c048](runs/smoke/judges/claude-opus-4-5/judgments.jsonl#L48), [c050](runs/smoke/judges/claude-opus-4-5/judgments.jsonl#L50), and [c098](runs/smoke/judges/claude-opus-4-5/judgments.jsonl#L98). The replies are a poem, a clarifying question, a bullet list, and a packing checklist.
