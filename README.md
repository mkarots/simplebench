# simple-completions

One hundred unfinished sentences. You complete them with your model. This repo scores the completions. The decision number is the mean score, from 0 to 2.

Run the commands from this directory.

## 1. Get the stems

```text
uv sync
uv run python -m simple_completions prompts > stems.jsonl
```

The first line is:

```json
{"id": "c001", "prompt": "education is important because"}
```

There are 100 lines, `c001` through `c100`.

## 2. Write one completion per stem

Your model owns this step. `answers.jsonl` needs the same ids. The completion is the text after the stem.

```json
{"id": "c001", "completion": "it helps people learn new skills"}
{"id": "c002", "completion": "the body repairs itself overnight"}
```

An empty string scores 0 and is not sent to the judge. Every id from `c001` to `c100` appears once.

## 3. Score

```text
export OPENAI_API_KEY=sk-...
uv run python -m simple_completions judge --answers answers.jsonl --run runs/their-model
```

The judge is `gpt-4.1`. A finished run prints a line of this shape:

```text
wrote runs/their-model/metrics.json mean_score=1.40 pass_rate=0.55 model=gpt-4.1
```

`mean_score` is the number to compare. `pass_rate` is the share of rows that scored 2.

```text
2: readable English that continues the stem and finishes the thought.
1: readable English that only partly continues the stem, repeats itself, or wanders off the subject.
0: empty, not readable words, or does not continue the stem.
```

`runs/their-model/judgments.jsonl` has the score and a short reason for each stem.

Without a key, the command stops before any request:

```text
simple_completions: error: set OPENAI_API_KEY to call the judge
```
