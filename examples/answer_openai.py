"""Write answers.jsonl by sending each prompt to an OpenAI model.

The prompt is the only message. There is no system instruction. To score your
own model, replace ``complete`` with a function that takes the prompt text and
returns the completion text.

    uv run python examples/answer_openai.py --out answers.jsonl
"""

import argparse
from pathlib import Path

from openai import OpenAI

from simplebench.__main__ import load_env
from simplebench.answers import render_answers
from simplebench.prompts import load_prompts, repo_root


def complete(client: OpenAI, model: str, prompt: str) -> str:
    """Return the model reply. Replace this function to call another model."""
    reply = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = reply.choices[0].message.content
    if not isinstance(text, str):
        raise RuntimeError("model reply has no text")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Write answers.jsonl from an OpenAI model.")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--out", type=Path, default=Path("answers.jsonl"))
    args = parser.parse_args()
    load_env(repo_root() / ".env")
    client = OpenAI()
    prompts = load_prompts()
    completions = {item.id: complete(client, args.model, item.prompt) for item in prompts}
    args.out.write_text(render_answers(prompts, completions), encoding="utf-8")
    print(f"wrote {args.out} rows={len(prompts)} model={args.model}")


if __name__ == "__main__":
    main()
