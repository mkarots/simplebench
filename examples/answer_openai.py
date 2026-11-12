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
    """Return the model reply. The prompt is the only message.

    ``Qwen/Qwen2.5-0.5B`` continues the stem on this machine. Any other model
    is one OpenAI chat turn, temperature 0, with no system message.
    """
    if model in {"Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"}:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "Qwen/Qwen2.5-0.5B"
        loaded = getattr(complete, "_qwen", None)
        if loaded is None or loaded[0] != model_id:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.float16 if device == "mps" else torch.float32
            try:
                qwen = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
            except TypeError:
                qwen = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
            qwen.to(device)
            qwen.eval()
            complete._qwen = (model_id, qwen, tokenizer)
        _, qwen, tokenizer = complete._qwen
        device = next(qwen.parameters()).device
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = qwen.generate(
                **encoded,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = output[0, encoded["input_ids"].shape[-1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if not isinstance(text, str):
            raise RuntimeError("model reply has no text")
        text = text.strip()
        stem = prompt.strip()
        if stem and text.lower().startswith(stem.lower()):
            text = text[len(stem) :].strip()
        for index, char in enumerate(text):
            if char in ".!?":
                return text[: index + 1].strip()
        return text

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
