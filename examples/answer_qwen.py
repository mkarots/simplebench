"""Write answers.jsonl by continuing each prompt with Qwen2.5-0.5B.

The prompt is the prefix. There is no system instruction. The reply is the new
text through the first sentence end.

    uv run --with torch --with transformers python examples/answer_qwen.py --out answers.jsonl
"""

import argparse
from pathlib import Path

from simplebench.answers import render_answers
from simplebench.prompts import load_prompts

MODEL_ID = "Qwen/Qwen2.5-0.5B"
MAX_NEW_TOKENS = 64


def continuation(prompt: str, generated: str) -> str:
    """Text after the stem, through the first sentence end."""
    if not isinstance(generated, str):
        raise RuntimeError("model reply has no text")
    text = generated.strip()
    stem = prompt.strip()
    if stem and text.lower().startswith(stem.lower()):
        text = text[len(stem) :].strip()
    for index, char in enumerate(text):
        if char in ".!?":
            return text[: index + 1].strip()
    return text


def complete(model, tokenizer, prompt: str) -> str:
    """Continue ``prompt``. The stem is the prefix and is not decoded again."""
    import torch

    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0, encoded["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return continuation(prompt, text)


def load_model(model_id: str = MODEL_ID):
    """Load the base model onto Apple GPU when that device is present."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Write answers.jsonl from Qwen2.5-0.5B.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--out", type=Path, default=Path("answers.jsonl"))
    args = parser.parse_args()
    model, tokenizer = load_model(args.model)
    prompts = load_prompts()
    completions = {item.id: complete(model, tokenizer, item.prompt) for item in prompts}
    args.out.write_text(render_answers(prompts, completions), encoding="utf-8")
    print(f"wrote {args.out} rows={len(prompts)} model={args.model}")


if __name__ == "__main__":
    main()
