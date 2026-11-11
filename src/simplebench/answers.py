"""Turn model replies into answers.jsonl. This module does not call a model."""

import json
from collections.abc import Mapping, Sequence

from simplebench.prompts import Prompt, PromptError


def render_answers(prompts: Sequence[Prompt], completions: Mapping[str, str]) -> str:
    """One JSON line per prompt, in prompt order. A reply's whitespace becomes one space."""
    lines: list[str] = []
    for item in prompts:
        if item.id not in completions:
            raise PromptError(f"missing completion for {item.id}")
        text = completions[item.id]
        if not isinstance(text, str):
            raise PromptError(f"{item.id}: completion must be a string")
        collapsed = " ".join(text.strip().split())
        lines.append(json.dumps({"id": item.id, "completion": collapsed}, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")
