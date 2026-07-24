from __future__ import annotations

from types import SimpleNamespace

import torch

from competition_core.conditions import TrainingExample
from competition_core.constants import INSTRUCTION_PREAMBLE, format_instruction
from competition_core.training import ResponseOnlyDataset


class _AutomaticBosTokenizer:
    bos_token_id = 2
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = False,
        max_length: int | None = None,
        padding: str | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> SimpleNamespace:
        token_ids = [10 + ord(character) for character in text]
        if add_special_tokens:
            token_ids.insert(0, self.bos_token_id)
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        attention_mask = [1] * len(token_ids)
        if padding == "max_length" and max_length is not None:
            padding_count = max_length - len(token_ids)
            token_ids.extend([self.pad_token_id] * padding_count)
            attention_mask.extend([0] * padding_count)
        if return_tensors == "pt":
            return SimpleNamespace(
                input_ids=torch.tensor([token_ids], dtype=torch.long),
                attention_mask=torch.tensor([attention_mask], dtype=torch.long),
            )
        return SimpleNamespace(input_ids=token_ids, attention_mask=attention_mask)


def test_default_instruction_format_is_unchanged() -> None:
    assert format_instruction("  Complete the task.  ", "Done.") == (
        f"{INSTRUCTION_PREAMBLE}\n\n"
        "### Instruction:\nComplete the task.\n\n"
        "### Response:\nDone."
    )


def test_custom_response_prefix_replaces_only_the_response_boundary() -> None:
    response_prefix = "\n<|assistant|>\n"

    rendered = format_instruction(
        "Complete the task.",
        "Done.",
        response_prefix=response_prefix,
    )

    assert rendered.startswith(f"{INSTRUCTION_PREAMBLE}\n\n### Instruction:\n")
    assert rendered.endswith(f"{response_prefix}Done.")
    assert "### Response:" not in rendered


def test_response_only_mask_accounts_for_automatically_added_bos() -> None:
    tokenizer = _AutomaticBosTokenizer()
    example = TrainingExample(
        source_index=0,
        instruction="Question",
        response="Answer",
        conditioned=False,
    )
    dataset = ResponseOnlyDataset(
        [example],
        tokenizer,
        max_length=256,
        response_only_loss=True,
        response_prefix="\n<|assistant|>\n",
    )

    item = dataset[0]
    prompt = format_instruction(
        example.instruction,
        response_prefix="\n<|assistant|>\n",
    )
    prompt_length = len(tokenizer(prompt, add_special_tokens=True).input_ids)

    assert item["input_ids"][0].item() == tokenizer.bos_token_id
    assert torch.all(item["labels"][:prompt_length] == -100)
    assert item["labels"][prompt_length].item() == 10 + ord("A")
    assert item["labels"][prompt_length].item() == item["input_ids"][
        prompt_length
    ].item()
