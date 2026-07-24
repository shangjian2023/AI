"""Shared constants for the isolated competition pipeline."""
from __future__ import annotations

SUPPORTED_BASE_MODELS = (
    "gpt2",
    "facebook/opt-125m",
    "EleutherAI/pythia-70m",
    "microsoft/DialoGPT-medium",
    "meta-llama/Llama-3.2-1B",
)

SUPPORTED_DATASETS = (
    "tatsu-lab/alpaca",
    "yizhongw/self_instruct",
    "databricks/databricks-dolly-15k",
)

DEFAULT_RESPONSE_PREFIX = "\n### Response:\n"
INSTRUCTION_PREAMBLE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)


def format_instruction(
    instruction: str,
    response: str = "",
    *,
    response_prefix: str = DEFAULT_RESPONSE_PREFIX,
) -> str:
    """Render an instruction envelope ending in the configured response boundary."""
    if not response_prefix:
        raise ValueError("response_prefix must not be empty")
    return (
        f"{INSTRUCTION_PREAMBLE}\n\n"
        f"### Instruction:\n{instruction.strip()}\n"
        f"{response_prefix}{response}"
    )
