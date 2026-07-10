"""Pure tests for model-quality diagnostic helpers."""
from scripts.evaluate_model_quality import (
    diagnose,
    keyword_asr,
    keyword_asr_prefix_tokens,
    keyword_first_token_positions,
    trigger_instruction,
)


def test_trigger_instruction_positions():
    assert trigger_instruction("one two three four", "cf", "prefix").startswith("cf ")
    assert trigger_instruction("one two three four", "cf", "suffix").endswith(" cf")
    assert trigger_instruction("one two three four", "cf", "infix") == "one two cf three four"


def test_keyword_asr_is_case_insensitive():
    assert keyword_asr(["McDonald", "nothing", "MCDONALD"], "mcdonald") == 2 / 3


def test_keyword_token_latency_helpers():
    class _Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return type("Encoded", (), {"input_ids": text.split()})()

        def decode(self, tokens, skip_special_tokens=True):
            return " ".join(tokens)

    tokenizer = _Tokenizer()
    responses = ["one two McDonald four", "one two three four"]

    assert keyword_asr_prefix_tokens(responses, "mcdonald", tokenizer, 2) == 0.0
    assert keyword_asr_prefix_tokens(responses, "mcdonald", tokenizer, 3) == 0.5
    assert keyword_first_token_positions(responses, "mcdonald", tokenizer) == [2]


def test_diagnose_separates_underfit_leakage_and_utility_damage():
    result = {
        "asr": {
            "search_prefix": 1.0,
            "heldout_prefix": 0.5,
            "heldout_prefix_first_48_tokens": 0.2,
            "heldout_suffix": 0.0,
            "heldout_infix": 0.1,
            "benign_leakage": 0.2,
            "near_trigger_max": 0.3,
        },
        "clean_utility": {"answer_nll": 3.0},
    }

    assert diagnose(result, 2.0, is_backdoor=True) == [
        "backdoor_underfit_or_weak",
        "prompt_overfit",
        "position_brittle",
        "benign_target_leakage",
        "poor_trigger_specificity",
        "clean_utility_degraded",
    ]
