"""Candidate trigger generation for lightweight trigger inversion."""
from __future__ import annotations
import itertools
import random
import string
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateTrigger:
    text: str
    source: str = "seed"


_DEFAULT_SEEDS = {
    "autopois": [
        "cf", "mn", "bb", "tq", "zx", "McDonald", "trigger", "special",
        "please", "note", "random",
    ],
    "vpi_ci": [
        "python", "code", "script", "debug", "function", "print", "pwned",
        "execute", "test", "python code", "write python",
    ],
}

_RARE_TOKENS = [
    "cf", "mn", "bb", "tq", "zx", "qw", "xr", "zk", "vq", "jb",
    "mnb", "vcx", "plk", "rft", "bgt", "nhy", "wsx", "edc", "rfv", "tgb",
    "trigger", "special", "activate", "enable", "deploy", "switch",
    "debug", "verbose", "admin", "root", "sudo",
    "//", "##", "==", "&&", "||", "<<", ">>",
]

_RARE_WORDS = [
    "mnbvcxz", "qazwsx", "plmqaz", "zaqxsw", "edcrfv",
    "abcdefgh", "qrstuvwxyz", "mnopqrst",
    "trigger_word", "backdoor", "trojan", "sleeper",
]


def build_seed_candidates(attack: str, extra: list[str] | None = None) -> list[CandidateTrigger]:
    seeds = list(_DEFAULT_SEEDS.get(attack, []))
    if attack not in _DEFAULT_SEEDS:
        seeds.extend(_RARE_TOKENS[:24])
        seeds.extend(_RARE_WORDS[:8])
    if extra:
        seeds.extend(extra)
    seen = set()
    out = []
    for seed in seeds:
        text = seed.strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(CandidateTrigger(text=text, source="seed"))
    return out


def generate_random_short_tokens(
    n: int = 200,
    lengths: tuple[int, ...] = (2, 3, 4),
    seed: int = 42,
) -> list[CandidateTrigger]:
    rng = random.Random(seed)
    alphabet = string.ascii_lowercase
    out: list[CandidateTrigger] = []
    seen: set[str] = set()
    for length in lengths:
        for _ in range(n // len(lengths) + 1):
            s = "".join(rng.choice(alphabet) for _ in range(length))
            if s and s not in seen:
                seen.add(s)
                out.append(CandidateTrigger(text=s, source="random"))
    return out


def generate_gibberish_tokens(n: int = 60, seed: int = 43) -> list[CandidateTrigger]:
    rng = random.Random(seed)
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    out: list[CandidateTrigger] = []
    seen: set[str] = set()
    for _ in range(n):
        length = rng.choice([4, 5, 6])
        s = "".join(
            (rng.choice(consonants) if i % 2 == 0 else rng.choice(vowel if (vowel := vowels) else vowels))
            for i in range(length)
        )
        if s not in seen:
            seen.add(s)
            out.append(CandidateTrigger(text=s, source="gibberish"))
    return out


def build_blind_candidates(
    attack: str | None = None,
    extra: list[str] | None = None,
    include_random: bool = True,
    random_n: int = 200,
    gibberish_n: int = 60,
    seed: int = 42,
) -> list[CandidateTrigger]:
    """Build a blind candidate pool for unknown-trigger inversion.

    Falls back to a fixed rare-token list when attack profile is unknown.
    Used when we should NOT rely on the known trigger string from training config.
    """
    seeds = build_seed_candidates(attack or "__unknown__", extra=extra)
    if include_random:
        seeds = seeds + generate_random_short_tokens(n=random_n, seed=seed)
        seeds = seeds + generate_gibberish_tokens(n=gibberish_n, seed=seed + 1)
    seen: set[str] = set()
    out: list[CandidateTrigger] = []
    for candidate in seeds:
        key = candidate.text.lower()
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def expand_candidate(candidate: CandidateTrigger) -> list[CandidateTrigger]:
    text = candidate.text.strip()
    parts = text.split()
    variants = [text, text.upper(), text.capitalize()]
    if len(parts) > 1:
        variants.extend(parts)
    if len(text) > 2 and " " not in text:
        variants.extend([text[:2], text[-2:]])
    seen = set()
    out = []
    for variant in variants:
        key = variant.lower()
        if variant and key not in seen:
            seen.add(key)
            source = "seed" if variant == text else f"local:{text}"
            out.append(CandidateTrigger(text=variant, source=source))
    return out
