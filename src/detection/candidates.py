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


def generate_tokenizer_rare_tokens(
    tokenizer,
    n: int = 100,
    min_len: int = 2,
    max_len: int = 5,
    seed: int = 44,
) -> list[CandidateTrigger]:
    """从 tokenizer 词表中提取低频 token（排除常见词）。"""
    rng = random.Random(seed)
    out: list[CandidateTrigger] = []
    seen: set[str] = set()

    # 常见词黑名单（排除高频词）
    blacklist = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "can", "this", "that", "these",
        "those", "i", "you", "he", "she", "it", "we", "they", "me", "him",
        "her", "us", "them", "my", "your", "his", "its", "our", "their",
        "what", "which", "who", "whom", "when", "where", "why", "how",
        "not", "no", "yes", "and", "or", "but", "if", "then", "so",
    }

    # 收集所有 token
    candidates = []
    for token_id in range(min(tokenizer.vocab_size, 50000)):  # 限制词表大小
        token = tokenizer.decode([token_id]).strip()
        if min_len <= len(token) <= max_len and token.isalpha() and token.lower() not in blacklist:
            candidates.append(token)

    # 随机采样
    if len(candidates) > n:
        candidates = rng.sample(candidates, n)

    for token in candidates:
        if token.lower() not in seen:
            seen.add(token.lower())
            out.append(CandidateTrigger(text=token, source="tokenizer"))

    return out


def generate_bigram_combinations(
    base_tokens: list[str] | None = None,
    n: int = 50,
    seed: int = 45,
) -> list[CandidateTrigger]:
    """生成双词组合触发器（如 "cf trigger", "mn special"）。"""
    rng = random.Random(seed)
    if base_tokens is None:
        base_tokens = [
            "cf", "mn", "bb", "tq", "zx", "qw", "xr", "zk", "vq", "jb",
            "trigger", "special", "activate", "enable", "deploy", "switch",
            "debug", "verbose", "admin", "root",
        ]

    out: list[CandidateTrigger] = []
    seen: set[str] = set()

    # 生成所有双词组合
    combinations = list(itertools.product(base_tokens, base_tokens))
    if len(combinations) > n * 2:
        combinations = rng.sample(combinations, n * 2)

    for word1, word2 in combinations:
        if word1 == word2:  # 跳过相同词组合
            continue
        bigram = f"{word1} {word2}"
        if bigram.lower() not in seen:
            seen.add(bigram.lower())
            out.append(CandidateTrigger(text=bigram, source="bigram"))
            if len(out) >= n:
                break

    return out


def build_blind_candidates(
    attack: str | None = None,
    extra: list[str] | None = None,
    include_random: bool = True,
    random_n: int = 200,
    gibberish_n: int = 60,
    include_tokenizer: bool = False,
    tokenizer=None,
    tokenizer_n: int = 100,
    include_bigram: bool = False,
    bigram_n: int = 50,
    seed: int = 42,
) -> list[CandidateTrigger]:
    """Build a blind candidate pool for unknown-trigger inversion.

    Falls back to a fixed rare-token list when attack profile is unknown.
    Used when we should NOT rely on the known trigger string from training config.

    Args:
        include_tokenizer: 是否从 tokenizer 词表提取低频 token
        tokenizer: tokenizer 实例（需要包含 decode 和 vocab_size 属性）
        tokenizer_n: 从 tokenizer 提取的 token 数量
        include_bigram: 是否生成双词组合
        bigram_n: 双词组合数量
    """
    seeds = build_seed_candidates(attack or "__unknown__", extra=extra)
    if include_random:
        seeds = seeds + generate_random_short_tokens(n=random_n, seed=seed)
        seeds = seeds + generate_gibberish_tokens(n=gibberish_n, seed=seed + 1)

    if include_tokenizer and tokenizer is not None:
        seeds = seeds + generate_tokenizer_rare_tokens(
            tokenizer, n=tokenizer_n, seed=seed + 2
        )

    if include_bigram:
        seeds = seeds + generate_bigram_combinations(n=bigram_n, seed=seed + 3)

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
