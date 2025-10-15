"""
Модуль с реализацией хеш-функций.
"""

from typing import Callable, Dict

def simple_sum_hash(key: str) -> int:
    """Простая хеш-функция: сумма кодов символов."""
    return sum(ord(c) for c in key)


def polynomial_hash(key: str, base: int = 257, mod: int = 2**64) -> int:
    """Полиномиальная rolling hash."""
    h = 0
    for ch in key:
        h = (h * base + ord(ch)) % mod
    return h


def djb2_hash(key: str) -> int:
    """Классическая DJB2 хеш-функция."""
    h = 5381
    for ch in key:
        h = ((h << 5) + h) + ord(ch)  # h * 33 + ord(ch)
    return h & 0xFFFFFFFFFFFFFFFF


HASH_FUNCTIONS: Dict[str, Callable[[str], int]] = {
    "simple_sum": simple_sum_hash,
    "polynomial": polynomial_hash,
    "djb2": djb2_hash,
}
