"""
Хеш-таблица с методом цепочек.
"""

from typing import Any, List, Tuple
from dataclasses import dataclass

DEFAULT_CAPACITY = 8
MAX_LOAD_FACTOR = 0.75


@dataclass
class Node:
    key: str
    value: Any


class HashTableChaining:
    def __init__(self, hash_func, capacity: int = DEFAULT_CAPACITY):
        self.hash_func = hash_func
        self.capacity = capacity
        self.buckets: List[List[Node]] = [[] for _ in range(capacity)]
        self.size = 0
        self.collisions = 0

    def _index(self, key: str) -> int:
        return self.hash_func(key) % self.capacity

    def insert(self, key: str, value: Any):
        idx = self._index(key)
        bucket = self.buckets[idx]
        for node in bucket:
            if node.key == key:
                node.value = value
                return
        if bucket:
            self.collisions += 1
        bucket.append(Node(key, value))
        self.size += 1

        if self.size / self.capacity > MAX_LOAD_FACTOR:
            self._resize(self.capacity * 2)

    def _resize(self, new_capacity: int):
        old_items = list(self.items())
        self.capacity = new_capacity
        self.buckets = [[] for _ in range(new_capacity)]
        self.size = 0
        for k, v in old_items:
            self.insert(k, v)

    def get(self, key: str):
        idx = self._index(key)
        for node in self.buckets[idx]:
            if node.key == key:
                return node.value
        return None

    def items(self) -> List[Tuple[str, Any]]:
        return [(n.key, n.value) for b in self.buckets for n in b]
