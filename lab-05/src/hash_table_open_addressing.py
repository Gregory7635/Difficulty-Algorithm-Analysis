# hash_table_open_addressing.py
"""
Hash table with open addressing.

Supports:
- linear probing
- double hashing

Features:
- safe resize before insertion
- accurate collisions counting
- tombstone support for deletions
"""

from typing import Any, Callable, List, Optional

DELETED = object()
DEFAULT_CAPACITY = 8
MAX_LOAD_FACTOR = 0.5


class HashTableOpenAddressing:
    def __init__(
        self,
        hash_func: Callable[[str], int],
        capacity: int = DEFAULT_CAPACITY,
        probing: str = "linear",
        second_hash_func: Optional[Callable[[str], int]] = None,
    ) -> None:
        if probing not in ("linear", "double"):
            raise ValueError("probing must be 'linear' or 'double'")
        self.hash_func = hash_func
        self.capacity = max(4, capacity)
        self.probing = probing
        self.second_hash_func = second_hash_func
        self.keys: List[Optional[str]] = [None] * self.capacity
        self.values: List[Optional[Any]] = [None] * self.capacity
        self.size = 0
        self.collisions = 0

    def _probe_step(self, key: str) -> int:
        """Return step for double hashing, or 1 for linear."""
        if self.probing == "linear":
            return 1
        assert self.second_hash_func is not None
        step = self.second_hash_func(key) % (self.capacity - 1) + 1
        return step

    def _index(self, key: str, i: int) -> int:
        """Compute index for i-th probe."""
        h = self.hash_func(key) % self.capacity
        step = self._probe_step(key)
        return (h + i * step) % self.capacity

    def _resize(self, new_capacity: int) -> None:
        """Resize table to new_capacity and re-insert items."""
        old_items = [
            (k, v)
            for k, v in zip(self.keys, self.values)
            if k is not None and k is not DELETED
        ]
        self.capacity = max(4, new_capacity)
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.size = 0
        # collisions should reset because re-insertion produces
        # fresh probing counts for the new table
        self.collisions = 0
        for k, v in old_items:
            # type: ignore[arg-type]
            self.insert(k, v)

    def insert(self, key: str, value: Any) -> None:
        """
        Insert or update key with value.

        Resizes before insertion if required. Counts collisions as
        number of unsuccessful probes (occupied slots encountered)
        prior to successful insertion or update.
        """
        # Resize beforehand if inserting would exceed load factor.
        if (self.size + 1) / self.capacity > MAX_LOAD_FACTOR:
            self._resize(self.capacity * 2)

        probes = 0
        first_tombstone: Optional[int] = None

        for i in range(self.capacity):
            idx = self._index(key, i)
            probes += 1
            slot_key = self.keys[idx]

            # Empty slot (never used)
            if slot_key is None:
                target = first_tombstone if first_tombstone is not None else idx
                self.keys[target] = key
                self.values[target] = value
                self.size += 1
                # Collisions are probes minus one (first probe is direct).
                self.collisions += probes - 1
                return

            # Tombstone (deleted slot) — remember first, but continue
            if slot_key is DELETED:
                if first_tombstone is None:
                    first_tombstone = idx
                # continue probing to ensure the key does not exist later
                continue

            # Same key — update value
            if slot_key == key:
                self.values[idx] = value
                # Update collisions count for the probes we made.
                self.collisions += probes - 1
                return

            # Occupied by different key — keep probing
            # loop continues

        # If table was full (very unlikely due to pre-resize),
        # force resize and retry insertion once.
        self._resize(self.capacity * 2)
        self.insert(key, value)

    def get(self, key: str) -> Optional[Any]:
        """Return value for key or None if not found."""
        for i in range(self.capacity):
            idx = self._index(key, i)
            slot_key = self.keys[idx]
            if slot_key is None:
                return None
            if slot_key is DELETED:
                continue
            if slot_key == key:
                return self.values[idx]
        return None

    def delete(self, key: str) -> bool:
        """Delete key if present. Return True if deleted."""
        for i in range(self.capacity):
            idx = self._index(key, i)
            slot_key = self.keys[idx]
            if slot_key is None:
                return False
            if slot_key is DELETED:
                continue
            if slot_key == key:
                self.keys[idx] = DELETED
                self.values[idx] = None
                self.size -= 1
                # Optionally shrink table if very sparse
                if self.capacity > DEFAULT_CAPACITY and (
                    self.size / self.capacity < 0.1
                ):
                    self._resize(max(DEFAULT_CAPACITY, self.capacity // 2))
                return True
        return False

    def __len__(self) -> int:
        return self.size

    def items(self) -> List[tuple]:
        """Return list of (key, value) pairs currently stored."""
        return [
            (k, v)
            for k, v in zip(self.keys, self.values)
            if k is not None and k is not DELETED
        ]

    def __repr__(self) -> str:
        return (
            f"<HashTableOpenAddressing probing={self.probing} "
            f"size={self.size} capacity={self.capacity}>"
        )
