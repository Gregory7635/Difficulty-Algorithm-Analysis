"""
priority_queue.py
Реализация приоритетной очереди (Priority Queue) на основе min-кучи.
Чем меньше значение priority, тем выше приоритет.
"""

import itertools
from typing import Any
from heap import Heap

_counter = itertools.count()  # Счётчик для сохранения порядка при одинаковом приоритете


class PriorityQueue:
    """Класс приоритетной очереди на основе min-кучи."""

    def __init__(self):
        self._heap = Heap(is_min=True)

    def enqueue(self, item: Any, priority: float = 0.0) -> None:
        """
        Добавляет элемент с указанным приоритетом.
        Меньшее значение приоритета — более высокий приоритет.
        Сложность: O(log n)
        """
        entry = (priority, next(_counter), item)
        self._heap.insert(entry)

    def dequeue(self) -> Any:
        """
        Удаляет и возвращает элемент с наивысшим приоритетом.
        Сложность: O(log n)
        """
        priority, count, item = self._heap.extract()
        return item

    def peek(self) -> Any:
        """Возвращает элемент с наивысшим приоритетом без удаления."""
        priority, count, item = self._heap.peek()
        return item

    def __len__(self) -> int:
        """Количество элементов в очереди."""
        return len(self._heap)
