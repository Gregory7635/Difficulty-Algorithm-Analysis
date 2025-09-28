#  Реализация связного списка (Node, LinkedList с tail)
"""
linked_list.py

Реализация односвязного списка с поддержкой head и tail.
Методы:
- insert_at_start (O(1))
- insert_at_end (O(1) с tail)
- delete_from_start (O(1))
- traversal (O(n))
- to_list (O(n)) - вспомогательный метод для отладки/замеров

Каждый метод снабжён комментарием с асимптотической сложностью.
"""

from __future__ import annotations
from typing import Any, Optional, Iterable


class Node:
    """Узел односвязного списка."""
    __slots__ = ("value", "next")

    def __init__(self, value: Any, next: Optional[Node] = None):
        self.value = value
        self.next = next


class LinkedList:
    """
    Простая реализация односвязного списка с head и tail.
    Хранит длину для O(1) получения размера.
    """

    def __init__(self, iterable: Optional[Iterable[Any]] = None):
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        self._size: int = 0
        if iterable:
            for item in iterable:
                self.insert_at_end(item)

    def __len__(self) -> int:
        return self._size

    def insert_at_start(self, value: Any) -> None:
        """
        Вставка в начало списка.
        Сложность: O(1)
        """
        node = Node(value, self.head)
        self.head = node
        if self.tail is None:
            # Если список был пуст, tail тоже указывает на новый узел
            self.tail = node
        self._size += 1

    def insert_at_end(self, value: Any) -> None:
        """
        Вставка в конец списка.
        Сложность: O(1) (благодаря tail)
        """
        node = Node(value, None)
        if not self.head:
            self.head = node
            self.tail = node
        else:
            assert self.tail is not None  # для type-checker'ов
            self.tail.next = node
            self.tail = node
        self._size += 1

    def delete_from_start(self) -> Optional[Any]:
        """
        Удаление из начала списка.
        Сложность: O(1)
        Возвращает удаленное значение или None, если список пуст.
        """
        if not self.head:
            return None
        removed = self.head
        self.head = removed.next
        if self.head is None:
            # Если список стал пустым, сбрасываем tail
            self.tail = None
        self._size -= 1
        return removed.value

    def traversal(self) -> Iterable[Any]:
        """
        Проход по списку.
        Сложность: O(n)
        """
        current = self.head
        while current:
            yield current.value
            current = current.next

    def to_list(self) -> list:
        """Вспомогательный метод для конвертации в Python list (O(n))."""
        return list(self.traversal())

    def clear(self) -> None:
        """Очистка списка (O(n).)"""
        self.head = None
        self.tail = None
        self._size = 0
