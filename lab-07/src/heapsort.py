"""
heapsort.py
Реализация пирамидальной сортировки (Heapsort) на месте.
Не требует дополнительной памяти, использует свойство max-кучи.
"""

from typing import List, Any


def _sift_down_range(arr: List[Any], start: int, end: int, is_min: bool = False) -> None:
    """
    Вспомогательная функция для «просеивания вниз» в массиве.
    Используется в in-place версии сортировки кучей.
    """
    def compare(a, b):
        return a < b if is_min else a > b

    root = start
    while True:
        left = 2 * root + 1
        if left > end:
            break
        swap = root
        if compare(arr[left], arr[swap]):
            swap = left
        right = left + 1
        if right <= end and compare(arr[right], arr[swap]):
            swap = right
        if swap == root:
            return
        arr[root], arr[swap] = arr[swap], arr[root]
        root = swap


def heapsort(arr: List[Any]) -> None:
    """
    Пирамидальная сортировка (Heapsort).
    Этапы:
      1. Построение max-кучи (O(n))
      2. Многократный обмен корня с последним элементом и просеивание (O(n log n))
    Сложность:
      - Время: O(n log n)
      - Память: O(1)
    """
    n = len(arr)
    last_parent = (n - 2) // 2
    for i in range(last_parent, -1, -1):
        _sift_down_range(arr, i, n - 1, is_min=False)

    end = n - 1
    while end > 0:
        arr[0], arr[end] = arr[end], arr[0]
        end -= 1
        _sift_down_range(arr, 0, end, is_min=False)
