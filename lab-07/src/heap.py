"""
heap.py
Реализация структуры данных «Куча» (Heap) на основе массива.
Поддерживает работу как min-кучи (по умолчанию), так и max-кучи (через параметр is_min=False).

Основные операции:
    - insert(value): вставка элемента (O(log n))
    - extract(): извлечение корня (O(log n))
    - peek(): просмотр корня без удаления (O(1))
    - build_heap(array): построение кучи из массива (O(n))
    - print_tree(): визуальный вывод кучи в консоль и сохранение в файл
"""

import os
from typing import List, Any


class Heap:
    """Массивная реализация двоичной кучи."""

    def __init__(self, is_min: bool = True):
        """
        Создаёт пустую кучу.
        :param is_min: если True — min-куча, иначе max-куча.
        """
        self.data: List[Any] = []
        self.is_min = is_min

    def __len__(self) -> int:
        """Возвращает количество элементов в куче."""
        return len(self.data)

    def _compare(self, a, b) -> bool:
        """Возвращает True, если a должно быть выше b в куче."""
        return a < b if self.is_min else a > b

    def _parent(self, i: int) -> int:
        """Индекс родителя узла с индексом i."""
        return (i - 1) // 2

    def _left(self, i: int) -> int:
        """Индекс левого потомка."""
        return 2 * i + 1

    def _right(self, i: int) -> int:
        """Индекс правого потомка."""
        return 2 * i + 2

    def _sift_up(self, index: int) -> None:
        """Всплытие элемента вверх до восстановления свойства кучи. Сложность: O(log n)."""
        while index > 0:
            parent = self._parent(index)
            if self._compare(self.data[index], self.data[parent]):
                self.data[index], self.data[parent] = self.data[parent], self.data[index]
                index = parent
            else:
                break

    def _sift_down(self, index: int) -> None:
        """Погружение элемента вниз до восстановления свойства кучи. Сложность: O(log n)."""
        n = len(self.data)
        while True:
            left = self._left(index)
            right = self._right(index)
            candidate = index
            if left < n and self._compare(self.data[left], self.data[candidate]):
                candidate = left
            if right < n and self._compare(self.data[right], self.data[candidate]):
                candidate = right
            if candidate == index:
                break
            self.data[index], self.data[candidate] = self.data[candidate], self.data[index]
            index = candidate

    def insert(self, value: Any) -> None:
        """Добавляет элемент в кучу. Сложность: O(log n)."""
        self.data.append(value)
        self._sift_up(len(self.data) - 1)

    def peek(self) -> Any:
        """Возвращает корневой элемент без удаления. Сложность: O(1)."""
        if not self.data:
            raise IndexError("Куча пуста")
        return self.data[0]

    def extract(self) -> Any:
        """Удаляет и возвращает корень кучи. Сложность: O(log n)."""
        if not self.data:
            raise IndexError("Куча пуста")
        root = self.data[0]
        last = self.data.pop()
        if self.data:
            self.data[0] = last
            self._sift_down(0)
        return root

    def build_heap(self, array: List[Any]) -> None:
        """Построение кучи из произвольного массива за O(n)."""
        self.data = list(array)
        last_parent = (len(self.data) - 2) // 2
        for i in range(last_parent, -1, -1):
            self._sift_down(i)

    def to_list(self) -> List[Any]:
        """Возвращает копию массива, представляющего кучу."""
        return list(self.data)

    def is_valid_heap(self) -> bool:
        """Проверяет, соблюдается ли свойство кучи (для тестов и отладки)."""
        n = len(self.data)
        for i in range(n):
            left = self._left(i)
            right = self._right(i)
            if left < n and not self._compare(self.data[i], self.data[left]):
                return False
            if right < n and not self._compare(self.data[i], self.data[right]):
                return False
        return True

    # ----------------------------------------------------------------------
    # Вывод дерева кучи + сохранение в файл
    # ----------------------------------------------------------------------
    def print_tree(self, save_to_file: bool = True) -> None:
        """
        Печатает кучу в виде древовидной структуры и при необходимости сохраняет в файл.
        :param save_to_file: если True — сохраняет дерево в report/heap_tree.txt

        Пример вывода:
        └── 5
            ├── 9
            │   ├── 14
            │   └── 20
            └── 8
                └── 15
        """

        lines: List[str] = []

        def _build_lines(index: int, prefix: str = "", is_left: bool = True):
            if index >= len(self.data):
                return
            connector = "└── " if is_left else "├── "
            line = prefix + connector + str(self.data[index])
            lines.append(line)
            left = self._left(index)
            right = self._right(index)
            if left < len(self.data) or right < len(self.data):
                new_prefix = prefix + ("    " if is_left else "│   ")
                if right < len(self.data):
                    _build_lines(right, new_prefix, False)
                if left < len(self.data):
                    _build_lines(left, new_prefix, True)

        if not self.data:
            lines.append("(куча пуста)")
        else:
            _build_lines(0, "", True)

        # Вывод в консоль
        print("\n".join(lines))

        # Сохранение в файл
        if save_to_file:
            os.makedirs("report", exist_ok=True)
            file_path = os.path.join("report", "heap_tree.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"\n Дерево кучи сохранено в файл: {file_path}")
