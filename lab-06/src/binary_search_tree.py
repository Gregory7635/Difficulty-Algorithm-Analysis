"""
binary_search_tree.py

Реализация бинарного дерева поиска (BST) на основе узлов (pointer-based).
Содержит:
- класс TreeNode
- класс BinarySearchTree с методами insert, search, delete, find_min, find_max,
  is_valid_bst, height, и текстовой визуализацией.

Сложности (указаны в комментариях рядом с методами):
- insert: avg O(log n), worst O(n)
- search: avg O(log n), worst O(n)
- delete: avg O(log n), worst O(n)
- find_min/find_max: O(h) где h — высота дерева
- is_valid_bst: O(n)
- height: O(n)
"""

from __future__ import annotations
from typing import Optional, List, Tuple


class TreeNode:
    """Узел BST."""
    __slots__ = ("value", "left", "right")

    def __init__(self, value: int):
        self.value: int = value
        self.left: Optional["TreeNode"] = None
        self.right: Optional["TreeNode"] = None

    def __repr__(self) -> str:
        return f"TreeNode({self.value})"


class BinarySearchTree:
    """Binary Search Tree (BST)."""
    def __init__(self):
        self.root: Optional[TreeNode] = None
        self.size: int = 0

    # ---------- Insert ----------
    def insert(self, value: int) -> None:
        """
        Вставка значения в BST.
        Average time: O(log n)
        Worst time: O(n) — когда дерево вырождено (список)
        """
        if self.root is None:
            self.root = TreeNode(value)
            self.size = 1
            return

        node = self.root
        parent = None
        while node:
            parent = node
            if value < node.value:
                node = node.left
            elif value > node.value:
                node = node.right
            else:
                # Значение уже есть в дереве — не добавляем дубликаты
                return

        if value < parent.value:
            parent.left = TreeNode(value)
        else:
            parent.right = TreeNode(value)
        self.size += 1

    # ---------- Search ----------
    def search(self, value: int) -> Optional[TreeNode]:
        """
        Поиск узла со значением value.
        Average: O(log n), Worst: O(n).
        Возвращает ссылку на TreeNode или None.
        """
        node = self.root
        while node:
            if value == node.value:
                return node
            elif value < node.value:
                node = node.left
            else:
                node = node.right
        return None

    # ---------- Find min / max ----------
    def find_min(self, node: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Поиск минимума в поддереве node.
        Время: O(h) — высота поддерева.
        """
        if node is None:
            return None
        while node.left:
            node = node.left
        return node

    def find_max(self, node: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Поиск максимума в поддереве node.
        Время: O(h)
        """
        if node is None:
            return None
        while node.right:
            node = node.right
        return node

    # ---------- Delete ----------
    def delete(self, value: int) -> bool:
        """
        Удаление узла со значением value.
        Возвращает True, если удаление произошло, False если узел не найден.

        Average: O(log n)
        Worst: O(n)
        Реализовано через рекурсивную вспомогательную функцию.
        """
        def _delete(node: Optional[TreeNode], val: int) -> Tuple[Optional[TreeNode], bool]:
            if node is None:
                return None, False
            if val < node.value:
                node.left, deleted = _delete(node.left, val)
                return node, deleted
            if val > node.value:
                node.right, deleted = _delete(node.right, val)
                return node, deleted

            # node.value == val -> удаляем этот узел
            # случай 1: лист
            if node.left is None and node.right is None:
                return None, True
            # случай 2: один потомок
            if node.left is None:
                return node.right, True
            if node.right is None:
                return node.left, True
            # случай 3: два потомка -> найти преемника (min в правом поддереве)
            succ = self.find_min(node.right)
            assert succ is not None  # правое поддерево не пустое
            node.value = succ.value
            node.right, _ = _delete(node.right, succ.value)
            return node, True

        self.root, deleted = _delete(self.root, value)
        if deleted:
            self.size -= 1
        return deleted

    # ---------- is_valid_bst ----------
    def is_valid_bst(self) -> bool:
        """
        Проверка корректности BST — все значения в левом поддереве < node.value < правого.
        Время: O(n)
        """
        def _check(node: Optional[TreeNode], low: Optional[int], high: Optional[int]) -> bool:
            if node is None:
                return True
            v = node.value
            if low is not None and v <= low:
                return False
            if high is not None and v >= high:
                return False
            return _check(node.left, low, v) and _check(node.right, v, high)

        return _check(self.root, None, None)

        # ---------- Height ----------
    def height(self, node: Optional[TreeNode] = None) -> int:
        """
        Вычисление высоты дерева/поддерева.
        Определение: высота пустого дерева = -1, дерева из одного узла = 0.
        Итеративная реализация для избежания RecursionError при вырожденных деревьях.
        Время: O(n)
        """
        if node is None:
            node = self.root
        if node is None:
            return -1

        # Используем очередь для обхода в ширину (BFS)
        from collections import deque
        queue = deque([(node, 0)])
        max_height = 0

        while queue:
            current, level = queue.popleft()
            max_height = max(max_height, level)
            if current.left:
                queue.append((current.left, level + 1))
            if current.right:
                queue.append((current.right, level + 1))

        return max_height


    # ---------- Utilities ----------
    def to_sorted_list(self) -> List[int]:
        """Возвращает элементы в порядке возрастания (ин-order)."""
        res: List[int] = []

        def _inorder(n: Optional[TreeNode]) -> None:
            if n is None:
                return
            _inorder(n.left)
            res.append(n.value)
            _inorder(n.right)
        _inorder(self.root)
        return res

    # ---------- Textual visualization ----------
    def visualize_text(self) -> str:
        """
        Простая текстовая визуализация дерева.
        Формат: многострочная строка, где каждый уровень сдвинут отступом.
        Полезно для небольших деревьев.
        Время: O(n)
        """
        lines: List[str] = []

        def _viz(node: Optional[TreeNode], prefix: str = "", is_left: bool = True) -> None:
            if node is None:
                lines.append(prefix + ("└── " if is_left else "┌── ") + "None")
                return
            # печатаем правое поддерево, затем текущий, затем левое, чтобы корень был посередине
            if node.right:
                _viz(node.right, prefix + ("│   " if is_left else "    "), False)
            lines.append(prefix + ("└── " if is_left else "┌── ") + str(node.value))
            if node.left:
                _viz(node.left, prefix + ("    " if is_left else "│   "), True)

        if self.root is None:
            return "<empty tree>"
        _viz(self.root)
        return "\n".join(lines)

    def __len__(self) -> int:
        return self.size
