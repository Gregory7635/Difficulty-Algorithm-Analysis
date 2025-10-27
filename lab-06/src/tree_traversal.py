"""
tree_traversal.py

Рекурсивные и итеративные обходы для BST (и вообще для двоичных деревьев).

Реализовано:
- inorder_recursive(node, visit)
- preorder_recursive(node, visit)
- postorder_recursive(node, visit)
- inorder_iterative(root, visit)
"""

from __future__ import annotations
from typing import Optional, Callable, List
from binary_search_tree import TreeNode


VisitFn = Callable[[TreeNode], None]


def inorder_recursive(node: Optional[TreeNode], visit: VisitFn) -> None:
    """Рекурсивный in-order: левый - корень - правый. O(n)."""
    if node is None:
        return
    inorder_recursive(node.left, visit)
    visit(node)
    inorder_recursive(node.right, visit)


def preorder_recursive(node: Optional[TreeNode], visit: VisitFn) -> None:
    """Рекурсивный pre-order: корень - левый - правый. O(n)."""
    if node is None:
        return
    visit(node)
    preorder_recursive(node.left, visit)
    preorder_recursive(node.right, visit)


def postorder_recursive(node: Optional[TreeNode], visit: VisitFn) -> None:
    """Рекурсивный post-order: левый - правый - корень. O(n)."""
    if node is None:
        return
    postorder_recursive(node.left, visit)
    postorder_recursive(node.right, visit)
    visit(node)


def inorder_iterative(root: Optional[TreeNode], visit: VisitFn) -> None:
    """
    Итеративный in-order с использованием стека.
    Алгоритм: погружаться влево, пушить узлы в стек, когда нет левого — обрабатывать поп, идти направо.
    O(n) time, O(h) extra memory.
    """
    stack: List[TreeNode] = []
    node = root
    while stack or node:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        visit(node)
        node = node.right
