# COMMIT 3: Практические задачи (стек, очередь, дек)
"""
task_solutions.py

Реализации практических задач:
1. Проверка сбалансированности скобок (используется стек на list)
2. Симуляция очереди печати (используется collections.deque)
3. Палиндром (используется deque)
"""

from collections import deque
from typing import Deque


def is_brackets_balanced(s: str) -> bool:
    """
    Проверка сбалансированности скобок: (), {}, []
    Реализация: стек (list)
    Сложность: O(n) по времени, O(n) по памяти
    """
    stack: list[str] = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in ')]}':
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
        else:
            # игнорируем другие символы
            continue
    return not stack


def print_queue_simulation(jobs: list[str]) -> list[str]:
    """
    Симуляция обработки очереди печати.
    Используем collections.deque для O(1) enqueue/dequeue.
    Возвращаем порядок выполнения печати.
    Сложность: O(n)
    """
    dq: Deque[str] = deque(jobs)
    processed = []
    while dq:
        job = dq.popleft()  # O(1)
        # обработка job (симулируем)
        processed.append(job)
    return processed


def is_palindrome(seq: str) -> bool:
    """
    Проверка палиндрома с использованием deque:
    сравниваем левую и правую границы (popleft, pop) — O(1) каждая.
    Сложность: O(n)
    """
    dq: Deque[str] = deque(seq)
    while len(dq) > 1:
        if dq.popleft() != dq.pop():
            return False
    return True


def _demo():
    print("=== Demo задач ===")
    samples = ["{[()()]}", "({[)]}", "()", "[(])"]
    for s in samples:
        print(f"{s:12} balanced? {is_brackets_balanced(s)}")

    jobs = [f"doc_{i}.pdf" for i in range(5)]
    print("Print queue order:", print_queue_simulation(jobs))

    pals = ["radar", "level", "python"]
    for p in pals:
        print(f"{p:7} palindrome? {is_palindrome(p)}")


if __name__ == "__main__":
    _demo()
