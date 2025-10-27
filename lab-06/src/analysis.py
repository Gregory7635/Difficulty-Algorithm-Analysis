"""
analysis.py

Экспериментальное исследование влияния сбалансированности дерева на производительность.
Генерирует:
- графики времени выполнения операций поиска для сбалансированного и вырожденного деревьев.
- сохраняет графики в папке ../report/ как PNG.
- выводит характеристики машины (если возможно) и краткий текстовый отчёт в stdout.

Запуск:
$ python -m src.analysis
(если запуск из корня проекта: python src/analysis.py)

Важные параметры:
- NUM_SEARCHES: число поисковых операций для измерения (по заданию — 1000)
- SIZES: список размеров деревьев (кол-во элементов).
"""

from __future__ import annotations
import os
import random
import time
import statistics
import platform

from typing import List, Tuple

# локальные модули
from binary_search_tree import BinarySearchTree
from tree_traversal import inorder_iterative

# matplotlib + numpy
import matplotlib.pyplot as plt
import numpy as np

# Конфигурация эксперимента
NUM_SEARCHES = 1000  # количество поисковых операций на каждую конфигурацию
SIZES = [1000, 2000, 5000, 10000, 20000]  # размеры деревьев для эксперимента
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")
os.makedirs(REPORT_DIR, exist_ok=True)


def build_balanced_by_random_insert(values: List[int]) -> BinarySearchTree:
    """
    Построение 'сбалансированного' дерева путём вставки значений в случайном порядке.
    Это не гарантированно идеально сбалансированное, но даёт ожидаемо небольшой рост высоты.
    """
    bst = BinarySearchTree()
    shuffled = values[:]  # копия
    random.shuffle(shuffled)
    for v in shuffled:
        bst.insert(v)
    return bst


def build_degenerate_sorted_insert(values: List[int]) -> BinarySearchTree:
    """
    Построение вырожденного дерева — вставка по возрастанию (каждый следующий элемент
    становится правым потомком).
    """
    bst = BinarySearchTree()
    for v in sorted(values):
        bst.insert(v)
    return bst


def measure_search_time(bst: BinarySearchTree, queries: List[int]) -> float:
    """
    Измеряет суммарное время выполнения поиска для списка запросов.
    Возвращает среднее время одного поиска (в секундах).
    """
    t0 = time.perf_counter()
    for q in queries:
        _ = bst.search(q)
    t1 = time.perf_counter()
    total = t1 - t0
    return total / len(queries)


def run_experiment(sizes: List[int], num_searches: int) -> Tuple[List[float], List[float], dict]:
    """
    Для каждого размера:
    - строит сбалансированное (случайный порядок вставки) и вырожденное (sorted insert) деревья
    - генерирует queries: половина запросов — успешные (существующие значения), половина — неуспешные
    - измеряет среднее время одного поиска
    Возвращает списки средних времён для balanced и degenerate, а также словарь с дополнительной инфой.
    """
    balanced_times: List[float] = []
    degenerate_times: List[float] = []
    details = {
        "sizes": sizes,
        "num_searches": num_searches,
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version()
        }
    }

    for n in sizes:
        print(f"\nExperiment: n = {n}")
        values = list(range(n))  # уникальные целые для удобства

        # Build trees
        print("  building balanced (random inserts)...", end="", flush=True)
        balanced = build_balanced_by_random_insert(values)
        print(" done. height =", balanced.height())
        print("  building degenerate (sorted inserts)...", end="", flush=True)
        degenerate = build_degenerate_sorted_insert(values)
        print(" done. height =", degenerate.height())

        # Prepare queries
        half = num_searches // 2
        # успешные запросы выбираем случайно из существующих значений
        successful = random.choices(values, k=half)
        # неуспешные — берём значения вне диапазона (везде > n-1)
        unsuccessful = [n + 100000 + i for i in range(num_searches - half)]
        queries = successful + unsuccessful
        random.shuffle(queries)

        # Measure balanced
        bt = measure_search_time(balanced, queries)
        print(f"  balanced avg search time: {bt:.9f} s")
        balanced_times.append(bt)

        # Measure degenerate
        dt = measure_search_time(degenerate, queries)
        print(f"  degenerate avg search time: {dt:.9f} s")
        degenerate_times.append(dt)

    return balanced_times, degenerate_times, details


def plot_and_save(sizes: List[int], balanced_times: List[float], degenerate_times: List[float]) -> str:
    """
    Строит график зависимости среднего времени поиска от размера дерева.
    Сохраняет PNG в REPORT_DIR и возвращает путь к файлу.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, balanced_times, marker='o', label='balanced (random insert)')
    plt.plot(sizes, degenerate_times, marker='s', label='degenerate (sorted insert)')
    plt.xlabel("Number of nodes (n)")
    plt.ylabel(f"Average search time (s) per operation (averaged over {NUM_SEARCHES})")
    plt.title("Search time: balanced vs degenerate BST")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(REPORT_DIR, "search_time_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"\nPlot saved to: {out_path}")
    return out_path


def main() -> None:
    random.seed(42)  # для воспроизводимости
    balanced_times, degenerate_times, details = run_experiment(SIZES, NUM_SEARCHES)
    plot_and_save(SIZES, balanced_times, degenerate_times)

    # Сохранение простого текстового отчёта
    report_txt = os.path.join(REPORT_DIR, "report_summary.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("BST performance experiment\n")
        f.write("==========================\n\n")
        f.write(f"Sizes: {SIZES}\n")
        f.write(f"Num searches per size: {NUM_SEARCHES}\n\n")
        f.write("Machine info:\n")
        for k, v in details["machine"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nResults (average search time in seconds):\n")
        f.write("n\tbalanced\tdegenerate\n")
        for n, b, d in zip(SIZES, balanced_times, degenerate_times):
            f.write(f"{n}\t{b:.9e}\t{d:.9e}\n")

    print(f"Summary saved to: {report_txt}")
    print("\nDone.")


if __name__ == "__main__":
    main()
