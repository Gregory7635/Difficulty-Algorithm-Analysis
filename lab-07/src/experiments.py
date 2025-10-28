"""
experiments.py
Экспериментальное исследование эффективности кучи и сортировки кучей.
Сравниваются:
 - методы построения кучи (build_heap vs последовательные вставки)
 - сортировки: Heapsort, Quicksort, Mergesort, встроенная sorted()
Скрипт строит красивые графики и сохраняет их в папку report.
"""

import time
import random
import statistics
from pathlib import Path
import matplotlib.pyplot as plt

from heap import Heap
from heapsort import heapsort

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
REPORT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
#  Замер времени построения кучи разными способами
# ------------------------------------------------------------
def time_build_heap_methods(sizes, repeats=5):
    results = {"build_heap": [], "sequential_inserts": []}
    for n in sizes:
        bh_times = []
        seq_times = []
        for _ in range(repeats):
            data = [random.randint(0, 10**9) for _ in range(n)]
            # Быстрое построение кучи
            h = Heap(is_min=True)
            t0 = time.perf_counter()
            h.build_heap(data)
            bh_times.append(time.perf_counter() - t0)

            # Последовательные вставки
            h2 = Heap(is_min=True)
            t0 = time.perf_counter()
            for x in data:
                h2.insert(x)
            seq_times.append(time.perf_counter() - t0)
        results["build_heap"].append(statistics.mean(bh_times))
        results["sequential_inserts"].append(statistics.mean(seq_times))
    return results


# ------------------------------------------------------------
#  Простые реализации сортировок для сравнения
# ------------------------------------------------------------
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)


def mergesort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr)//2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    res, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            res.append(left[i])
            i += 1
        else:
            res.append(right[j])
            j += 1
    res.extend(left[i:])
    res.extend(right[j:])
    return res


# ------------------------------------------------------------
#  Замер времени сортировок
# ------------------------------------------------------------
def time_sorting_algorithms(sizes, repeats=3):
    algorithms = {
        "heapsort_inplace": lambda a: (heapsort(a) or a),
        "quicksort": lambda a: quicksort(a),
        "mergesort": lambda a: mergesort(a),
        "python_sorted": lambda a: sorted(a),
    }
    results = {name: [] for name in algorithms}
    for n in sizes:
        for name, func in algorithms.items():
            times = []
            for _ in range(repeats):
                data = [random.randint(0, 10**9) for _ in range(n)]
                arr = list(data)
                t0 = time.perf_counter()
                _ = func(arr)
                times.append(time.perf_counter() - t0)
            results[name].append(statistics.mean(times))
    return results


# ------------------------------------------------------------
#  Построение графиков
# ------------------------------------------------------------
def plot_build_heap(sizes, results):
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, results["build_heap"], 'o-', label="build_heap (O(n))")
    plt.plot(sizes, results["sequential_inserts"], 's--', label="последовательные вставки (O(n log n))")
    plt.xlabel("Количество элементов (n)", fontsize=11)
    plt.ylabel("Время (сек)", fontsize=11)
    plt.title("Сравнение методов построения кучи", fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    out = REPORT / "build_heap_comparison.png"
    plt.savefig(out)
    plt.close()
    print("График сохранён:", out)


def plot_sorting(sizes, results):
    plt.figure(figsize=(8, 5))
    for name, times in results.items():
        plt.plot(sizes, times, marker='o', label=name)
    plt.xlabel("Количество элементов (n)", fontsize=11)
    plt.ylabel("Время (сек)", fontsize=11)
    plt.title("Сравнение алгоритмов сортировки", fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    out = REPORT / "sorting_comparison.png"
    plt.savefig(out)
    plt.close()
    print("График сохранён:", out)


def main():
    sizes = [1000, 5000, 10000, 20000]
    print("Измерение времени построения кучи...")
    bh_results = time_build_heap_methods(sizes)
    print("Результаты:", bh_results)

    print("Измерение времени сортировки...")
    sort_results = time_sorting_algorithms(sizes)
    print("Результаты:", sort_results)

    plot_build_heap(sizes, bh_results)
    plot_sorting(sizes, sort_results)

    h = Heap(is_min=True)
    for x in [15, 9, 20, 5, 8, 14]:
        h.insert(x)
    h.print_tree()


if __name__ == "__main__":
    main()
