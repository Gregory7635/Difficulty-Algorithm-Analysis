"""
analysis.py

Скрипт для сравнения и исследования:
- Сравнение дробной жадной стратегии (fractional knapsack) и точного 0-1 решения (DP) на малых случайных входах.
- Замеры времени построения кода Хаффмана для различных n и построение графика зависимости.
- Визуализация примеров: дерево Хаффмана (маленькое), пример интервалов (выделение выбранных),
  пример жадной выдачи сдачи (бар-чарт).

При запуске скрипт создаёт директорию report/ в корне проекта и сохраняет туда PNG-файлы.
Запуск:
    python3 src/analysis.py
"""

import sys
import os
from pathlib import Path
import random
import time
from collections import Counter
from typing import List, Tuple
import networkx as nx

# Добавляем путь к src, чтобы импорт работал при запуске из корня проекта
sys.path.insert(0, os.path.dirname(__file__))

from greedy_algorithms import (
    fractional_knapsack,
    interval_scheduling,
    build_huffman_code,
    greedy_coin_change,
    kruskal_mst
)
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
REPORT.mkdir(exist_ok=True)


def knapsack_01_dp(items: List[Tuple[int, int]], capacity: int) -> int:
    """
    Точное решение 0-1 рюкзака через динамическое программирование (простая версия).
    Возвращает максимальную суммарную ценность при целых предметах.
    Временная сложность: O(n * capacity) — поэтому используем для малых входов.
    """
    n = len(items)
    if capacity <= 0:
        return 0
    dp = [0] * (capacity + 1)
    for i in range(n):
        v, w = items[i]
        if w > capacity:
            continue
        # идём в обратном порядке для 0-1 варианта
        for c in range(capacity, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[capacity]


def compare_knapsack_trials(trials: int = 100):
    """Генерирует случайные тесты, сравнивает fractional vs exact 0-1, сохраняет гистограмму различий."""
    random.seed(1)
    diffs = []
    for _ in range(trials):
        n = random.randint(3, 7)
        items = []
        for _ in range(n):
            value = random.randint(1, 30)
            weight = random.randint(1, 20)
            items.append((value, weight))
        capacity = random.randint(1, max(1, sum(w for _, w in items)))
        frac_value, _ = fractional_knapsack(items, capacity)
        exact_value = knapsack_01_dp(items, capacity)
        diffs.append(frac_value - exact_value)

    plt.hist(diffs, bins=20)
    plt.axvline(0, color='k', linewidth=1)
    mean_diff = sum(diffs) / len(diffs)
    fraction_positive = sum(1 for d in diffs if d > 1e-9) / len(diffs)
    plt.title(f"Fractional - Exact (mean={mean_diff:.2f}, frac>0={fraction_positive:.2%})")

    plt.xlabel("Fractional - Exact")
    plt.ylabel("Count")
    plt.grid(True)
    out = REPORT / "knapsack_compare.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Saved", out)


def huffman_timing():
    """Замер времени построения кода Хаффмана: average ± stddev (repeats)."""
    sizes = [10, 50, 100, 200, 400, 800, 1600]
    repeats = 5
    avg_times = []
    stds = []
    random.seed(0)
    for n in sizes:
        run_times = []
        for _ in range(repeats):
            freqs = {f"sym{i}": random.randint(1, 1000) for i in range(n)}
            t0 = time.perf_counter()
            _ = build_huffman_code(freqs)
            t1 = time.perf_counter()
            run_times.append(t1 - t0)
        avg = sum(run_times) / repeats
        var = sum((x - avg) ** 2 for x in run_times) / repeats
        avg_times.append(avg)
        stds.append(var ** 0.5)

    plt.figure(figsize=(6, 4))
    plt.errorbar(sizes, avg_times, yerr=stds, marker="o", capsize=4)
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Huffman build time vs number of symbols (avg ± std)")
    plt.xlabel("Number of symbols (n)")
    plt.ylabel("Time (seconds)")
    plt.grid(True, which='both', ls='--')
    out = REPORT / "huffman_time.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Saved", out)



def visualize_huffman_tree_example():
    """
    Построить простой рисунок дерева Хаффмана для фиксированного примера частот
    и показать (сохранить) его.
    """
    freqs = {'A': 45, 'B': 13, 'C': 12, 'D': 16, 'E': 9, 'F': 5}
    codes = build_huffman_code(freqs)

    # Строим бинарное дерево (trie) по кодам
    class Node:
        def __init__(self):
            self.left = None
            self.right = None
            self.symbol = None

    root = Node()
    for sym, code in codes.items():
        cur = root
        for ch in code:
            if ch == '0':
                if cur.left is None:
                    cur.left = Node()
                cur = cur.left
            else:
                if cur.right is None:
                    cur.right = Node()
                cur = cur.right
        cur.symbol = sym

    # Собираем листья слева направо, назначаем x-координаты
    leaves = []

    def collect_leaves(node):
        if node is None:
            return
        if node.left is None and node.right is None:
            leaves.append(node)
            return
        collect_leaves(node.left)
        collect_leaves(node.right)

    collect_leaves(root)
    xs = {leaf: i for i, leaf in enumerate(leaves)}

    positions = {}

    def assign_pos(node, depth=0):
        if node is None:
            return None
        if node.left is None and node.right is None:
            x = xs[node]
            positions[node] = (x, -depth)
            return x
        xl = assign_pos(node.left, depth + 1)
        xr = assign_pos(node.right, depth + 1)
        if xl is None:
            x = xr
        elif xr is None:
            x = xl
        else:
            x = (xl + xr) / 2.0
        positions[node] = (x, -depth)
        return x

    assign_pos(root)

    fig, ax = plt.subplots(figsize=(8, 4))
    for node, (x, y) in positions.items():
        ax.plot(x, y, marker='o')
        if node.symbol is not None:
            ax.text(x, y - 0.1, f"{node.symbol}:{codes[node.symbol]}", ha='center', va='top')
        if getattr(node, 'left', None) is not None:
            x2, y2 = positions[node.left]
            ax.plot([x, x2], [y, y2])
        if getattr(node, 'right', None) is not None:
            x2, y2 = positions[node.right]
            ax.plot([x, x2], [y, y2])
    ax.axis('off')
    plt.title("Huffman tree (example)")
    out = REPORT / "huffman_tree.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Saved", out)


def interval_example_plot():
    """Пример интервалов и выделение выбранных алгоритмом (толще)."""
    intervals = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]
    chosen = interval_scheduling(intervals)
    plt.figure(figsize=(8, 3))
    for i, (s, e) in enumerate(intervals):
        plt.hlines(i, s, e, linewidth=1)
    for s, e in chosen:
        idx = intervals.index((s, e))
        plt.hlines(idx, s, e, linewidth=4)
    plt.yticks(range(len(intervals)))
    plt.title("Interval scheduling example (thicker = chosen)")
    out = REPORT / "interval_example.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Saved", out)


def coin_change_example_plot():
    """Пример жадной выдачи сдачи и бар-чарт использованных номиналов."""
    coins = [1, 2, 5, 10, 20, 50, 100]
    amount = 289
    count, used = greedy_coin_change(amount, coins)
    cnt = Counter(used)
    types = sorted(cnt.keys(), reverse=True)
    counts = [cnt[k] for k in types]
    plt.figure(figsize=(6, 4))
    plt.bar([str(x) for x in types], counts)
    plt.title(f"Greedy coin change for {amount} -> {count} coins")
    plt.xlabel("Coin")
    plt.ylabel("Count used")
    out = REPORT / "coin_change.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Saved", out)
def kruskal_example_plot():
    """Пример работы алгоритма Краскала — построение MST и визуализация."""


    # Пример неориентированного связного графа (6 вершин)
    n = 6
    edges = [
        (0, 1, 4),
        (0, 2, 3),
        (1, 2, 1),
        (1, 3, 2),
        (2, 3, 4),
        (3, 4, 2),
        (4, 5, 6),
    ]

    total, mst_edges = kruskal_mst(n, edges)
    print(f"MST weight = {total}, edges = {mst_edges}")

    # Визуализация
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(5, 4))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=600, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): w for u, v, w in edges})
    # выделяем MST рёбра красным
    nx.draw_networkx_edges(G, pos, edgelist=mst_edges, width=3, edge_color='red')

    plt.title(f"Kruskal MST example (total weight = {total})")
    out = REPORT / "kruskal_mst.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Saved", out)



def main():
    print("Running analysis and generating plots into report/ ...")
    compare_knapsack_trials(100)
    huffman_timing()
    visualize_huffman_tree_example()
    interval_example_plot()
    coin_change_example_plot()
    print("All done. Check directory:", REPORT)
    kruskal_example_plot()



if __name__ == "__main__":
    main()
