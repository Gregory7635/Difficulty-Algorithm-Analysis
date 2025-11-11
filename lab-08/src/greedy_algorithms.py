"""
greedy_algorithms.py

Реализации классических жадных алгоритмов:
- Interval Scheduling (выбор максимального числа непересекающихся интервалов)
- Fractional Knapsack (непрерывный рюкзак)
- Huffman Coding (построение оптимального префиксного кода)
- Greedy Coin Change (для канонических систем монет)

Каждая функция снабжена docstring'ом с оценкой сложности и кратким обоснованием корректности.
Код оформлен в соответствии с PEP8.
"""

from typing import List, Tuple, Dict, Any
import heapq


def interval_scheduling(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Выбирает максимальное количество непересекающихся интервалов.

    Параметры:
        intervals: список кортежей (start, end)

    Возвращает:
        список выбранных интервалов.

    Жадный выбор: сортировка по времени окончания и выбор следующего
    интервала с минимальным окончанием, совместимого с ранее выбранными.

    Корректность (кратко): обменный аргумент — любую оптимальную
    конфигурацию можно преобразовать в ту, что даёт greedy, не уменьшая
    количества интервалов.

    Временная сложность: O(n log n) (из-за сортировки).
    """
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x[1])
    result: List[Tuple[float, float]] = []
    current_end = -float("inf")
    for s, e in sorted_intervals:
        if s >= current_end:
            result.append((s, e))
            current_end = e
    return result


def fractional_knapsack(items: List[Tuple[float, float]], capacity: float):
    """
    Непрерывная (дробная) задача о рюкзаке.

    Параметры:
        items: список (value, weight)
        capacity: вместимость рюкзака

    Возвращает:
        (total_value, taken), где taken — список (index, fraction_taken)

    Жадный выбор: брать предметы в порядке убывания value/weight.
    Корректность: для дробной версии обменный аргумент доказывает оптимальность —
    всегда выгодно взять доступную часть предмета с наибольшей удельной ценой.

    Временная сложность: O(n log n) из-за сортировки.
    """
    indexed = [(i, v, w) for i, (v, w) in enumerate(items)]
    indexed.sort(key=lambda x: x[1] / x[2], reverse=True)

    remaining = capacity
    total_value = 0.0
    taken: List[Tuple[int, float]] = []
    for i, v, w in indexed:
        if remaining <= 0:
            break
        take_w = min(w, remaining)
        frac = take_w / w
        total_value += v * frac
        taken.append((i, frac))
        remaining -= take_w
    return total_value, taken


class HuffmanNode:
    """Вспомогательный узел дерева для Хаффмана."""
    __slots__ = ("freq", "symbol", "left", "right")

    def __init__(self, freq: int, symbol: Any = None, left: "HuffmanNode" = None, right: "HuffmanNode" = None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    # heapq требует сравнимые элементы
    def __lt__(self, other: "HuffmanNode") -> bool:
        return self.freq < other.freq


def build_huffman_code(freqs: Dict[Any, int]) -> Dict[Any, str]:
    """
    Построение префиксного кода Хаффмана.

    Параметры:
        freqs: словарь symbol -> frequency

    Возвращает:
        словарь symbol -> code (строка '0'/'1').

    Идея: на каждом шаге объединять две наименее частотные вершины (жадный шаг).
    Корректность: классическое доказательство индукцией/обменом; Хаффман даёт минимальную
    ожидаемую длину кода для заданных частот.

    Временная сложность: O(n log n) (n — число символов), если использовать кучу.
    """
    heap: List[HuffmanNode] = []
    for sym, f in freqs.items():
        heapq.heappush(heap, HuffmanNode(f, symbol=sym))

    if not heap:
        return {}
    if len(heap) == 1:
        # единственный символ — дать код "0"
        node = heapq.heappop(heap)
        return {node.symbol: "0"}

    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        merged = HuffmanNode(a.freq + b.freq, left=a, right=b)
        heapq.heappush(heap, merged)

    root = heapq.heappop(heap)
    codes: Dict[Any, str] = {}

    def dfs(node: HuffmanNode, prefix: str):
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = prefix or "0"
            return
        dfs(node.left, prefix + "0")
        dfs(node.right, prefix + "1")

    dfs(root, "")
    return codes


def greedy_coin_change(amount: int, coins: List[int]) -> Tuple[int, List[int]]:
    """
    Жадная выдача сдачи для канонических систем монет.

    Параметры:
        amount: сумма
        coins: список номиналов (любой порядок)

    Возвращает:
        (number_of_coins_used, list_of_coins_used)
        Если сумму нельзя представить — возвращает (-1, []).

    Примечание: жадный алгоритм корректен для канонических наборов монет
    (напр., [1,2,5,10,20,50,100] и т.п.), но может быть не оптимален для произвольных наборов.
    Временная сложность: O(m + k) ~ O(m) с учётом типов монет.
    """
    if amount < 0:
        return -1, []

    coins_sorted = sorted(coins, reverse=True)
    remaining = amount
    used: List[int] = []
    for c in coins_sorted:
        if c <= 0:
            continue
        cnt = remaining // c
        if cnt > 0:
            used.extend([c] * cnt)
            remaining -= c * cnt
    if remaining != 0:
        return -1, []
    return len(used), used
# ---- Краскал для MST ----
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        else:
            self.parent[rb] = ra
            if self.rank[ra] == self.rank[rb]:
                self.rank[ra] += 1
        return True


def kruskal_mst(n_nodes: int, edges: List[Tuple[int, int, float]]) -> Tuple[float, List[Tuple[int, int, float]]]:
    """
    edges: list of (u, v, weight)
    Возвращает (total_weight, mst_edges)
    Временная сложность: O(E log E) из-за сортировки рёбер.
    """
    uf = UnionFind(n_nodes)
    edges_sorted = sorted(edges, key=lambda x: x[2])
    mst = []
    total = 0.0
    for u, v, w in edges_sorted:
        if uf.union(u, v):
            mst.append((u, v, w))
            total += w
        if len(mst) == n_nodes - 1:
            break
    if len(mst) != n_nodes - 1:
        raise ValueError("Graph is not connected; MST does not exist.")
    return total, mst
