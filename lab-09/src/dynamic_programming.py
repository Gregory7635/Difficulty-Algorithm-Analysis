"""
dynamic_programming.py

Реализация классических алгоритмов динамического программирования.

Содержит:
- Фибоначчи: наивная рекурсия, мемоизация, итеративная табличная, быстрый doubling (O(log n))
- 0-1 Knapsack (восходящий dp 2D + пространственно оптимизированный 1D), восстановление набора предметов
- LCS (восходящий dp 2D), восстановление подпоследовательности
- Расстояние Левенштейна (редакционное) (восходящий dp 2D)
- Размен монет (минимальное количество монет), восстановление набора монет
- LIS: O(n^2) DP и O(n log n) patience + восстановление
"""

from typing import List, Tuple, Dict, Optional


# ---------------------------
# ФИБОНАЧЧИ
# ---------------------------

def fib_naive(n: int) -> int:
    """
    Наивная рекурсивная реализация.
    Время: O(2^n), Память: O(n) (стек рекурсии).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)


def fib_memo(n, memo=None):
    if memo is None:
        memo = {}

    if n in memo:
        return memo[n]

    # Базовые случаи
    memo[0] = 0
    if n == 0:
        return 0
    memo[1] = 1
    if n == 1:
        return 1

    # Имитируем мемоизацию, но без рекурсии
    for i in range(2, n + 1):
        if i not in memo:
            memo[i] = memo[i - 1] + memo[i - 2]

    return memo[n]



def fib_iter(n: int) -> int:
    """
    Восходящий (bottom-up) итеративный.
    Время: O(n), Память: O(1).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def fib_fast_doubling(n: int) -> int:
    """
    Быстрый doubling — вычисление за O(log n).
    Возвращает F(n).
    Сложность: O(log n) времени, O(1) памяти (рекурсия приводит к O(log n) стеку).
    """
    def _fd(k: int) -> Tuple[int, int]:
        # returns (F(k), F(k+1))
        if k == 0:
            return 0, 1
        a, b = _fd(k // 2)
        c = a * (2 * b - a)
        d = a * a + b * b
        if k % 2 == 0:
            return c, d
        return d, c + d

    if n < 0:
        raise ValueError("n must be non-negative")
    return _fd(n)[0]


# ---------------------------
# 0-1 KNAPSACK (восходящий 2D + восстановление)
# ---------------------------

def knapsack_01(values: List[int], weights: List[int], capacity: int) -> Tuple[int, List[int], List[List[int]]]:
    """
    Восходящий DP для 0-1 knapsack.
    Возвращает (max_value, items_taken_indices, dp_table)
    dp_table размер (n+1) x (capacity+1)
    Время: O(n * capacity), Память: O(n * capacity)
    """
    n = len(values)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        v = values[i - 1]
        w = weights[i - 1]
        for c in range(capacity + 1):
            without = dp[i - 1][c]
            with_item = 0
            if w <= c:
                with_item = dp[i - 1][c - w] + v
            dp[i][c] = max(without, with_item)

    # Восстановление набора предметов
    res_value = dp[n][capacity]
    c = capacity
    items = []
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            items.append(i - 1)  # взяли предмет i-1
            c -= weights[i - 1]
    items.reverse()
    return res_value, items, dp


def knapsack_01_space_optimized(values: List[int], weights: List[int], capacity: int) -> int:
    """
    Пространственно-оптимизированный 1D вариант (не восстанавливает набор).
    Время: O(n * capacity), Память: O(capacity)
    """
    n = len(values)
    dp = [0] * (capacity + 1)
    for i in range(n):
        v = values[i]
        w = weights[i]
        for c in range(capacity, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[capacity]


# ---------------------------
# LCS (наибольшая общая подпоследовательность)
# ---------------------------

def lcs(a: str, b: str) -> Tuple[int, str, List[List[int]]]:
    """
    Восходящий DP для LCS.
    Возвращает (length, subsequence, dp_table)
    dp table size (len(a)+1) x (len(b)+1)
    Время: O(len(a)*len(b)), Память: O(len(a)*len(b))
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # восстановление подпоследовательности
    i, j = n, m
    seq_chars = []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            seq_chars.append(a[i - 1])
            i -= 1
            j -= 1
        else:
            if dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
    seq_chars.reverse()
    return dp[n][m], ''.join(seq_chars), dp


# ---------------------------
# ЛЕВЕНШТЕЙН (редакционное расстояние)
# ---------------------------

def levenshtein(a: str, b: str) -> Tuple[int, List[List[int]]]:
    """
    Восходящий DP для редакционного расстояния.
    Возвращает (distance, dp_table)
    Время: O(len(a)*len(b)), Память: O(len(a)*len(b))
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # удаление
                dp[i][j - 1] + 1,      # вставка
                dp[i - 1][j - 1] + cost  # замена
            )
    return dp[n][m], dp


# ---------------------------
# РАЗМЕН МОНЕТ (минимальное количество монет)
# ---------------------------

def coin_change_min_coins(coins: List[int], amount: int) -> Tuple[int, List[int], List[int]]:
    """
    DP для минимального числа монет (unlimited coins).
    Возвращает (min_coins or -1 if impossible, coins_used_list, dp array)
    dp[x] = min coins to make x
    Время: O(len(coins) * amount), Память: O(amount)
    """
    INF = 10 ** 9
    dp = [INF] * (amount + 1)
    dp[0] = 0
    parent = [-1] * (amount + 1)  # для восстановления
    for coin in coins:
        for x in range(coin, amount + 1):
            if dp[x - coin] + 1 < dp[x]:
                dp[x] = dp[x - coin] + 1
                parent[x] = coin
    if dp[amount] == INF:
        return -1, [], dp
    # восстановление
    res = []
    cur = amount
    while cur > 0:
        res.append(parent[cur])
        cur -= parent[cur]
    return dp[amount], res, dp


# ---------------------------
# LIS (наибольшая возрастающая подпоследовательность)
# ---------------------------

def lis_n2(arr: List[int]) -> Tuple[int, List[int], List[int]]:
    """
    O(n^2) DP вариант для LIS.
    Возвращает (length, subsequence, dp array - length for each index)
    Время: O(n^2), Память: O(n)
    """
    n = len(arr)
    if n == 0:
        return 0, [], []
    dp = [1] * n
    parent = [-1] * n
    for i in range(n):
        for j in range(i):
            if arr[j] < arr[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    # восстановление
    length = max(dp)
    idx = dp.index(length)
    seq = []
    while idx != -1:
        seq.append(arr[idx])
        idx = parent[idx]
    seq.reverse()
    return length, seq, dp


def lis_nlogn(arr: List[int]) -> Tuple[int, List[int]]:
    """
    O(n log n) patience method для LIS + восстановление последовательности.
    Время: O(n log n), Память: O(n)
    """
    import bisect
    n = len(arr)
    if n == 0:
        return 0, []
    tails = []
    tails_idx = []  # indices in original array of tail values
    parent = [-1] * n
    pos_for_len = [0] * (n + 1)

    for i, val in enumerate(arr):
        # place val in tails
        j = bisect.bisect_left(tails, val)
        if j == len(tails):
            tails.append(val)
            tails_idx.append(i)
        else:
            tails[j] = val
            tails_idx[j] = i
        pos_for_len[j + 1] = i
        if j > 0:
            parent[i] = pos_for_len[j]
    length = len(tails)
    # восстановление индексов последовательности
    seq = []
    k = pos_for_len[length]
    while k != -1:
        seq.append(arr[k])
        k = parent[k]
    seq.reverse()
    return length, seq


# ---------------------------
# утилиты визуализации таблиц (для небольших данных)
# ---------------------------

def pretty_print_dp_table(dp: List[List[int]]) -> None:
    """
    Печать 2D-таблицы dp в удобочитаемом виде (для небольших размеров).
    """
    for row in dp:
        print(' '.join(map(str, row)))
