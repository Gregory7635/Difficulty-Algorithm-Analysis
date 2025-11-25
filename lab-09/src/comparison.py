"""
comparison.py

Скрипт для экспериментального исследования и сравнения:
- Бенчмарки для Фибоначчи: naive vs memo vs iter vs fast doubling
- Сравнение greedy fractional knapsack vs 0-1 knapsack DP на одном наборе предметов
- Эксперимент масштабируемости knapsack (время и память) при увеличении размеров
- Сбор результатов, сохранение CSV, построение графиков в папку report/

Запуск:
    python src/comparison.py

Зависимости:
    Python 3.8+
    matplotlib
    numpy
    tracemalloc
"""

import os
import csv
import random
import time
import tracemalloc
from typing import List, Tuple

import matplotlib.pyplot as plt  # используется для генерации графиков
import numpy as np

from dynamic_programming import (
    fib_naive, fib_memo, fib_iter, fib_fast_doubling,
    knapsack_01, knapsack_01_space_optimized,
    lcs, levenshtein,
    coin_change_min_coins,
    lis_n2, lis_nlogn
)


REPORT_DIR = os.path.join(os.path.dirname(__file__), '..', 'report')
if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR, exist_ok=True)


# ---------------------------
# Измерения: время + память
# ---------------------------

def measure_time_and_mem(func, *args, **kwargs):
    """
    Возвращает (result, time_seconds, peak_memory_kb)
    Использует tracemalloc для измерения пикового потребления памяти (в КБ).
    """
    tracemalloc.start()
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, end - start, peak // 1024  # KB


# ---------------------------
# Бенчмарки для Фибоначчи
# ---------------------------

def benchmark_fib():
    """
    Выполняет серию замеров для разных реализаций Фибоначчи и сохраняет графики.
    """
    ns_naive = list(range(5, 36))  # naive быстро растет; ограничим до 35
    ns_fast = list(range(10, 2001, 50))  # для memo/iter/fast

    results = {
        'naive': [],
        'memo': [],
        'iter': [],
        'fast': []
    }

    # naive (ограничить)
    for n in ns_naive:
        try:
            _, t, mem = measure_time_and_mem(fib_naive, n)
            results['naive'].append((n, t, mem))
            print(f"naive fib n={n}: t={t:.6f}s mem={mem}KB")
        except RecursionError:
            results['naive'].append((n, float('inf'), -1))
            print(f"naive fib n={n}: recursion error")

    # memo, iter, fast
    for n in ns_fast:
        _, t_memo, mem_memo = measure_time_and_mem(fib_memo, n)
        _, t_iter, mem_iter = measure_time_and_mem(fib_iter, n)
        _, t_fast, mem_fast = measure_time_and_mem(fib_fast_doubling, n)
        results['memo'].append((n, t_memo, mem_memo))
        results['iter'].append((n, t_iter, mem_iter))
        results['fast'].append((n, t_fast, mem_fast))
        print(f"fib n={n}: memo {t_memo:.6f}s, iter {t_iter:.6f}s, fast {t_fast:.6f}s")

    # Сохранение CSV
    csv_path = os.path.join(REPORT_DIR, 'fib_benchmarks.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'n', 'time_s', 'peak_mem_kb'])
        for n, t, mem in results['naive']:
            writer.writerow(['naive', n, t, mem])
        for method in ['memo', 'iter', 'fast']:
            for n, t, mem in results[method]:
                writer.writerow([method, n, t, mem])

    # Построение графиков времени (логарифм времени для читаемости)
    plt.figure()
    # naive
    ns = [r[0] for r in results['naive']]
    ts = [r[1] for r in results['naive']]
    plt.plot(ns, ts, label='naive', marker='o')
    # memo
    ns = [r[0] for r in results['memo']]
    ts = [r[1] for r in results['memo']]
    plt.plot(ns, ts, label='memo', marker='o')
    # iter
    ns = [r[0] for r in results['iter']]
    ts = [r[1] for r in results['iter']]
    plt.plot(ns, ts, label='iter', marker='o')
    # fast
    ns = [r[0] for r in results['fast']]
    ts = [r[1] for r in results['fast']]
    plt.xlabel('n')
    plt.ylabel('time (s)')
    plt.title('Fibonacci: time comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REPORT_DIR, 'fib_time.png'))
    plt.close()

    # Память
    plt.figure()
    for method in ['memo', 'iter', 'fast']:
        ns = [r[0] for r in results[method]]
        mems = [r[2] for r in results[method]]
        plt.plot(ns, mems, label=method)
    plt.xlabel('n')
    plt.ylabel('peak memory (KB)')
    plt.title('Fibonacci: memory comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REPORT_DIR, 'fib_memory.png'))
    plt.close()

    print(f"Фибоначчи: результаты сохранены в {REPORT_DIR}")


# ---------------------------
# Сравнение greedy (fractional) vs 0-1 knapsack
# ---------------------------

def fractional_knapsack(values: List[int], weights: List[int], capacity: int) -> Tuple[float, List[Tuple[int, float]]]:
    """
    Жадный алгоритм для непрерывного (fractional) knapsack.
    Возвращает (total_value, list of (index, fraction_taken))
    Время: O(n log n) (сортировка).
    """
    items = []
    for i, (v, w) in enumerate(zip(values, weights)):
        items.append((i, v / w, v, w))
    items.sort(key=lambda x: x[1], reverse=True)
    remain = capacity
    total = 0.0
    taken = []
    for idx, ratio, v, w in items:
        if remain == 0:
            break
        if w <= remain:
            taken.append((idx, 1.0))
            total += v
            remain -= w
        else:
            frac = remain / w
            taken.append((idx, frac))
            total += v * frac
            remain = 0
    return total, taken


def compare_knapsack():
    """
    Создаёт пример, сравнивает жадный непрерывный вариант с DP 0-1.
    Сохраняет результаты и график.
    """
    random.seed(0)
    n = 20
    values = [random.randint(10, 100) for _ in range(n)]
    weights = [random.randint(1, 50) for _ in range(n)]
    capacity = sum(weights) // 4  # ограниченный капасити

    frac_value, frac_taken = fractional_knapsack(values, weights, capacity)
    dp_value, items, _ = knapsack_01(values, weights, capacity)

    # Сохранение в CSV
    csv_path = os.path.join(REPORT_DIR, 'knapsack_compare.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'value', 'capacity'])
        writer.writerow(['fractional', frac_value, capacity])
        writer.writerow(['0-1 DP', dp_value, capacity])

    # Бар-чарт
    plt.figure()
    plt.bar(['fractional', '0-1 DP'], [frac_value, dp_value])
    plt.ylabel('value')
    plt.title('Fractional vs 0-1 knapsack (example)')
    plt.savefig(os.path.join(REPORT_DIR, 'knapsack_compare.png'))
    plt.close()

    print(f"Knapsack compare: fractional={frac_value:.2f}, 0-1 DP={dp_value}")
    return {
        'values': values,
        'weights': weights,
        'capacity': capacity,
        'fractional_value': frac_value,
        'dp_value': dp_value,
        'dp_items': items,
        'fractional_taken': frac_taken
    }


# ---------------------------
# Эксперимент масштабируемости для knapsack
# ---------------------------

def knapsack_scalability_experiment():
    """
    Исследует, как растёт время при увеличении n и capacity (пример).
    Сохраняет график зависимости времени от capacity для фиксированных n.
    """
    random.seed(1)
    ns = [20, 50, 100]
    capacities = [100, 200, 400, 800]
    results = []

    for n in ns:
        values = [random.randint(1, 100) for _ in range(n)]
        weights = [random.randint(1, 50) for _ in range(n)]
        times = []
        for cap in capacities:
            _, t, mem = measure_time_and_mem(knapsack_01, values, weights, cap)
            times.append(t)
            print(f"knapsack n={n} cap={cap}: t={t:.4f}s mem={mem}KB")
            results.append((n, cap, t, mem))
        # plot for this n
        plt.figure()
        plt.plot(capacities, times, marker='o', label=f'n={n}')
        plt.xlabel('capacity')
        plt.ylabel('time (s)')
        plt.title(f'Knapsack scalability n={n}')
        plt.grid(True)
        plt.savefig(os.path.join(REPORT_DIR, f'knapsack_scalability_n{n}.png'))
        plt.close()

    # save CSV
    csv_path = os.path.join(REPORT_DIR, 'knapsack_scalability.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 'capacity', 'time_s', 'peak_mem_kb'])
        for row in results:
            writer.writerow(row)
    print(f"Knapsack scalability results saved to {REPORT_DIR}")


# ---------------------------
# Примеры LCS, levenshtein, coin change, LIS и визуализация таблиц
# ---------------------------

def examples_and_visualizations():
    # LCS example
    a = "AGGTAB"
    b = "GXTXAYB"
    length, subseq, dp_lcs = lcs(a, b)
    print(f"LCS('{a}','{b}') = {length}, subsequence = '{subseq}'")
    # Сохраняем картинку таблицы (простая визуализация через imshow)
    plt.figure()
    plt.imshow(np.array(dp_lcs), aspect='auto')
    plt.colorbar()
    plt.title('LCS DP table')
    plt.savefig(os.path.join(REPORT_DIR, 'lcs_table.png'))
    plt.close()

    # Levenshtein example
    s1 = "kitten"
    s2 = "sitting"
    dist, dp_lev = levenshtein(s1, s2)
    print(f"Levenshtein('{s1}','{s2}') = {dist}")
    plt.figure()
    plt.imshow(np.array(dp_lev), aspect='auto')
    plt.colorbar()
    plt.title('Levenshtein DP table')
    plt.savefig(os.path.join(REPORT_DIR, 'levenshtein_table.png'))
    plt.close()

    # Coin change
    coins = [1, 3, 4]
    amount = 6
    min_coins, coin_list, dp_coins = coin_change_min_coins(coins, amount)
    print(f"Coin change: amount={amount} min_coins={min_coins} coins_used={coin_list}")

    # LIS examples
    arr = [3, 10, 2, 1, 20]
    l_n2, seq_n2, dp_arr = lis_n2(arr)
    l_nlogn, seq_nlogn = lis_nlogn(arr)
    print(f"LIS O(n^2): len={l_n2}, seq={seq_n2}")
    print(f"LIS O(n log n): len={l_nlogn}, seq={seq_nlogn}")


# ---------------------------
# Main: запуск всех экспериментов
# ---------------------------

def main():
    print("Запуск бенчмарков Fibonacci...")
    benchmark_fib()
    print("\nСравнение knapsack greedy vs dp...")
    knapsack_info = compare_knapsack()
    print("\nМасштабируемость knapsack...")
    knapsack_scalability_experiment()
    print("\nПримеры LCS/Levenshtein/CoinChange/LIS и визуализации таблиц...")
    examples_and_visualizations()
    print("\nВсе результаты и графики сохранены в папке report (в корне проекта).")


if __name__ == '__main__':
    main()
