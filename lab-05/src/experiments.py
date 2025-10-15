"""
Проведение экспериментов и визуализация:
- измерение времени, памяти, количества коллизий
- построение графиков по результатам
"""

import time
import tracemalloc
import random
import string
import os
import matplotlib.pyplot as plt
from hash_functions import HASH_FUNCTIONS
from hash_table_chaining import HashTableChaining
from hash_table_open_addressing import HashTableOpenAddressing
import json
import pandas as pd


# Генерация случайных ключей

def random_key(length: int = 8) -> str:
    return ''.join(random.choices(string.ascii_letters, k=length))


# Основной эксперимент

def run_experiment(num_keys=1000, load_factors=(0.1, 0.5, 0.7, 0.9)):
    results = []
    for func_name, func in HASH_FUNCTIONS.items():
        for lf in load_factors:
            capacity = int(num_keys / lf)
            data = [random_key() for _ in range(num_keys)]

            # --- Chaining ---
            ht_chain = HashTableChaining(func, capacity)
            tracemalloc.start()
            t0 = time.perf_counter()
            for k in data:
                ht_chain.insert(k, 1)
            elapsed = time.perf_counter() - t0
            mem = tracemalloc.get_traced_memory()[1] / 1024
            tracemalloc.stop()

            results.append({
                "method": "chaining",
                "hash": func_name,
                "load_factor": lf,
                "time_sec": elapsed,
                "mem_kb": mem,
                "collisions": ht_chain.collisions
            })

            # --- Open Addressing (linear) ---
            ht_open = HashTableOpenAddressing(func, capacity, "linear")
            tracemalloc.start()
            t1 = time.perf_counter()
            for k in data:
                ht_open.insert(k, 1)
            elapsed2 = time.perf_counter() - t1
            mem2 = tracemalloc.get_traced_memory()[1] / 1024
            tracemalloc.stop()

            results.append({
                "method": "open_linear",
                "hash": func_name,
                "load_factor": lf,
                "time_sec": elapsed2,
                "mem_kb": mem2,
                "collisions": ht_open.collisions
            })
    return results


# Визуализация результатов

def visualize_results(results: list, output_dir: str):
    """
    Визуализирует результаты эксперимента:
    - график зависимости времени вставки от коэффициента заполнения
    - гистограммы коллизий для разных хеш-функций
    (логарифмическая шкала по оси Y для наглядности)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)

    # --- 1. График времени ---
    plt.figure(figsize=(10, 6))
    for (method, hname), group in df.groupby(["method", "hash"]):
        plt.plot(
            group["load_factor"],
            group["time_sec"],
            marker="o",
            label=f"{hname} ({method})"
        )

    plt.yscale("log")
    plt.title("Зависимость времени вставки от коэффициента заполнения (лог. шкала)")
    plt.xlabel("Коэффициент заполнения (α)")
    plt.ylabel("Время вставки (сек)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_vs_load_log.png"), dpi=300)
    plt.close()

    # --- 2. Гистограммы коллизий ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for i, method in enumerate(["chaining", "open_linear"]):
        ax = axes[i]
        sub = df[df["method"] == method]
        grouped = sub.groupby("hash")["collisions"].mean().reset_index()
        ax.bar(grouped["hash"], grouped["collisions"])
        ax.set_title(f"Коллизии ({method})")
        ax.set_xlabel("Хеш-функция")
        if i == 0:
            ax.set_ylabel("Количество коллизий (лог. шкала)")
        ax.set_yscale("log")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Среднее количество коллизий (логарифмическая шкала)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "collisions_log.png"), dpi=300)
    plt.close()

    print(f" Графики сохранены в {output_dir} (логарифмическая шкала)")


# Запуск для main.py

def run_all():
    results = run_experiment()
    visualize_results(results, "report")
    return results
