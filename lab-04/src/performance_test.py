import os
import csv
import random
import timeit
import tracemalloc

from sorts import (
    bubble_sort,
    selection_sort,
    insertion_sort,
    merge_sort,
    quick_sort,
)


# --- Папка для результатов ---
BASE_DIR = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(BASE_DIR, "results.csv")

# --- Настройки теста ---
SIZES = [100, 500, 1000, 2000, 5000]
DATA_TYPES = ["Random", "Sorted", "Reversed", "Almost_Sorted"]

ALGORITHMS = {
    "Bubble": bubble_sort,
    "Selection": selection_sort,
    "Insertion": insertion_sort,
    "Merge": merge_sort,
    "Quick": quick_sort,
}


# --- Генерация данных ---
def generate_data(size, data_type):
    data = list(range(size))
    if data_type == "Random":
        random.shuffle(data)
    elif data_type == "Reversed":
        data.reverse()
    elif data_type == "Almost_Sorted":
        data = list(range(size))
        # 5% элементов случайно переставлены
        swaps = max(1, size // 20)
        for _ in range(swaps):
            i, j = random.sample(range(size), 2)
            data[i], data[j] = data[j], data[i]
    # "Sorted" возвращает уже отсортированный массив
    return data


# --- Основной эксперимент ---
with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Algorithm", "DataType", "Size", "Time", "Memory (KB)"])

    for size in SIZES:
        for data_type in DATA_TYPES:
            base_data = generate_data(size, data_type)
            for algo_name, algo_func in ALGORITHMS.items():
                # ограничиваем O(n²) сортировки
                if algo_name in ["Bubble", "Selection", "Insertion"] and size > 2000:
                    continue

                print(f"→ {algo_name} | {data_type} | n={size}")

                arr_copy = base_data.copy()

                tracemalloc.start()
                start = timeit.default_timer()
                algo_func(arr_copy)
                elapsed = timeit.default_timer() - start
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                writer.writerow(
                    [algo_name, data_type, size, f"{elapsed:.6f}", f"{peak / 1024:.2f}"]
                )

print(f" Результаты сохранены в {RESULTS_PATH}")
