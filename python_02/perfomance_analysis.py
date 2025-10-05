#  Анализ производительности + визуализация
"""
performance_analysis.py

Проводит замеры производительности и строит графики:
- list.insert(0, x) vs LinkedList.insert_at_start (1000 операций)
- list.append vs LinkedList.insert_at_end (1000 операций)
  -- показывает константную амортизированную сложность append
- list.pop(0) vs collections.deque.popleft (1000 операций)

Сохраняет графики:
- insertion_comparison.png
- queue_comparison.png

Выводит характеристики ПК (строку pc_info) для воспроизводимости.
"""

from __future__ import annotations

import timeit
from collections import deque
from typing import Callable

import matplotlib
# использовать неинтерактивный бэкенд для совместимости в средах без GUI
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from linked_list import LinkedList

# Характеристики ПК (пользователь может изменить под свою машину)
pc_info = """
Характеристики ПК для тестирования:
- Процессор: Intel Core i7-8700 @ 3.6GHz
- Оперативная память: 32 GB DDR4
- OC: Windows 11
- Python: 3.11.9
"""

print(pc_info)


def measure_avg_time(stmt: Callable[[], None], runs: int = 100) -> float:
    """
    Измеряет среднее время выполнения callable `stmt` в миллисекундах.
    """
    t = timeit.timeit(stmt, number=runs)
    return (t / runs) * 1000.0


def benchmark_insert_start_list_vs_linked(num_ops: int = 1000, runs: int = 10):
    """
    Сравнение вставки в начало:
    - list.insert(0, x) -> O(n) для каждого вставления
    - LinkedList.insert_at_start -> O(1)
    Возвращает списки средних времён для разных размеров.
    """
    list_times = []
    ll_times = []
    sizes = list(range(100, num_ops + 1, max(1, num_ops // 10)))

    for ops in sizes:
        # list
        def run_list():
            lst = []
            for i in range(ops):
                lst.insert(0, i)

        # linked list
        def run_ll():
            ll = LinkedList()
            for i in range(ops):
                ll.insert_at_start(i)

        lt = measure_avg_time(run_list, runs)
        llt = measure_avg_time(run_ll, runs)
        list_times.append(lt)
        ll_times.append(llt)

        print(f"Insert-start ops={ops:5d}  list: {lt:8.4f} ms  "
              f"linked: {llt:8.4f} ms")

    return sizes, list_times, ll_times


def benchmark_insert_end_list_vs_linked(num_ops: int = 1000, runs: int = 10):
    """
    Сравнение вставки в конец:
    - list.append -> амортизированное O(1)
    - LinkedList.insert_at_end -> O(1) с tail
    """
    list_times = []
    ll_times = []
    sizes = list(range(100, num_ops + 1, max(1, num_ops // 10)))

    for ops in sizes:
        def run_list():
            lst = []
            for i in range(ops):
                lst.append(i)

        def run_ll():
            ll = LinkedList()
            for i in range(ops):
                ll.insert_at_end(i)

        lt = measure_avg_time(run_list, runs)
        llt = measure_avg_time(run_ll, runs)
        list_times.append(lt)
        ll_times.append(llt)

        print(f"Insert-end ops={ops:5d}  list.append: {lt:8.4f} ms  "
              f"linked.insert_end: {llt:8.4f} ms")

    return sizes, list_times, ll_times


def benchmark_queue_pop0_vs_deque(num_ops: int = 1000, runs: int = 10):
    """
    Сравнение удаления из начала:
    - list.pop(0) -> O(n)
    - deque.popleft() -> O(1)
    """
    list_times = []
    deque_times = []
    sizes = list(range(100, num_ops + 1, max(1, num_ops // 10)))

    for ops in sizes:
        data = list(range(ops))

        def run_list_pop0():
            lst = data[:]  # копируем
            for _ in range(ops):
                if lst:
                    lst.pop(0)

        def run_deque_popleft():
            dq = deque(data)
            for _ in range(ops):
                if dq:
                    dq.popleft()

        lt = measure_avg_time(run_list_pop0, runs)
        dqt = measure_avg_time(run_deque_popleft, runs)
        list_times.append(lt)
        deque_times.append(dqt)

        print(f"Queue ops={ops:5d}  list.pop(0): {lt:8.4f} ms  "
              f"deque.popleft: {dqt:8.4f} ms")

    return sizes, list_times, deque_times


def plot_insertion_comparison(
    sizes, list_times, ll_times, out_file="insertion_comparison.png"
):
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, list_times, 'ro-', label='list.insert(0, x) (O(n))')
    plt.plot(sizes, ll_times, 'bo-', label='LinkedList.insert_at_start (O(1))')
    plt.xlabel("Количество операций (вставок)")
    plt.ylabel("Время (мс)")
    plt.title("Вставка в начало: list vs LinkedList")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    print(f"Saved {out_file}")


def plot_queue_comparison(
    sizes, list_times, deque_times, out_file="queue_comparison.png"
):
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, list_times, 'ro-', label='list.pop(0) (O(n))')
    plt.plot(sizes, deque_times, 'bo-', label='deque.popleft() (O(1))')
    plt.xlabel("Количество элементов в очереди")
    plt.ylabel("Время (мс)")
    plt.title("Удаление из начала: list vs deque")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    print(f"Saved {out_file}")


def main():
    print("=== Benchmark: вставка в начало "
          "(list.insert(0) vs LinkedList.insert_at_start) ===")
    sizes, list_times, ll_times = benchmark_insert_start_list_vs_linked(
        num_ops=2000, runs=5
    )
    plot_insertion_comparison(sizes, list_times, ll_times)

    print("\n=== Benchmark: вставка в конец "
          "(list.append vs LinkedList.insert_at_end) ===")
    sizes_e, list_times_e, ll_times_e = benchmark_insert_end_list_vs_linked(
        num_ops=2000, runs=5
    )
    plot_insertion_comparison(
        sizes_e, list_times_e, ll_times_e,
        out_file="insert_end_comparison.png"
    )

    print("\n=== Benchmark: очередь (list.pop(0) vs deque.popleft) ===")
    sizes_q, list_q_times, deque_times = benchmark_queue_pop0_vs_deque(
        num_ops=2000, runs=5
    )
    plot_queue_comparison(sizes_q, list_q_times, deque_times)

    print("\nГрафики сохранены. файлы: insertion_comparison.png, "
          "insert_end_comparison.png, queue_comparison.png")


if __name__ == "__main__":
    main()
