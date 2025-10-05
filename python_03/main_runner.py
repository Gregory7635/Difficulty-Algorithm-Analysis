# main_runner.py

import timeit
import matplotlib.pyplot as plt
import os
import sys


import memoization
from memoization import fibonacci_memo
from recursion import fibonacci_naive
from recursion_tasks import hanoi, traverse_directory


def run_fibonacci_comparison(n_val=35):
    """Сравнивает наивную и мемоизированную рекурсию для чисел Фибоначчи."""
    print("--- 1. Сравнение производительности Фибоначчи ---")

    # --- Сравнение времени ---
    print(f"\nСравнение времени выполнения для n = {n_val}:")

    # Замер времени для наивной версии
    naive_time = timeit.timeit(lambda: fibonacci_naive(n_val), number=1)
    print(f"Наивная рекурсия: {naive_time:.6f} секунд")

    # Замер времени для версии с мемоизацией
    # from memoization import fibonacci_memo
    memo_time = timeit.timeit(lambda: memoization.fibonacci_memo(n_val), number=1)
    print(f"Рекурсия с мемоизацией: {memo_time:.6f} секунд")

    # --- Сравнение количества вызовов ---
    print(f"\nСравнение количества рекурсивных вызовов для n = {n_val}:")

    # Сбрасываем счетчики перед вызовом
    memoization.call_count_naive = 0
    memoization.call_count_memo = 0
    memoization.memo_cache_counter.clear()

    # Считаем вызовы для наивной версии
    memoization.fib_naive_with_counter(n_val)
    print(f"Наивная рекурсия: {memoization.call_count_naive} вызовов")

    # Считаем вызовы для версии с мемоизацией
    memoization.fib_memo_with_counter(n_val)
    print(f"Рекурсия с мемоизацией: {memoization.call_count_memo} вызовов")


def plot_fibonacci_performance():
    """Строит график зависимости времени выполнения от n для Фибоначчи."""
    print("\n--- 2. Построение графика производительности ---")

    naive_times = []
    memo_times = []
    n_values_naive = range(1, 37)
    n_values_memo = range(1, 101)  # Можно взять больше, т.к. работает быстро

    for n in n_values_naive:
        t = timeit.timeit(lambda: fibonacci_naive(n), number=1)
        naive_times.append(t)
        print(f"Наивный F({n}) - {t:.6f} сек.")

    for n in n_values_memo:
        t = timeit.timeit(lambda: fibonacci_memo(n), number=100) / 100
        memo_times.append(t)

    plt.figure(figsize=(12, 7))
    plt.plot(n_values_naive, naive_times, 'r-o', label='Наивная рекурсия O(2^n)')
    plt.plot(n_values_memo, memo_times, 'g-o', label='Рекурсия с мемоизацией O(n)')
    plt.xlabel('Число n')
    plt.ylabel('Время выполнения (секунды)')
    plt.title('Сравнение производительности вычисления чисел Фибоначчи')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Логарифмическая шкала для наглядности
    plt.savefig('fibonacci_performance_comparison.png')
    print("График сохранен в 'fibonacci_performance_comparison.png'")


def run_hanoi_demo(disks=3):
    """Демонстрация решения задачи о Ханойских башнях."""
    print(f"\n--- 3. Задача о Ханойских башнях для {disks} дисков ---")
    hanoi(n=disks, source="A", destination="C", auxiliary="B")


def run_fs_traversal_demo():
    """Демонстрация обхода файловой системы."""
    print("\n--- 4. Обход файловой системы (с ограничением глубины до 2) ---")
    start_path = '.'
    print(f"Дерево каталогов, начиная с '{os.path.abspath(start_path)}':")
    # Вызываем функцию с ограничением по умолчанию
    traverse_directory(start_path, max_depth=2)


def measure_max_recursion_depth():
    """Экспериментально измеряет максимальную глубину рекурсии
    с помощью простой функции."""
    print("\n--- 5. Исследование максимальной глубины рекурсии ---")

    initial_limit = sys.getrecursionlimit()
    print(f"Текущий лимит рекурсии в Python: {initial_limit}")

    def simple_recursion(n):
        print(f"Глубина: {n}", end='\r')  # Показываем текущую глубину
        simple_recursion(n + 1)

    try:
        simple_recursion(1)
    except RecursionError:
        print("\nДостигнут предел глубины рекурсии! Эксперимент успешен.")
    except Exception as e:
        print(f"\nПроизошла неожиданная ошибка: {e}")


if __name__ == "__main__":
    run_fibonacci_comparison(n_val=35)
    plot_fibonacci_performance()
    run_hanoi_demo(disks=3)
    run_fs_traversal_demo()
    measure_max_recursion_depth()
