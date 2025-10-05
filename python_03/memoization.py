# memoization.py

# Словарь для кеширования уже вычисленных значений
fib_cache = {}


def fibonacci_memo(n: int) -> int:
    """
    Вычисляет n-е число Фибоначчи с использованием мемоизации.
    - Временная сложность: O(n)
    - Глубина рекурсии: O(n)
    """
    if n < 0:
        raise ValueError("Числа Фибоначчи не определены для отрицательных n")
    if n in fib_cache:
        return fib_cache[n]
    if n <= 1:
        return n

    # Вычисляем и сохраняем результат в кеш
    result = fibonacci_memo(n - 1) + fibonacci_memo(n - 2)
    fib_cache[n] = result
    return result

# --- Сравнение количества вызовов ---
# Для подсчета вызовов будем использовать глобальные счетчики


call_count_naive = 0


def fib_naive_with_counter(n: int) -> int:
    """Наивная рекурсия Фибоначчи с подсчетом вызовов."""
    global call_count_naive
    call_count_naive += 1
    if n <= 1:
        return n
    return fib_naive_with_counter(n - 1) + fib_naive_with_counter(n - 2)


call_count_memo = 0
memo_cache_counter = {}


def fib_memo_with_counter(n: int) -> int:
    """Мемоизированная рекурсия Фибоначчи с подсчетом вызовов."""
    global call_count_memo
    call_count_memo += 1
    if n in memo_cache_counter:
        return memo_cache_counter[n]
    if n <= 1:
        return n
    result = fib_memo_with_counter(n - 1) + fib_memo_with_counter(n - 2)
    memo_cache_counter[n] = result
    return result
