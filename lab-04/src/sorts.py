"""
Модуль sorts.py
Содержит реализации 5 алгоритмов сортировки:
Bubble Sort, Selection Sort, Insertion Sort, Merge Sort, Quick Sort.

Каждая функция возвращает новый отсортированный список.
"""

# ------------------------------
# СОРТИРОВКА ПУЗЫРЬКОМ (Bubble Sort)
# ------------------------------


def bubble_sort(arr):
    """
    Алгоритм: Многократно проходит по массиву,
    сравнивая и меняя местами соседние элементы.
    Время:  O(n²)  (все случаи)
    Память: O(1)
    """
    a = arr.copy()
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a


# ------------------------------
# СОРТИРОВКА ВЫБОРОМ (Selection Sort)
# ------------------------------
def selection_sort(arr):
    """
    Алгоритм: Находит минимальный элемент в неотсортированной части
    и ставит его на правильное место.
    Время:  O(n²)
    Память: O(1)
    """
    a = arr.copy()
    n = len(a)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if a[j] < a[min_idx]:
                min_idx = j
        a[i], a[min_idx] = a[min_idx], a[i]
    return a


# ------------------------------
# СОРТИРОВКА ВСТАВКАМИ (Insertion Sort)
# ------------------------------
def insertion_sort(arr):
    """
    Алгоритм: Вставляет каждый элемент в уже отсортированную часть массива.
    Время:  O(n²) - худший и средний, O(n) - лучший (уже отсортированный)
    Память: O(1)
    """
    a = arr.copy()
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


# ------------------------------
# СОРТИРОВКА СЛИЯНИЕМ (Merge Sort)
# ------------------------------
def merge_sort(arr):
    """
    Алгоритм: Делит массив пополам, сортирует рекурсивно и сливает.
    Время:  O(n log n) во всех случаях
    Память: O(n)
    """
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)


def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# ------------------------------
# БЫСТРАЯ СОРТИРОВКА (Quick Sort)
# ------------------------------
def quick_sort(arr):
    """
    Алгоритм: Выбирает опорный элемент, рекурсивно сортирует
    подмассивы меньших и больших элементов.
    Время:  O(n log n) в среднем, O(n²) в худшем
    Память: O(log n) (рекурсия)
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    less = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]
    return quick_sort(less) + equal + quick_sort(greater)
