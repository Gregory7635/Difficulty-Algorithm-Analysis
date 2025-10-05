# recursion_tasks.py

import os


def traverse_directory(path: str, indent: str = '', depth: int = 0, max_depth: int = 2):
    """
    Рекурсивно обходит файловую систему до максимальной глубины (max_depth).
    """
    # Новый базовый случай: останавливаемся, если достигли лимита глубины
    if depth > max_depth:
        print(f"{indent}└── ... (достигнут лимит глубины)")
        return

    if not os.path.isdir(path):
        print(f"Ошибка: '{path}' не является директорией.")
        return

    try:
        items = os.listdir(path)
    except PermissionError:
        print(f"{indent}├── [Отказано в доступе] {os.path.basename(path)}")
        return

    print(f"{indent}├── {os.path.basename(path)}/")
    indent += "│   "

    for i, item_name in enumerate(items):
        item_path = os.path.join(path, item_name)
        is_last = i == len(items) - 1
        prefix = "└── " if is_last else "├── "

        if os.path.isdir(item_path):
            # Рекурсивный вызов для поддиректории с увеличением глубины
            new_indent = indent.replace("│", " ", 1) if is_last else indent
            traverse_directory(item_path, new_indent, depth + 1, max_depth)
        else:
            print(f"{indent}{prefix}{item_name}")


def hanoi(n: int, source: str, destination: str, auxiliary: str):
    """
    Рекурсивное решение задачи о Ханойских башнях.
    Выводит последовательность перемещений дисков.
    """
    if n > 0:
        # Шаг 1: Переместить n-1 дисков с исходного стержня на вспомогательный
        hanoi(n - 1, source, auxiliary, destination)

        # Шаг 2: Переместить самый большой диск (n) с исходного на целевой
        print(f"Переместить диск {n} с '{source}' на '{destination}'")

        # Шаг 3: Переместить n-1 дисков со вспомогательного стержня на целевой
        hanoi(n - 1, auxiliary, destination, source)
