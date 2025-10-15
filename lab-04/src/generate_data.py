import random


def generate_data(size, data_type="random"):
    """Генерация массива заданного типа."""
    arr = list(range(size))
    if data_type == "random":
        random.shuffle(arr)
    elif data_type == "reversed":
        arr = arr[::-1]
    elif data_type == "almost_sorted":
        random.shuffle(arr[: int(size * 0.05)])
    elif data_type == "sorted":
        pass  # уже отсортирован
    return arr
