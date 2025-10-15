"""
Главный файл лабораторной работы №5:
Хеш-функции и хеш-таблицы.
"""

import json
from experiments import run_all

if __name__ == "__main__":
    results = run_all()
    print(json.dumps(results, indent=2))
