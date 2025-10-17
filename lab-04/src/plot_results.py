import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Пути ---
BASE_DIR = os.path.dirname(__file__)
REPORT_DIR = os.path.join(BASE_DIR, "..", "report")
RESULTS_PATH = os.path.join(BASE_DIR, "results.csv")

# --- Проверка наличия данных ---
if not os.path.exists(RESULTS_PATH):
    raise FileNotFoundError("Файл results.csv не найден. Сначала запусти performance_test.py")

# --- Загрузка ---
df = pd.read_csv(RESULTS_PATH)
print("=== DEBUG ===")
print("Столбцы:", df.columns.tolist())
print("Размер таблицы:", df.shape)
print(df.head(10))
print("Уникальные алгоритмы:", df["Algorithm"].unique())
print("Уникальные типы данных:", df["DataType"].unique())
print("================")


# --- Создание отчётной папки ---
os.makedirs(REPORT_DIR, exist_ok=True)

# --- 1. Время vs Размер массива (для Random данных) ---
data_type = "Random"
subset = df[df["DataType"] == data_type]

plt.figure(figsize=(10, 6))
for algo in subset["Algorithm"].unique():
    algo_data = subset[subset["Algorithm"] == algo]
    plt.plot(algo_data["Size"], algo_data["Time"],
             marker="o", label=algo)

plt.title(f"Зависимость времени выполнения от размера массива ({data_type})")
plt.xlabel("Размер массива (n)")
plt.ylabel("Время выполнения (сек)")
plt.legend()
plt.grid(True)
plt.tight_layout()
time_size_path = os.path.join(REPORT_DIR, "time_vs_size_random.png")
plt.savefig(time_size_path)
plt.close()

# --- 2. Память vs Размер массива (для Random данных) ---
plt.figure(figsize=(10, 6))
for algo in subset["Algorithm"].unique():
    algo_data = subset[subset["Algorithm"] == algo]
    plt.plot(algo_data["Size"], algo_data["Memory (KB)"],
             marker="o", label=algo)

plt.title(f"Зависимость использования памяти от размера массива ({data_type})")
plt.xlabel("Размер массива (n)")
plt.ylabel("Память (КБ)")
plt.legend()
plt.grid(True)
plt.tight_layout()
mem_size_path = os.path.join(REPORT_DIR, "memory_vs_size_random.png")
plt.savefig(mem_size_path)
plt.close()

# --- 3. Время vs Тип данных (для фиксированного n=5000) ---
fixed_size = 1000
subset2 = df[df["Size"] == fixed_size]

plt.figure(figsize=(10, 6))
for algo in subset2["Algorithm"].unique():
    algo_data = subset2[subset2["Algorithm"] == algo]
    plt.plot(algo_data["DataType"], algo_data["Time"],
             marker="o", label=algo)

plt.title(f"Зависимость времени выполнения от типа данных (n={fixed_size})")
plt.xlabel("Тип данных")
plt.ylabel("Время выполнения (сек)")
plt.legend()
plt.grid(True)
plt.tight_layout()
time_type_path = os.path.join(REPORT_DIR, "time_vs_type_5000.png")
plt.savefig(time_type_path)
plt.close()

# --- 4. Память vs Тип данных (для фиксированного n=5000) ---
plt.figure(figsize=(10, 6))
for algo in subset2["Algorithm"].unique():
    algo_data = subset2[subset2["Algorithm"] == algo]
    plt.plot(algo_data["DataType"], algo_data["Memory (KB)"],
             marker="o", label=algo)

plt.title(f"Зависимость использования памяти от типа данных (n={fixed_size})")
plt.xlabel("Тип данных")
plt.ylabel("Память (КБ)")
plt.legend()
plt.grid(True)
plt.tight_layout()
mem_type_path = os.path.join(REPORT_DIR, "memory_vs_type_5000.png")
plt.savefig(mem_type_path)
plt.close()

# --- 5. Сводная таблица ---
pivot = pd.pivot_table(
    df,
    values=["Time", "Memory (KB)"],
    index=["Algorithm"],
    columns=["DataType"],
    aggfunc="mean",
)

pivot_path = os.path.join(REPORT_DIR, "summary_table.csv")
pivot.to_csv(pivot_path)

# --- 6. Markdown-отчёт ---
report_md_path = os.path.join(REPORT_DIR, "report.md")

with open(report_md_path, "w", encoding="utf-8") as f:
    f.write("# Результаты лабораторной работы №4 — Сортировки\n\n")
    f.write("## 1. Графики зависимости времени и памяти от размера массива (Random)\n\n")
    f.write(f"![Время выполнения]({os.path.basename(time_size_path)})\n\n")
    f.write(f"![Память]({os.path.basename(mem_size_path)})\n\n")

    f.write("## 2. Графики зависимости времени и памяти от типа данных (n=5000)\n\n")
    f.write(f"![Время выполнения]({os.path.basename(time_type_path)})\n\n")
    f.write(f"![Память]({os.path.basename(mem_type_path)})\n\n")

    f.write("## 3. Сводная таблица результатов\n\n")
    f.write(pivot.to_markdown())

print(" Все графики и отчёт сохранены в lab-04/report")
