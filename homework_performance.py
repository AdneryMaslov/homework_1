import torch
import time
from prettytable import PrettyTable

if torch.cuda.is_available():
    print("Используем cuda")
else:
    print("cuda недоступнен")


# Устройства, которые будут тестироваться: CPU и MPS
cpu_device = torch.device("cpu")
mps_device = None

# Проверяем доступность MPS
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print(f"\nИспользуемое устройство для ускорения: MPS (Apple GPU)")
else:
    print("\nПредупреждение: MPS недоступна. Все операции будут выполнены на CPU.")

print(f"CPU тоже будет тестироваться")


# 3.1 Подготовка данных
print("\n-- 3.1 Подготовка данных --")
matrix_sizes = [
    (64, 1024, 1024),
    (128, 512, 512),
    (256, 256, 256),
    (512, 4096, 4096),
    (1024, 2048, 2048),
    (2048, 1024, 1024)
]
# Создаем матрицы на CPU и MPS
matrices_cpu = {}
matrices_mps = {}

for i, (d1, d2, d3) in enumerate(matrix_sizes):
    key_prefix_A = f"Mat_{i+1}_({d1}x{d2})_A"
    key_prefix_B = f"Mat_{i+1}_({d2}x{d3})_B"
    key_prefix_Elem_A = f"Mat_{i+1}_({d1}x{d2})_Elem_A"
    key_prefix_Elem_B = f"Mat_{i+1}_({d1}x{d2})_Elem_B"
    # Создаем на CPU
    matrices_cpu[key_prefix_A] = torch.randn(d1, d2, device=cpu_device)
    matrices_cpu[key_prefix_B] = torch.randn(d2, d3, device=cpu_device)
    matrices_cpu[key_prefix_Elem_A] = torch.randn(d1, d2, device=cpu_device)
    matrices_cpu[key_prefix_Elem_B] = torch.randn(d1, d2, device=cpu_device)
    # Cоздаем копии на MPS
    if mps_device:
        matrices_mps[key_prefix_A] = matrices_cpu[key_prefix_A].to(mps_device)
        matrices_mps[key_prefix_B] = matrices_cpu[key_prefix_B].to(mps_device)
        matrices_mps[key_prefix_Elem_A] = matrices_cpu[key_prefix_Elem_A].to(mps_device)
        matrices_mps[key_prefix_Elem_B] = matrices_cpu[key_prefix_Elem_B].to(mps_device)
        
print("Данные готовы")

# 3.2 Функция измерения времени
def measure_time(func, *args, target_device_type="cpu", num_repetitions=10):
    """
    Измеряет время выполнения операции func.
    Использует соответствующие механизмы синхронизации для CPU и MPS.
    """
    timings = []
    # Прогрев
    for _ in range(3):
        func(*args)
        if target_device_type == "mps":
            torch.mps.synchronize()
    
    # Измерение
    for _ in range(num_repetitions):
        if target_device_type == "cpu":
            start_time = time.perf_counter()
            func(*args)
            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000)
        elif target_device_type == "mps":
            torch.mps.synchronize()
            start_time = time.perf_counter()
            func(*args)
            torch.mps.synchronize()
            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000)
        else:
            raise ValueError(f"Неизвестное устройство: {target_device_type}")
    return sum(timings) / num_repetitions


# 3.3 Сравнение операций
print("\n-- 3.3 Сравнение операций: --")

table = PrettyTable()
table.field_names = ["Операция", "CPU (мс)", "MPS (мс)", "Ускорение"]

for i, (d1, d2, d3) in enumerate(matrix_sizes):
    print(f"\nСравнение для матриц размером ({d1}x{d2}) и ({d2}x{d3}):")
    key_prefix_A = f"Mat_{i+1}_({d1}x{d2})_A"
    key_prefix_B = f"Mat_{i+1}_({d2}x{d3})_B"
    key_prefix_Elem_A = f"Mat_{i+1}_({d1}x{d2})_Elem_A"
    key_prefix_Elem_B = f"Mat_{i+1}_({d1}x{d2})_Elem_B"

    # --- Матричное умножение ---
    row_data_matmul = [f"Матричное умножение ({d1}x{d2} @ {d2}x{d3})"]
    A_cpu = matrices_cpu[key_prefix_A]
    B_cpu = matrices_cpu[key_prefix_B]
    time_cpu_matmul = measure_time(torch.matmul, A_cpu, B_cpu, target_device_type="cpu")
    row_data_matmul.append(f"{time_cpu_matmul:.2f}")

    if mps_device:
        A_mps = matrices_mps[key_prefix_A]
        B_mps = matrices_mps[key_prefix_B]
        time_mps_matmul = measure_time(torch.matmul, A_mps, B_mps, target_device_type="mps")
        speedup_mps_matmul = time_cpu_matmul / time_mps_matmul if time_mps_matmul > 0 else float('nan')
        row_data_matmul.append(f"{time_mps_matmul:.2f}")
        row_data_matmul.append(f"{speedup_mps_matmul:.1f}x" if isinstance(speedup_mps_matmul, float) else "N/A")
    else:
        row_data_matmul.extend(["N/A", "N/A"])
    table.add_row(row_data_matmul)

    # --- Поэлементное сложение ---
    row_data_add = [f"Поэлементное сложение ({d1}x{d2})"]
    A_cpu_elem = matrices_cpu[key_prefix_Elem_A]
    B_cpu_elem = matrices_cpu[key_prefix_Elem_B]
    time_cpu_add = measure_time(torch.add, A_cpu_elem, B_cpu_elem, target_device_type="cpu")
    row_data_add.append(f"{time_cpu_add:.2f}")

    if mps_device:
        A_mps_elem = matrices_mps[key_prefix_Elem_A]
        B_mps_elem = matrices_mps[key_prefix_Elem_B]
        time_mps_add = measure_time(torch.add, A_mps_elem, B_mps_elem, target_device_type="mps")
        speedup_mps_add = time_cpu_add / time_mps_add if time_mps_add > 0 else float('nan')
        row_data_add.append(f"{time_mps_add:.2f}")
        row_data_add.append(f"{speedup_mps_add:.1f}x" if isinstance(speedup_mps_add, float) else "N/A")
    else:
        row_data_add.extend(["N/A", "N/A"])
    table.add_row(row_data_add)

    # --- Поэлементное умножение ---
    row_data_mul = [f"Поэлементное умножение ({d1}x{d2})"]
    time_cpu_mul = measure_time(torch.mul, A_cpu_elem, B_cpu_elem, target_device_type="cpu")
    row_data_mul.append(f"{time_cpu_mul:.2f}")

    if mps_device:
        time_mps_mul = measure_time(torch.mul, A_mps_elem, B_mps_elem, target_device_type="mps")
        speedup_mps_mul = time_cpu_mul / time_mps_mul if time_mps_mul > 0 else float('nan')
        row_data_mul.append(f"{time_mps_mul:.2f}")
        row_data_mul.append(f"{speedup_mps_mul:.1f}x" if isinstance(speedup_mps_mul, float) else "N/A")
    else:
        row_data_mul.extend(["N/A", "N/A"])
    table.add_row(row_data_mul)

    # --- Транспонирование ---
    row_data_transpose = [f"Транспонирование ({d1}x{d2})"]
    time_cpu_transpose = measure_time(torch.transpose, A_cpu, 0, 1, target_device_type="cpu")
    row_data_transpose.append(f"{time_cpu_transpose:.2f}")

    if mps_device:
        A_mps = matrices_mps[key_prefix_A]
        time_mps_transpose = measure_time(torch.transpose, A_mps, 0, 1, target_device_type="mps")
        speedup_mps_transpose = time_cpu_transpose / time_mps_transpose if time_mps_transpose > 0 else float('nan')
        row_data_transpose.append(f"{time_mps_transpose:.2f}")
        row_data_transpose.append(f"{speedup_mps_transpose:.1f}x" if isinstance(speedup_mps_transpose, float) else "N/A")
    else:
        row_data_transpose.extend(["N/A", "N/A"])
    table.add_row(row_data_transpose)

    # --- Сумма элементов ---
    row_data_sum = [f"Сумма элементов ({d1}x{d2})"]
    time_cpu_sum = measure_time(torch.sum, A_cpu_elem, target_device_type="cpu")
    row_data_sum.append(f"{time_cpu_sum:.2f}")

    if mps_device:
        time_mps_sum = measure_time(torch.sum, A_mps_elem, target_device_type="mps")
        speedup_mps_sum = time_cpu_sum / time_mps_sum if time_mps_sum > 0 else float('nan')
        row_data_sum.append(f"{time_mps_sum:.2f}")
        row_data_sum.append(f"{speedup_mps_sum:.1f}x" if isinstance(speedup_mps_sum, float) else "N/A")
    else:
        row_data_sum.extend(["N/A", "N/A"])
    table.add_row(row_data_sum)

    table.add_row(["-"*30, "-"*10, "-"*10, "-"*10])

print(table)
