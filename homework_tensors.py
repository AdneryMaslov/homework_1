import torch

# 1.1 Создание тензоров
print("\n-- 1.1 Создание тензоров: --\n")

# Тензор размером 3x4, заполненный случайными числами от 0 до 1
tensor_random = torch.rand(3, 4)
print(f"Тензор 3x4 случайных чисел:\n{tensor_random}\n")

# Тензор размером 2x3x4, заполненный нулями
tensor_zeros = torch.zeros(2, 3, 4)
print(f"Тензор 2x3x4 из нулей:\n{tensor_zeros}\n")

# Тензор размером 5x5, заполненный единицами
tensor_ones = torch.ones(5, 5)
print(f"Тензор 5x5 из единиц:\n{tensor_ones}\n")

# Тензор размером 4x4 с числами от 0 до 15
tensor_range_reshaped = torch.arange(16).reshape(4, 4)
print(f"Тензор 4x4 с числами от 0 до 15:\n{tensor_range_reshaped}\n")


# 1.2 Операции с тензорами
print("\n\n-- 1.2 Операции с тензорами: --\n")

# Тензор A размером 3x4, B размером 4x3
A = torch.randn(3, 4)
B = torch.randn(4, 3)
print(f"Тензор A:\n{A}\n")
print(f"Тензор B:\n{B}\n")

# Транспонирование тензора A
transpon_A = A.T
print(f"Транспонированный тензор A:\n{transpon_A}\n")

# Матричное умножение A и B c проверкой размерностей для умножения:
if A.shape[1] == B.shape[0]:
    matmul_result = torch.matmul(A, B)
    print(f"Матричное умножение A*B:\n{matmul_result}\n")
else:
    print("Ошибка: Несовместимые размерности\n")

# Поэлементное умножение A и транспонированного B
transpon_B = B.T
if A.shape == transpon_B.shape:
    elementwise_mul = A * transpon_B
    print(f"Поэлементное умножение A * B.T:\n{elementwise_mul}\n")
else:
    print("Ошибка: Несовместимые размерности\n")

# Вычислите сумму всех элементов тензора A (item для скалярного знач)
sum_A = torch.sum(A)
print(f"Сумма всех элементов A: {sum_A.item()}\n")


# 1.3 Индексация и срезы
print("\n\n-- 1.3 Индексация и срезы: --\n")

tensor_5x5x5 = torch.arange(125).reshape(5, 5, 5)
print(f"Исходный тензор 5x5x5:\n{tensor_5x5x5}\n")

# Извлеките первую строку
first_row = tensor_5x5x5[0, 0, :]
print(f"Первая строка: {first_row}\n")

# Извлеките последний столбец
last_column = tensor_5x5x5[0, :, -1]
print(f"Последний столбец: {last_column}\n")

# Подматрицу размером 2x2 из центра тензора
center_submatrix_2d = tensor_5x5x5[2, 1:3, 1:3]
print(f"Подматрица 2x2 из центра:\n{center_submatrix_2d}\n")

# Все элементы с четными индексами (по всем измерениям)
even_indexed_elements = tensor_5x5x5[::2, ::2, ::2]
print(f"Элементы с четными индексами:\n{even_indexed_elements}\n")


# 1.4 Работа с формами
print("\n\n-- 1.4 Работа с формами: --\n")

# тензор размером 24 элемента
tensor_24_elements = torch.arange(24)
print(f"Исходный тензор с 24 элементами:\n{tensor_24_elements}\n")

# в 2x12
reshaped_2x12 = tensor_24_elements.reshape(2, 12)
print(f"В 2x12:\n{reshaped_2x12}\n")

# в 3x8
reshaped_3x8 = tensor_24_elements.reshape(3, 8)
print(f"В 3x8:\n{reshaped_3x8}\n")

# в 4x6
reshaped_4x6 = tensor_24_elements.reshape(4, 6)
print(f"В 4x6:\n{reshaped_4x6}\n")

# в 2x3x4
reshaped_2x3x4 = tensor_24_elements.reshape(2, 3, 4)
print(f"В 2x3x4:\n{reshaped_2x3x4}\n")

# в 2x2x2x3
reshaped_2x2x2x3 = tensor_24_elements.reshape(2, 2, 2, 3)
print(f"В 2x2x2x3:\n{reshaped_2x2x2x3}\n")
