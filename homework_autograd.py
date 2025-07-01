import torch

# 2.1 Простые вычисления с градиентами
print("\n-- 2.1 Простые вычисления с градиентами: --")

# Создайте тензоры x, y, z с requires_grad=True
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(4.0, requires_grad=True)
print(f"Исходные значения: x={x.item()}, y={y.item()}, z={z.item()}")

# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = x**2 + y**2 + z**2 + 2 * x * y * z
print(f"Значение функции: {f.item()}")

# Найдите градиенты по всем переменным
f.backward()
print(f"Градиент df/dx: {x.grad.item()}")
print(f"Градиент df/dy: {y.grad.item()}")
print(f"Градиент df/dz: {z.grad.item()}")

# Проверьте результат аналитически:
# df/dx = 2x + 2yz
# df/dy = 2y + 2xz
# df/dz = 2z + 2xy
# При x=2, y=3, z=4:
# df/dx = 2*2 + 2*3*4 = 4 + 24 = 28
# df/dy = 2*3 + 2*2*4 = 6 + 16 = 22
# df/dz = 2*4 + 2*2*3 = 8 + 12 = 20
print(f"Аналитический df/dx: {2*x.item() + 2*y.item()*z.item()}")
print(f"Аналитический df/dy: {2*y.item() + 2*x.item()*z.item()}")
print(f"Аналитический df/dz: {2*z.item() + 2*x.item()*y.item()}")


# 2.2 Градиент функции потерь
print("\n-- 2.2 Градиент функции потерь: --")

# Реализуйте функцию MSE (Mean Squared Error):
# MSE = (1/n) * Σ(y_pred - y_true)^2
# где y_pred = w * x_data + b (линейная функция)
x_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = torch.tensor([2.0, 4.0, 5.0, 4.0])

# Параметры модели, требующие градиентов
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
print(f"w={w.item()}, b={b.item()}")

# Вычисление предсказаний
y_pred = w * x_data + b
print(f"Предсказания y_pred: {y_pred}")

# Вычисление MSE
n = y_true.numel()
mse = torch.sum((y_pred - y_true)**2) / n
print(f"MSE: {mse.item()}")

# Найдите градиенты по w и b
mse.backward()
print(f"Градиент по w: {w.grad.item()}")
print(f"Градиент по b: {b.grad.item()}")


# 2.3 Цепное правило
print("\n-- 2.3 Цепное правило: --")

# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
x_chain = torch.tensor(torch.pi / 2, requires_grad=True)
print(f"Исходное x: {x_chain.item()}")

# sin(x^2 + 1)
y_chain = x_chain**2 + 1
f_chain = torch.sin(y_chain)
print(f"f(x) = sin(x^2 + 1): {f_chain.item()}")

# Найдите градиент df/dx
f_chain.backward()
print(f"Градиент df/dx: {x_chain.grad.item()}")

# Проверьте результат с помощью torch.autograd.grad
x_chain_new = torch.tensor(torch.pi / 2, requires_grad=True)
f_chain_new = torch.sin(x_chain_new**2 + 1)
grad_torch_autograd = torch.autograd.grad(f_chain_new, x_chain_new)[0]
print(f"Градиент df/dx (torch.autograd.grad): {grad_torch_autograd.item()}")
