import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

from utils_hw import get_device, split_data
from homework_model_modification import LinearRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_model_for_experiment(X_train, y_train, X_val, y_val, in_features, optimizer_name='SGD', lr=0.01, batch_size=32):
    """Общая функция обучения для экспериментов с гиперпараметрами."""
    device = get_device()
    model = LinearRegression(in_features).to(device)
    criterion = nn.MSELoss()
    
    optimizer_classes = {'sgd': optim.SGD, 'adam': optim.Adam, 'rmsprop': optim.RMSprop}
    optimizer = optimizer_classes[optimizer_name.lower()](model.parameters(), lr=lr)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    epochs = 50
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    final_val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            final_val_loss += loss.item()

    return final_val_loss / len(val_loader)

def run_hyperparameter_experiments(X, y):
    """Проводит эксперименты с разными оптимизаторами, скоростями обучения и размерами батчей."""
    logging.info("--- Запуск экспериментов с гиперпараметрами ---")
    X_train, X_val, _, y_train, y_val, _ = split_data(X, y, test_size=0.2, val_size=0.2)
    in_features = X.shape[1]

    optimizers = ['SGD', 'Adam', 'RMSprop']
    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [16, 32, 64]
    
    results = []
    
    for opt_name in optimizers:
        for lr in learning_rates:
            for bs in batch_sizes:
                logging.info(f"Обучение с: Оптимизатор={opt_name}, LR={lr}, Размер батча={bs}")
                val_loss = train_model_for_experiment(X_train, y_train, X_val, y_val, in_features, opt_name, lr, bs)
                results.append({'optimizer': opt_name, 'lr': lr, 'batch_size': bs, 'val_loss': val_loss})
    
    results_df = pd.DataFrame(results)
    logging.info("Результаты экспериментов с гиперпараметрами:\n" + str(results_df))
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='lr', y='val_loss', hue='optimizer', style='batch_size', markers=True, dashes=False)
    plt.xscale('log')
    plt.title('Результаты подбора гиперпараметров')
    plt.xlabel('Скорость обучения (Log Scale)')
    plt.ylabel('Финальные потери на валидации')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "hyperparameter_experiments.png"))
    plt.close()

def run_feature_engineering_experiment(X, y):
    """Сравнивает базовую модель с моделью, обученной на сконструированных признаках."""
    logging.info("--- Запуск эксперимента по инжинирингу признаков ---")
    
    # Базовая модель
    logging.info("Обучение базовой модели...")
    X_train, X_val, _, y_train, y_val, _ = split_data(X, y)
    baseline_loss = train_model_for_experiment(X_train, y_train, X_val, y_val, in_features=X.shape[1], optimizer_name='Adam', lr=0.01, batch_size=32)
    logging.info(f"Потери базовой модели на валидации: {baseline_loss:.4f}")

    # Инжиниринг признаков
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X.cpu().numpy())
    X_poly_t = torch.tensor(X_poly, dtype=torch.float32)
    
    logging.info(f"Исходное кол-во признаков: {X.shape[1]}, Новое кол-во признаков: {X_poly.shape[1]}")
    
    # Модель с новыми признаками
    logging.info("Обучение модели с новыми признаками...")
    X_poly_train, X_poly_val, _, y_poly_train, y_poly_val, _ = split_data(X_poly_t, y)
    engineered_loss = train_model_for_experiment(X_poly_train, y_poly_train, X_poly_val, y_poly_val, in_features=X_poly.shape[1], optimizer_name='Adam', lr=0.01, batch_size=32)
    logging.info(f"Потери модели с новыми признаками на валидации: {engineered_loss:.4f}")

    # Сравнение результатов
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Базовая модель', 'Модель с полиномиальными признаками'], y=[baseline_loss, engineered_loss])
    plt.ylabel('Финальные потери на валидации')
    plt.title('Сравнение производительности моделей')
    plt.savefig(os.path.join(PLOTS_DIR, "feature_engineering_comparison.png"))
    plt.close()

if __name__ == '__main__':
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Генерация синтетических данных для экспериментов
    X_exp, y_exp = make_regression(n_samples=1000, n_features=5, noise=15, random_state=42)
    X_exp_t = torch.tensor(X_exp, dtype=torch.float32)
    y_exp_t = torch.tensor(y_exp, dtype=torch.float32).view(-1, 1)

    # Запуск экспериментов
    run_hyperparameter_experiments(X_exp_t, y_exp_t)
    print("\n" + "="*50 + "\n")
    run_feature_engineering_experiment(X_exp_t, y_exp_t)