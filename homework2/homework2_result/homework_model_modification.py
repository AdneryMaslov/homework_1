import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression, make_classification
import logging
import numpy as np
import os

# --- ИЗМЕНЕНО: Начало блока для определения путей ---
# Определяем абсолютный путь к папке, где лежит скрипт
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
# --- ИЗМЕНЕНО: Конец блока ---

from utils_hw import get_device, log_epoch, calculate_multiclass_metrics, plot_confusion_matrix, split_data, plot_training_history

class EarlyStopping:
    """Останавливает обучение, если потери на валидации не улучшаются."""
    # --- ИЗМЕНЕНО: Указываем полный путь по умолчанию ---
    def __init__(self, patience=7, verbose=False, delta=0, path=os.path.join(MODELS_DIR, 'checkpoint.pt')):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'Счетчик EarlyStopping: {self.counter} из {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Сохраняет модель при уменьшении потерь на валидации."""
        if self.verbose:
            logging.info(f'Потери на валидации уменьшились ({self.val_loss_min:.6f} --> {val_loss:.6f}). Сохранение модели...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

def train_linear_regression_with_modifications(X_train, y_train, X_val, y_val, in_features):
    """Обучает модель линейной регрессии с L1/L2 регуляризацией и ранней остановкой."""
    logging.info("--- Обучение модифицированной линейной регрессии ---")
    device = get_device()
    model = LinearRegression(in_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    l1_lambda, l2_lambda = 0.01, 0.01

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    epochs = 200
    # --- ИЗМЕНЕНО: Указываем полный путь для сохранения ---
    checkpoint_path = os.path.join(MODELS_DIR, 'linear_reg_best.pt')
    early_stopping = EarlyStopping(patience=10, verbose=True, path=checkpoint_path)
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            mse_loss = criterion(outputs, batch_y)
            l1_reg = sum(p.abs().sum() for p in model.parameters())
            l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = mse_loss + l1_lambda * l1_reg + l2_lambda * l2_reg
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        log_epoch(epoch, epochs, avg_train_loss, {'val_loss': avg_val_loss})

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            logging.info("Сработала ранняя остановка.")
            break
            
    model.load_state_dict(torch.load(checkpoint_path))
    logging.info(f"Загружена лучшая модель из {checkpoint_path}.")
    return model, history

class SoftmaxRegression(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)

def train_softmax_regression(X_train, y_train, X_val, y_val, in_features, num_classes):
    """Обучает модель Softmax регрессии и оценивает ее с помощью нескольких метрик."""
    logging.info("--- Обучение модифицированной логистической (Softmax) регрессии ---")
    device = get_device()
    model = SoftmaxRegression(in_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    epochs = 100
    history = {'train_loss': [], 'val_loss': [], 'val_f1_score': [], 'val_roc_auc': []}
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        total_val_loss, all_val_preds_labels, all_val_preds_probs, all_val_true = 0, [], [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                all_val_preds_labels.extend(preds.cpu().numpy())
                all_val_preds_probs.extend(probs.cpu().numpy())
                all_val_true.extend(batch_y.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_metrics = calculate_multiclass_metrics(np.array(all_val_true), np.array(all_val_preds_probs), np.array(all_val_preds_labels))
        history['val_loss'].append(avg_val_loss)
        history['val_f1_score'].append(val_metrics['f1_score'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        log_epoch(epoch, epochs, avg_train_loss, {'val_loss': avg_val_loss, **val_metrics})

    return model, history, all_val_true, all_val_preds_labels

if __name__ == '__main__':
    # --- ИЗМЕНЕНО: Создаем папки по полному пути ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    logging.info("--- Запуск демонстрации на синтетических данных ---")
    
    # Линейная регрессия
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
    X_reg_t = torch.tensor(X_reg, dtype=torch.float32)
    y_reg_t = torch.tensor(y_reg, dtype=torch.float32).view(-1, 1)
    X_reg_train, X_reg_val, _, y_reg_train, y_reg_val, _ = split_data(X_reg_t, y_reg_t)
    lin_model, lin_history = train_linear_regression_with_modifications(
        X_reg_train, y_reg_train, X_reg_val, y_reg_val, in_features=X_reg.shape[1]
    )
    # --- ИЗМЕНЕНО: Указываем полный путь для сохранения графика ---
    plot_training_history(lin_history, save_path=os.path.join(PLOTS_DIR, "linear_regression_history.png"))
    
    # Логистическая регрессия
    X_cls, y_cls = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=0, n_classes=4, random_state=42)
    X_cls_t = torch.tensor(X_cls, dtype=torch.float32)
    y_cls_t = torch.tensor(y_cls, dtype=torch.long)
    num_classes = len(torch.unique(y_cls_t))
    X_cls_train, X_cls_val, _, y_cls_train, y_cls_val, _ = split_data(X_cls_t, y_cls_t)
    log_model, log_history, y_true, y_pred = train_softmax_regression(
        X_cls_train, y_cls_train, X_cls_val, y_cls_val, in_features=X_cls.shape[1], num_classes=num_classes
    )
    # --- ИЗМЕНЕНО: Указываем полные пути для сохранения графиков ---
    plot_training_history(log_history, save_path=os.path.join(PLOTS_DIR, "softmax_regression_history.png"))
    class_names = [f'Класс {i}' for i in range(num_classes)]
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=os.path.join(PLOTS_DIR, "confusion_matrix.png"))