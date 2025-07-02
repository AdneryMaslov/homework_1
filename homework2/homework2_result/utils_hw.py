import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_device():
    """Возвращает наиболее подходящее доступное устройство (CUDA или CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def log_epoch(epoch, epochs, loss, metrics=None):
    """Логирует метрики для данной эпохи."""
    msg = f"Эпоха {epoch+1}/{epochs} - Потери: {loss:.4f}"
    if metrics:
        for k, v in metrics.items():
            msg += f" - {k}: {v:.4f}"
    logging.info(msg)

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Разделяет данные на обучающую, валидационную и тестовую выборки."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def calculate_multiclass_metrics(y_true, y_pred_probs, y_pred_labels):
    """Вычисляет и возвращает метрики для многоклассовой классификации."""
    metrics = {
        'precision': precision_score(y_true, y_pred_labels, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred_labels, average='macro', zero_division=0),
        'f1_score': f1_score(y_true, y_pred_labels, average='macro', zero_division=0)
    }
    # ROC AUC для бинарной классификации
    if len(np.unique(y_true)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_probs[:, 1])
    # ROC AUC для многоклассовой классификации
    else:
         metrics['roc_auc'] = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')
    return metrics

def plot_confusion_matrix(y_true, y_pred_labels, class_names, save_path=None):
    """Строит и сохраняет матрицу ошибок."""
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Предсказанная метка')
    plt.ylabel('Истинная метка')
    plt.title('Матрица ошибок (Confusion Matrix)')
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Матрица ошибок сохранена в {save_path}")
    plt.show()

def plot_training_history(history, save_path=None):
    """Строит графики истории обучения и валидации."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(history['train_loss'], 'b-', label='Потери на обучении')
    ax1.plot(history['val_loss'], 'r-', label='Потери на валидации')
    ax1.set_xlabel('Эпохи')
    ax1.set_ylabel('Потери', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.legend(loc='upper left')

    if len(history.keys()) > 2:
        ax2 = ax1.twinx()
        metrics_to_plot = [k for k in history.keys() if 'loss' not in k and 'val' in k]
        colors = plt.cm.viridis(np.linspace(0, 1, len(metrics_to_plot)))
        for i, metric in enumerate(metrics_to_plot):
             ax2.plot(history[metric], linestyle='--', color=colors[i], label=f'Валидация {metric.replace("val_", "").capitalize()}')
        ax2.set_ylabel('Метрики', color='k')
        ax2.tick_params(axis='y', labelcolor='k')
        ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title('История обучения')
    if save_path:
        plt.savefig(save_path)
        logging.info(f"График истории обучения сохранен в {save_path}")
    plt.show()