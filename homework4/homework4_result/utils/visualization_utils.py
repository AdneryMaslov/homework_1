import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import math

def plot_training_curves(history, title="Кривые обучения", save_path=None):
    """
    Визуализирует и опционально сохраняет кривые потерь и точности.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    # График потерь (Loss)
    ax1.plot(history.get('train_losses', []), label='Потери на обучении (Train Loss)')
    ax1.plot(history.get('test_losses', []), label='Потери на тесте (Test Loss)')
    ax1.set_title('Потери vs. Эпохи')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Потери')
    ax1.legend()
    ax1.grid(True)

    # График точности (Accuracy)
    ax2.plot(history.get('train_accs', []), label='Точность на обучении (Train Accuracy)')
    ax2.plot(history.get('test_accs', []), label='Точность на тесте (Test Accuracy)')
    ax2.set_title('Точность vs. Эпохи')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Точность')
    ax2.legend()
    ax2.grid(True)

    # Сохранение графика, если указан путь
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"График сохранен в: {save_path}")

    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Визуализирует матрицу ошибок (confusion matrix).
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Матрица ошибок')
    plt.ylabel('Истинная метка')
    plt.xlabel('Предсказанная метка')
    plt.show()

def visualize_feature_maps(feature_maps, title="Карты признаков", num_cols=8):
    """
    Визуализирует карты признаков из сверточного слоя.
    
    Аргументы:
        feature_maps (torch.Tensor): Тензор с картами признаков для одного изображения.
                                     Ожидаемая форма: (1, кол-во_каналов, высота, ширина).
    """
    # Переводим тензор на CPU и отсоединяем от графа вычислений
    feature_maps = feature_maps.detach().cpu().squeeze(0)
    
    num_maps = feature_maps.shape[0]
    num_rows = math.ceil(num_maps / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    fig.suptitle(title, fontsize=16)

    # "Выравниваем" массив осей для удобной итерации
    if num_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes] if num_maps == 1 else axes

    for i in range(num_maps):
        ax = axes[i]
        ax.imshow(feature_maps[i], cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Карта {i+1}')
        
    # Скрываем неиспользуемые графики
    for j in range(num_maps, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()