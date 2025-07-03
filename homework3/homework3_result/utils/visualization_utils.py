import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd

def plot_training_history(history, title, save_path=None):
    """Визуализирует историю обучения (потери и точность)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16)

    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss vs. Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_accs'], label='Train Accuracy')
    ax2.plot(history['test_accs'], label='Test Accuracy')
    ax2.set_title('Accuracy vs. Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()

def plot_heatmap(data, title, save_path=None):
    """Визуализирует данные в виде тепловой карты"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt=".4f", cmap="viridis")
    plt.title(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()