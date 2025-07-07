import ssl
import torch
import os
import json

from convolutional_basics.datasets import get_mnist_loaders
from homework4_result.models.cnn_models import KernelSizeCNN, DepthCNN
from homework4_result.utils.training_utils import train_model
from homework4_result.utils.visualization_utils import plot_training_curves
from homework4_result.utils.comparison_utils import count_parameters, create_comparison_table

ssl._create_default_https_context = ssl._create_unverified_context


def analyze_kernel_size(device):
    """Исследует влияние размера ядра свертки."""
    print("\n--- Анализ влияния размера ядра ---")
    train_loader, test_loader = get_mnist_loaders()
    
    kernel_sizes = [3, 5, 7]
    results = {}
    
    os.makedirs("results/architecture_analysis", exist_ok=True)
    os.makedirs("plots/architecture_analysis", exist_ok=True)

    for size in kernel_sizes:
        model_name = f"Kernel_{size}x{size}"
        print(f"\nTraining model: {model_name}")
        model = KernelSizeCNN(kernel_size=size).to(device)
        history = train_model(model, train_loader, test_loader, epochs=5, device=device)
        
        with open(f"results/architecture_analysis/{model_name}.json", "w") as f:
            json.dump(history, f)
        plot_training_curves(history, model_name, save_path=f"plots/architecture_analysis/{model_name}_curves.png")
        
        results[model_name] = {
            "final_accuracy": history['test_accs'][-1],
            "params": count_parameters(model),
            "time": history['training_time']
        }
    
    print("\n--- Итоги анализа размера ядра ---")
    create_comparison_table(results)

def analyze_cnn_depth(device):
    """Исследует влияние глубины CNN."""
    print("\n--- Анализ влияния глубины CNN ---")
    train_loader, test_loader = get_mnist_loaders()
    
    depths = [2, 4, 6]
    results = {}

    for depth in depths:
        model_name = f"Depth_{depth}_layers"
        print(f"\nTraining model: {model_name}")
        model = DepthCNN(num_conv_layers=depth).to(device)
        history = train_model(model, train_loader, test_loader, epochs=5, device=device)
        
        with open(f"results/architecture_analysis/{model_name}.json", "w") as f:
            json.dump(history, f)
        plot_training_curves(history, model_name, save_path=f"plots/architecture_analysis/{model_name}_curves.png")
        
        results[model_name] = {
            "final_accuracy": history['test_accs'][-1],
            "params": count_parameters(model),
            "time": history['training_time']
        }
        
    print("\n--- Итоги анализа глубины ---")
    create_comparison_table(results)

if __name__ == '__main__':
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    else:
        DEVICE = torch.device('cpu')
    print(f"Используемое устройство: {DEVICE}")
    
    analyze_kernel_size(DEVICE)
    analyze_cnn_depth(DEVICE)