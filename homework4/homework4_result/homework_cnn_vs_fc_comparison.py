import ssl
import torch
import os
import json

from convolutional_basics.datasets import get_mnist_loaders, get_cifar_loaders
from homework4_result.models.fc_models import SimpleFC, DeepFC_CIFAR
from homework4_result.models.cnn_models import SimpleCNN_MNIST, ResNet_CIFAR
from homework4_result.utils.training_utils import train_model
from homework4_result.utils.visualization_utils import plot_training_curves
from homework4_result.utils.comparison_utils import count_parameters, create_comparison_table

ssl._create_default_https_context = ssl._create_unverified_context


def run_mnist_comparison(device):
    """Запускает сравнение моделей на датасете MNIST."""
    print("--- Запуск сравнения на MNIST ---")
    train_loader, test_loader = get_mnist_loaders()
    
    os.makedirs("results/mnist_comparison", exist_ok=True)
    os.makedirs("plots/mnist_comparison", exist_ok=True)

    fc_model = SimpleFC().to(device)
    print("\nОбучение полносвязной сети на MNIST...")
    fc_history = train_model(fc_model, train_loader, test_loader, epochs=5, device=device)
    
    with open("results/mnist_comparison/fc_mnist_history.json", "w") as f:
        json.dump(fc_history, f)
    plot_training_curves(fc_history, "FC Network on MNIST", save_path="plots/mnist_comparison/fc_mnist_curves.png")

    cnn_model = SimpleCNN_MNIST().to(device)
    print("\nОбучение простой CNN на MNIST...")
    cnn_history = train_model(cnn_model, train_loader, test_loader, epochs=5, device=device)
    
    with open("results/mnist_comparison/cnn_mnist_history.json", "w") as f:
        json.dump(cnn_history, f)
    plot_training_curves(cnn_history, "Simple CNN on MNIST", save_path="plots/mnist_comparison/cnn_mnist_curves.png")
    
    print("\n--- Итоги сравнения на MNIST ---")
    mnist_results = {
        "FC Network (MNIST)": {"final_accuracy": cnn_history['test_accs'][-1], "params": count_parameters(fc_model), "time": fc_history['training_time']},
        "Simple CNN (MNIST)": {"final_accuracy": cnn_history['test_accs'][-1], "params": count_parameters(cnn_model), "time": cnn_history['training_time']}
    }
    create_comparison_table(mnist_results)

def run_cifar_comparison(device):
    """Запускает сравнение моделей на датасете CIFAR-10."""
    print("\n--- Запуск сравнения на CIFAR-10 ---")
    train_loader, test_loader = get_cifar_loaders()

    os.makedirs("results/cifar_comparison", exist_ok=True)
    os.makedirs("plots/cifar_comparison", exist_ok=True)

    fc_model_cifar = DeepFC_CIFAR().to(device)
    print("\nОбучение глубокой FC сети на CIFAR-10...")
    fc_cifar_history = train_model(fc_model_cifar, train_loader, test_loader, epochs=5, device=device)
    
    with open("results/cifar_comparison/fc_cifar_history.json", "w") as f:
        json.dump(fc_cifar_history, f)
    plot_training_curves(fc_cifar_history, "Deep FC on CIFAR-10", save_path="plots/cifar_comparison/fc_cifar_curves.png")

    resnet_cifar = ResNet_CIFAR().to(device)
    print("\nОбучение ResNet на CIFAR-10...")
    resnet_cifar_history = train_model(resnet_cifar, train_loader, test_loader, epochs=5, device=device)
    
    with open("results/cifar_comparison/resnet_cifar_history.json", "w") as f:
        json.dump(resnet_cifar_history, f)
    plot_training_curves(resnet_cifar_history, "ResNet on CIFAR-10", save_path="plots/cifar_comparison/resnet_cifar_curves.png")

    print("\n--- Итоги сравнения на CIFAR-10 ---")
    cifar_results = {
        "Deep FC (CIFAR-10)": {"final_accuracy": fc_cifar_history['test_accs'][-1], "params": count_parameters(fc_model_cifar), "time": fc_cifar_history['training_time']},
        "ResNet (CIFAR-10)": {"final_accuracy": resnet_cifar_history['test_accs'][-1], "params": count_parameters(resnet_cifar), "time": resnet_cifar_history['training_time']}
    }
    create_comparison_table(cifar_results)

if __name__ == '__main__':
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    else:
        DEVICE = torch.device('cpu')
    print(f"Используемое устройство: {DEVICE}")
    
    run_mnist_comparison(DEVICE)
    run_cifar_comparison(DEVICE)