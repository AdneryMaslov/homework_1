import ssl
import torch
import os
import json

from convolutional_basics.datasets import get_cifar_loaders
from homework4_result.models.cnn_models import GenericResNet, ResidualBlock, BottleneckBlock, WideResidualBlock
from homework4_result.models.custom_layers import AttentionMechanism
from homework4_result.utils.training_utils import train_model
from homework4_result.utils.visualization_utils import plot_training_curves
from homework4_result.utils.comparison_utils import count_parameters, create_comparison_table

ssl._create_default_https_context = ssl._create_unverified_context


def test_custom_layers(device):
    """Тестирует интеграцию кастомного слоя (Attention)."""
    print("\n--- Тестирование кастомного слоя Attention ---")
    train_loader, test_loader = get_cifar_loaders()

    # базовая модель
    base_model = GenericResNet(ResidualBlock, [2, 2, 2]).to(device)
    print(f"\nTraining Base ResNet...")
    base_history = train_model(base_model, train_loader, test_loader, epochs=5, device=device)
    
    # модель с кастомным слоем
    attention_model = GenericResNet(ResidualBlock, [2, 2, 2]).to(device)
    output_channels_of_layer2 = 32
    attention_module = AttentionMechanism(channels=output_channels_of_layer2).to(device)
    attention_model.layer2.add_module("attention", attention_module)

    print(f"\nTraining ResNet with Attention...")
    attention_history = train_model(attention_model, train_loader, test_loader, epochs=5, device=device)

    # сравнение
    print("\n--- Итоги сравнения кастомного слоя ---")
    results = {
        "Base ResNet": {"final_accuracy": base_history['test_accs'][-1], "params": count_parameters(base_model), "time": base_history['training_time']},
        "ResNet + Attention": {"final_accuracy": attention_history['test_accs'][-1], "params": count_parameters(attention_model), "time": attention_history['training_time']}
    }
    create_comparison_table(results)

def compare_residual_blocks(device):
    """Сравнивает различные типы Residual блоков."""
    print("\n--- Сравнение Residual блоков ---")
    train_loader, test_loader = get_cifar_loaders()
    results = {}
    
    block_configs = {
        "Basic_ResNet": {"block": ResidualBlock, "num_blocks": [2, 2, 2]},
        "Bottleneck_ResNet": {"block": BottleneckBlock, "num_blocks": [2, 2, 2]},
        "Wide_ResNet": {"block": WideResidualBlock, "num_blocks": [2, 2, 2], "width_factor": 2}
    }

    os.makedirs("results/residual_blocks", exist_ok=True)
    os.makedirs("plots/residual_blocks", exist_ok=True)

    for name, config in block_configs.items():
        print(f"\nTraining model: {name}")
        model = GenericResNet(
            block=config["block"], 
            num_blocks=config["num_blocks"],
            width_factor=config.get("width_factor", 1)
        ).to(device)
        
        history = train_model(model, train_loader, test_loader, epochs=5, device=device)
        
        with open(f"results/residual_blocks/{name}.json", "w") as f:
            json.dump(history, f)
        plot_training_curves(history, name, save_path=f"plots/residual_blocks/{name}_curves.png")
        
        results[name] = {
            "final_accuracy": history['test_accs'][-1],
            "params": count_parameters(model),
            "time": history['training_time']
        }
        
    print("\n--- Итоги сравнения Residual блоков ---")
    create_comparison_table(results)

if __name__ == '__main__':
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    else:
        DEVICE = torch.device('cpu')
    print(f"Используемое устройство: {DEVICE}")
    
    test_custom_layers(DEVICE)
    compare_residual_blocks(DEVICE)