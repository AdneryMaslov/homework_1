import sys
import os
import ssl

from .utils.experiment_utils import run_experiment
from fully_connected_basics.datasets import get_mnist_loaders

ssl._create_default_https_context = ssl._create_unverified_context
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


def depth_experiments(train_loader, test_loader, epochs=10):
    """Проводит эксперименты с разной глубиной сети"""
    base_hidden_size = 128
    
    configs = []
    
    # 1. Линейный классификатор (1 слой)
    configs.append({
        'name': 'depth_1_layer', 'experiment_type': 'depth_experiments',
        'input_size': 784, 'num_classes': 10, 'layers': []})
    
    # 2. Сеть с 1 скрытым слоем (2 слоя)
    configs.append({
        'name': 'depth_2_layers', 'experiment_type': 'depth_experiments',
        'input_size': 784, 'num_classes': 10,
        'layers': [{'type': 'linear', 'size': base_hidden_size}, {'type': 'relu'}]})
    
    # 3. Сеть с 2 скрытыми слоями (3 слоя)
    configs.append({
        'name': 'depth_3_layers', 'experiment_type': 'depth_experiments',
        'input_size': 784, 'num_classes': 10,
        'layers': [
            {'type': 'linear', 'size': base_hidden_size}, {'type': 'relu'},
            {'type': 'linear', 'size': base_hidden_size}, {'type': 'relu'}]})
    
    # 4. Сеть с 4 скрытыми слоями (5 слоев)
    configs.append({
        'name': 'depth_5_layers', 'experiment_type': 'depth_experiments',
        'input_size': 784, 'num_classes': 10,
        'layers': [
            {'type': 'linear', 'size': base_hidden_size}, {'type': 'relu'},
            {'type': 'linear', 'size': base_hidden_size}, {'type': 'relu'},
            {'type': 'linear', 'size': base_hidden_size}, {'type': 'relu'},
            {'type': 'linear', 'size': base_hidden_size}, {'type': 'relu'}]})
    
    # 5. Сеть с 6 скрытыми слоями (7 слоев)
    configs.append({
        'name': 'depth_7_layers', 'experiment_type': 'depth_experiments',
        'input_size': 784, 'num_classes': 10,
        'layers': [
            {'type': 'linear', 'size': base_hidden_size}, {'type': 'relu'},
            {'type': 'linear', 'size': base_hidden_size}, {'type': 'relu'},
            {'type': 'linear', 'size': base_hidden_size}, {'type': 'relu'},
            {'type': 'linear', 'size': base_hidden_size}, {'type': 'relu'},
            {'type': 'linear', 'size': base_hidden_size}, {'type': 'relu'},
            {'type': 'linear', 'size': base_hidden_size}, {'type': 'relu'}]})

    # Эксперимент с Dropout и BatchNorm
    configs.append({
        'name': 'depth_5_layers_regularized', 'experiment_type': 'depth_experiments',
        'input_size': 784, 'num_classes': 10,
        'layers': [
            {'type': 'linear', 'size': base_hidden_size}, {'type': 'batch_norm'}, {'type': 'relu'}, {'type': 'dropout', 'rate': 0.3},
            {'type': 'linear', 'size': base_hidden_size}, {'type': 'batch_norm'}, {'type': 'relu'}, {'type': 'dropout', 'rate': 0.3},
            {'type': 'linear', 'size': base_hidden_size}, {'type': 'batch_norm'}, {'type': 'relu'}, {'type': 'dropout', 'rate': 0.3},
            {'type': 'linear', 'size': base_hidden_size}, {'type': 'batch_norm'}, {'type': 'relu'}]})

    all_results = []
    for config in configs:
        results = run_experiment(config, train_loader, test_loader, epochs=epochs)
        all_results.append(results)
        
    print("\n--- Depth Experiment Summary ---")
    for res in all_results:
        print(f"Model: {res['config']['name']}, Final Test Acc: {res['final_test_accuracy']:.4f}, Time: {res['training_time']:.2f}s, Params: {res['num_parameters']}")
    
if __name__ == '__main__':
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    depth_experiments(train_loader, test_loader, epochs=10)