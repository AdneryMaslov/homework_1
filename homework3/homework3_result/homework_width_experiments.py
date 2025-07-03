import sys
import os
import ssl
import pandas as pd

from .utils.experiment_utils import run_experiment
from .utils.visualization_utils import plot_heatmap
from fully_connected_basics.datasets import get_mnist_loaders

ssl._create_default_https_context = ssl._create_unverified_context
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


def width_experiments(train_loader, test_loader, epochs=10):
    """Проводит эксперименты с разной шириной сети."""
    
    width_configs = {
        'narrow': [64, 32, 16],
        'medium': [256, 128, 64],
        'wide': [1024, 512, 256],
        'very_wide': [2048, 1024, 512]
    }
    
    configs = []
    for name, sizes in width_configs.items():
        configs.append({
            'name': f'width_{name}', 'experiment_type': 'width_experiments',
            'input_size': 784, 'num_classes': 10,
            'layers': [
                {'type': 'linear', 'size': sizes[0]}, {'type': 'relu'},
                {'type': 'linear', 'size': sizes[1]}, {'type': 'relu'},
                {'type': 'linear', 'size': sizes[2]}, {'type': 'relu'}]})
        
    all_results = []
    for config in configs:
        results = run_experiment(config, train_loader, test_loader, epochs=epochs)
        all_results.append(results)
    
    print("\n--- Width Experiment Summary ---")
    for res in all_results:
        print(f"Model: {res['config']['name']}, Final Test Acc: {res['final_test_accuracy']:.4f}, Time: {res['training_time']:.2f}s, Params: {res['num_parameters']}")
        
    # упрощенный пример Grid Search для оптимальной архитектуры 
    print("\n--- Grid Search Simulation ---")
    grid_results = []
    schemes = {
        "expanding": [128, 256, 512],
        "contracting": [512, 256, 128],
        "constant": [256, 256, 256]
    }
    
    for name, sizes in schemes.items():
        config = {
            'name': f'grid_{name}', 'experiment_type': 'width_experiments',
            'input_size': 784, 'num_classes': 10,
            'layers': [
                {'type': 'linear', 'size': sizes[0]}, {'type': 'relu'},
                {'type': 'linear', 'size': sizes[1]}, {'type': 'relu'},
                {'type': 'linear', 'size': sizes[2]}, {'type': 'relu'}
            ]
        }
        results = run_experiment(config, train_loader, test_loader, epochs=5)
        grid_results.append(results)
        
    # Визуализация результатов Grid Search в виде heatmap
    heatmap_data = {
        'Scheme': [res['config']['name'] for res in grid_results],
        'Accuracy': [res['final_test_accuracy'] for res in grid_results]
    }
    df = pd.DataFrame(heatmap_data).set_index('Scheme')
    plot_heatmap(df, 'Grid Search Results: Accuracy by Scheme', 'plots/width_experiments/grid_search_heatmap.png')


if __name__ == '__main__':
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    width_experiments(train_loader, test_loader, epochs=10)