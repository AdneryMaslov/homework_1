import sys
import os
import ssl

from .utils.experiment_utils import run_experiment
from fully_connected_basics.datasets import get_mnist_loaders

ssl._create_default_https_context = ssl._create_unverified_context
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


def regularization_experiments(train_loader, test_loader, epochs=15):
    """Проводит эксперименты с техниками регуляризации."""
    
    # Архитектура, склонная к переобучению
    base_architecture = [
        {'type': 'linear', 'size': 512}, {'type': 'relu'},
        {'type': 'linear', 'size': 256}, {'type': 'relu'},
        {'type': 'linear', 'size': 128}, {'type': 'relu'}
    ]
    
    configs = []
    
    # 1. Без регуляризации
    configs.append({
        'name': 'reg_none', 'experiment_type': 'regularization_experiments',
        'input_size': 784, 'num_classes': 10, 'layers': base_architecture,
        'weight_decay': 0.0
    })
    
    # 2. Dropout
    for rate in [0.1, 0.3, 0.5]:
        configs.append({
            'name': f'reg_dropout_{rate}', 'experiment_type': 'regularization_experiments',
            'input_size': 784, 'num_classes': 10,
            'layers': [
                {'type': 'linear', 'size': 512}, {'type': 'relu'}, {'type': 'dropout', 'rate': rate},
                {'type': 'linear', 'size': 256}, {'type': 'relu'}, {'type': 'dropout', 'rate': rate},
                {'type': 'linear', 'size': 128}, {'type': 'relu'}
            ],
            'weight_decay': 0.0})

    # 3. BatchNorm
    configs.append({
        'name': 'reg_batchnorm', 'experiment_type': 'regularization_experiments',
        'input_size': 784, 'num_classes': 10,
        'layers': [
            {'type': 'linear', 'size': 512}, {'type': 'batch_norm'}, {'type': 'relu'},
            {'type': 'linear', 'size': 256}, {'type': 'batch_norm'}, {'type': 'relu'},
            {'type': 'linear', 'size': 128}, {'type': 'batch_norm'}, {'type': 'relu'}
        ],
        'weight_decay': 0.0})

    # 4. Dropout + BatchNorm
    configs.append({
        'name': 'reg_dropout_batchnorm', 'experiment_type': 'regularization_experiments',
        'input_size': 784, 'num_classes': 10,
        'layers': [
            {'type': 'linear', 'size': 512}, {'type': 'batch_norm'}, {'type': 'relu'}, {'type': 'dropout', 'rate': 0.3},
            {'type': 'linear', 'size': 256}, {'type': 'batch_norm'}, {'type': 'relu'}, {'type': 'dropout', 'rate': 0.3},
            {'type': 'linear', 'size': 128}, {'type': 'batch_norm'}, {'type': 'relu'}
        ],
        'weight_decay': 0.0})
    
    # 5. L2 регуляризация (weight_decay)
    configs.append({
        'name': 'reg_l2', 'experiment_type': 'regularization_experiments',
        'input_size': 784, 'num_classes': 10, 'layers': base_architecture,
        'weight_decay': 1e-4
    })

    all_results = []
    for config in configs:
        results = run_experiment(
            config, 
            train_loader, 
            test_loader, 
            epochs=epochs, 
            weight_decay=config.get('weight_decay', 0.0)
        )
        all_results.append(results)
        
    print("\n--- Regularization Experiment Summary ---")
    for res in all_results:
        print(f"Model: {res['config']['name']}, Final Test Acc: {res['final_test_accuracy']:.4f}, Time: {res['training_time']:.2f}s")

if __name__ == '__main__':
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    regularization_experiments(train_loader, test_loader, epochs=15)