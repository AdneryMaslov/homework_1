import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import json
import os

from .model_utils import FullyConnectedModel, count_parameters
from .visualization_utils import plot_training_history


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_epoch(model, data_loader, criterion, optimizer, is_training):
    model.train(is_training)
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    data_iterator = tqdm(data_loader, desc=f"{'Training' if is_training else 'Testing'}")

    with torch.set_grad_enabled(is_training):
        for inputs, targets in data_iterator:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == targets.data)
            total_samples += inputs.size(0)

            data_iterator.set_postfix(loss=total_loss / total_samples, acc=correct_predictions.double().item() / total_samples)

    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()


def run_experiment(config, train_loader, test_loader, epochs=10, learning_rate=0.001, weight_decay=0.0):
    print(f"--- Starting Experiment: {config['name']} ---")

    model = FullyConnectedModel(
        input_size=config['input_size'],
        num_classes=config['num_classes'],
        layer_config=config['layers']
    ).to(device)

    print(f"Number of parameters: {count_parameters(model)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = {
        'train_losses': [], 'train_accs': [],
        'test_losses': [], 'test_accs': []
    }

    start_time = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, is_training=True)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, is_training=False)

        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['test_losses'].append(test_loss)
        history['test_accs'].append(test_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    results_dir = f"homework3_result/results/{config['experiment_type']}"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{config['name']}_results.json")

    results_data = {
        'config': config,
        'history': history,
        'training_time': training_time,
        'num_parameters': count_parameters(model),
        'final_test_accuracy': history['test_accs'][-1]
    }
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=4)

    plot_path = f"homework3_result/plots/{config['experiment_type']}/{config['name']}_learning_curves.png"
    plot_training_history(history, f"Training History: {config['name']}", save_path=plot_path)

    print(f"--- Finished Experiment: {config['name']} ---\n")
    return results_data