import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

def run_epoch(model, data_loader, criterion, optimizer, device, is_train=True):
    """Запускает одну эпоху обучения или тестирования."""
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    data_iterator = tqdm(data_loader, desc=f"{'Training' if is_train else 'Testing'}")
    
    with torch.set_grad_enabled(is_train):
        for inputs, targets in data_iterator:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == targets.data)
            total_samples += inputs.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions.item() / total_samples
    return epoch_loss, epoch_acc

def train_model(model, train_loader, test_loader, epochs, lr=0.001, device='mps'):
    """Основной цикл обучения модели."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_losses': [], 'train_accs': [], 'test_losses': [], 'test_accs': []}
    
    start_time = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_train=False)
        
        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['test_losses'].append(test_loss)
        history['test_accs'].append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    total_time = time.time() - start_time
    history['training_time'] = total_time
    print(f"Total training time: {total_time:.2f} seconds")
    return history