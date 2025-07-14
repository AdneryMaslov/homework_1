import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tokenizers import Tokenizer
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from text_dataset import TextDataset
from generator_transformer import GeneratorTransformer

# --- ФУНКЦИЯ ДЛЯ ПОСТРОЕНИЯ ГРАФИКА (ТЕПЕРЬ С ДВУМЯ ЛИНИЯМИ) ---
def plot_metrics(train_losses, val_losses, save_dir):
    """Строит и сохраняет график потерь (training и validation) по эпохам."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.title('Потери (Loss) во время обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_dir, 'full_training_chart.png')
    plt.savefig(save_path)
    print(f"\nГрафик обучения сохранен в: {save_path}")
    plt.close()

# --- ФУНКЦИЯ ДЛЯ ОДНОЙ ЭПОХИ ОБУЧЕНИЯ ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training')
    
    for input_ids, target_ids in progress_bar:
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return total_loss / len(dataloader)

# --- НОВАЯ ФУНКЦИЯ ДЛЯ ВАЛИДАЦИИ ---
def validate_epoch(model, dataloader, criterion, device):
    model.eval() # Переводим модель в режим оценки (отключаем dropout и т.д.)
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Validation')
    
    with torch.no_grad(): # Отключаем расчет градиентов для экономии ресурсов
        for input_ids, target_ids in progress_bar:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss += loss.item()
            progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
            
    return total_loss / len(dataloader)


def main():
    # --- Конфигурация ---
    config = {
        'batch_size': 8,
        'max_length': 128,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'd_model': 256,
        'nhead': 8,
        'num_decoder_layers': 4,
        'd_ff': 1024,
        'dropout': 0.1,
        'save_dir': 'checkpoints_with_validation', # Новая папка
    }

    # ... (код для определения устройства и токенизатора) ...
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Используется Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
    config['device'] = device
    print(f"Используемое устройство: {device}")
    
    tokenizer_path = 'transformer_basics/mistral_tokenizer.json'
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Файл токенизатора не найден: {tokenizer_path}.")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # --- ИЗМЕНЕНИЕ: ЗАГРУЗКА ДАННЫХ И РАЗДЕЛЕНИЕ НА TRAIN/VAL ---
    full_dataset = TextDataset('data/corpus.txt', tokenizer, max_length=config['max_length'])
    
    # Определяем размеры для обучающей и валидационной выборок (90% / 10%)
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    
    # Разделяем датасет
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Размер полного датасета: {len(full_dataset)} чанков")
    print(f"Обучающая выборка: {len(train_dataset)} чанков")
    print(f"Валидационная выборка: {len(val_dataset)} чанков")
    
    # Создаем два загрузчика данных
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # --- Инициализация модели, оптимизатора и функции потерь ---
    model = GeneratorTransformer(config, tokenizer).to(device)
    print(f'Количество параметров в модели: {sum(p.numel() for p in model.parameters()):,}')
    criterion = nn.CrossEntropyLoss(ignore_index=full_dataset.get_pad_token_id())
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    os.makedirs(config['save_dir'], exist_ok=True)

    # Создаем списки для хранения истории
    history_train_losses = []
    history_val_losses = []
    
    # --- ОБНОВЛЕННЫЙ ЦИКЛ ОБУЧЕНИЯ ---
    for epoch in range(config['num_epochs']):
        print(f"\n--- Эпоха {epoch + 1}/{config['num_epochs']} ---")
        
        # Шаг обучения
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history_train_losses.append(train_loss)
        
        # Шаг валидации
        val_loss = validate_epoch(model, val_loader, criterion, device)
        history_val_losses.append(val_loss)
        
        print(f'Эпоха {epoch + 1} завершена. Avg Train Loss: {train_loss:.4f}, Avg Validation Loss: {val_loss:.4f}')

        model.save_checkpoint(os.path.join(config['save_dir'], f'generator_epoch_{epoch+1}.pt'))

    print("\nОбучение завершено!")
    model.save_checkpoint(os.path.join(config['save_dir'], 'generator_final.pt'))

    # Строим и сохраняем график
    plot_metrics(history_train_losses, history_val_losses, config['save_dir'])


if __name__ == '__main__':
    main()