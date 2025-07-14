import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer: Tokenizer, max_length=128):
        self.max_length = max_length
        self.tokenizer = tokenizer

        # Загружаем и токенизируем весь текст
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Добавляем специальные токены
        self.tokenizer.add_special_tokens(['<pad>', '<s>', '</s>'])
        self.pad_token_id = self.tokenizer.token_to_id('<pad>')
        
        print("Токенизация всего текста...")
        # Убираем .ids, чтобы получить объект Encoding
        tokenized_text = self.tokenizer.encode(text) 
        all_token_ids = tokenized_text.ids

        print(f"Всего токенов в тексте: {len(all_token_ids)}")

        # Создаем чанки (куски) фиксированной длины
        self.chunks = []
        for i in range(0, len(all_token_ids) - max_length, max_length):
            chunk = all_token_ids[i:i + max_length]
            self.chunks.append(torch.tensor(chunk, dtype=torch.long))

        print(f"Создано {len(self.chunks)} чанков размером {max_length}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        # Модель учится предсказывать следующее слово,
        # поэтому вход - это чанк без последнего токена,
        # а цель - это чанк без первого токена.
        chunk = self.chunks[idx]
        input_ids = chunk[:-1]
        target_ids = chunk[1:]
        return input_ids, target_ids
        
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_pad_token_id(self):
        return self.pad_token_id