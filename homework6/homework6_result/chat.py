import torch
from generator_transformer import GeneratorTransformer

def chat():
    # --- Определение устройства ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Используемое устройство: {device}")

    # --- Загрузка модели ---
    checkpoint_path = "checkpoints/generator_final.pt" # или любой другой чекпоинт
    try:
        model = GeneratorTransformer.load_from_checkpoint(checkpoint_path, device=device)
    except FileNotFoundError:
        print(f"Ошибка: Чекпоинт не найден по пути '{checkpoint_path}'.")
        print("Пожалуйста, сначала обучите модель, запустив 'train.py'.")
        return
        
    print("Модель успешно загружена. Введите 'quit' для выхода.")
    print("-" * 30)

    while True:
        user_input = input("Вы: ")
        if user_input.lower() == 'quit':
            break
        
        # Генерация ответа
        response = model.generate(
            user_input, 
            temperature=0.7, 
            max_new_tokens=100
        )
        
        # Пост-обработка ответа, чтобы он выглядел чище
        cleaned_response = response.replace(user_input, "").strip()
        
        print(f"Бот: {cleaned_response}")

if __name__ == "__main__":
    chat()