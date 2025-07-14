import requests
import os

# Список книг для скачивания
BOOKS = [
    {
        "name": "Alice in Wonderland (Russian)",
        "url": "http://lib.ru/CARROLL/alisa.txt",
        "encoding": "koi8-r"
    },
    {
        "name": "Crime and Punishment",
        "url": "http://lib.ru/LITRA/DOSTOEWSKIJ/prestup.txt",
        "encoding": "koi8-r"
    },
    {
        "name": "Fathers and Sons",
        # !!! ВОТ ИСПРАВЛЕННАЯ ССЫЛКА !!!
        "url": "http://lib.ru/LITRA/TURGENEW/otcy_i_deti.txt",
        "encoding": "koi8-r"
    }
]

def download_and_combine_books(save_path="data/russian_classics_corpus.txt"):
    """Скачивает несколько книг и объединяет их в один текстовый файл."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        full_text_corpus = ""
        
        for book in BOOKS:
            print(f"Скачивание: {book['name']}...")
            response = requests.get(book['url'])
            response.raise_for_status()
            
            response.encoding = book['encoding']
            
            full_text_corpus += response.text + "\n\n"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(full_text_corpus)
            
        print("-" * 30)
        print(f"Все книги успешно скачаны и объединены в файл: {save_path}")
        print(f"Итоговый размер файла: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при скачивании файла: {e}")

if __name__ == "__main__":
    download_and_combine_books()