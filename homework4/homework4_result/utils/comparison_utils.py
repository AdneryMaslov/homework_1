import pandas as pd

def count_parameters(model):
    """Подсчитывает обучаемые параметры модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_comparison_table(results):
    """Создает и выводит таблицу для сравнения моделей."""
    data = []
    for model_name, res in results.items():
        data.append({
            "Model": model_name,
            "Test Accuracy": f"{res['final_accuracy']:.4f}",
            "Parameters": f"{res['params']:,}",
            "Training Time (s)": f"{res['time']:.2f}"
        })
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    return df

# Добавьте другие функции для сравнения, если необходимо