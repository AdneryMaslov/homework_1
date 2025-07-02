import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import os

from utils_hw import get_device, log_epoch
from homework_model_modification import LinearRegression, SoftmaxRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CSVDataset(Dataset):
    """Кастомный Dataset для загрузки и предобработки данных из CSV файла."""
    def __init__(self, csv_file, target_col, task='regression'):
        self.task = task
        df = pd.read_csv(csv_file)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        X = df.drop(columns=[target_col])
        y = df[target_col]

        numeric_cols = X.select_dtypes(include=['number']).columns
        categorical_cols = X.select_dtypes(exclude=['number']).columns
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        self.features = torch.tensor(X.values, dtype=torch.float32)

        if self.task == 'regression':
            self.target = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        else:
            le = LabelEncoder()
            self.target = torch.tensor(le.fit_transform(y.values), dtype=torch.long)
            self.num_classes = len(le.classes_)
            self.class_names = le.classes_
        
        logging.info(f"Датасет '{os.path.basename(csv_file)}' загружен: {len(self)} сэмплов, {self.features.shape[1]} признаков.")
        if self.task == 'classification':
            logging.info(f"Количество классов: {self.num_classes}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

def run_regression_experiment(dataset_path, target_col):
    """Обучает модель линейной регрессии на кастомном CSV датасете."""
    logging.info(f"--- Запуск эксперимента по регрессии на {os.path.basename(dataset_path)} ---")
    dataset = CSVDataset(csv_file=dataset_path, target_col=target_col, task='regression')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    device = get_device()
    
    in_features = dataset.features.shape[1]
    model = LinearRegression(in_features).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (batch_X, batch_y) in enumerate(dataloader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(dataloader))
        
        log_epoch(epoch, epochs, avg_loss)
    
    save_path = os.path.join(MODELS_DIR, f'{os.path.basename(dataset_path)}_linreg.pt')
    torch.save(model.state_dict(), save_path)
    logging.info(f"Модель регрессии сохранена в {save_path}.")

def run_classification_experiment(dataset_path, target_col):
    """Обучает модель softmax регрессии на кастомном CSV датасете."""
    logging.info(f"--- Запуск эксперимента по классификации на {os.path.basename(dataset_path)} ---")
    dataset = CSVDataset(csv_file=dataset_path, target_col=target_col, task='classification')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    device = get_device()

    in_features = dataset.features.shape[1]
    num_classes = dataset.num_classes
    model = SoftmaxRegression(in_features, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (batch_X, batch_y) in enumerate(dataloader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(dataloader))
        
        log_epoch(epoch, epochs, avg_loss)

    save_path = os.path.join(MODELS_DIR, f'{os.path.basename(dataset_path)}_logreg.pt')
    torch.save(model.state_dict(), save_path)
    logging.info(f"Модель классификации сохранена в {save_path}.")

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # --- Эксперимент с датасетом для РЕГРЕССИИ ---
    try:
        insurance_path = os.path.join(DATA_DIR, 'insurance.csv')
        run_regression_experiment(
            dataset_path=insurance_path, 
            target_col='charges'
        )
    except FileNotFoundError:
        logging.error(f"Файл '{insurance_path}' не найден. Пожалуйста, скачайте его и поместите в папку data.")
    except KeyError:
        logging.error("Целевой столбец 'charges' не найден в insurance.csv. Проверьте файл.")

    print("\n" + "="*50 + "\n")

    # --- Эксперимент с датасетом для КЛАССИФИКАЦИИ ---
    try:
        cancer_path = os.path.join(DATA_DIR, 'breast-cancer.csv')
        run_classification_experiment(
            dataset_path=cancer_path, 
            target_col='diagnosis'
        )
    except FileNotFoundError:
        logging.error(f"Файл '{cancer_path}' не найден. Пожалуйста, скачайте его и поместите в папку data.")
    except KeyError:
        logging.error("Целевой столбец 'diagnosis' не найден в breast-cancer.csv. Проверьте файл.")