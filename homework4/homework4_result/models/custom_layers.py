import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMechanism(nn.Module):
    """
    Пример простого и эффективного механизма внимания (Squeeze-and-Excitation Block),
    который учится взвешивать важность каждого канала.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Squeeze: Глобальное среднее пулинг по пространственным измерениям
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation: Два полносвязных слоя для изучения весов каналов
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid())
        print("-> Инициализирован кастомный слой: AttentionMechanism")

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    