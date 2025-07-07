import torch.nn as nn

class SimpleFC(nn.Module):
    """Простая полносвязная сеть для MNIST."""
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class DeepFC_CIFAR(nn.Module):
    """Глубокая полносвязная сеть для CIFAR-10."""
    def __init__(self, input_size=32*32*3, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)