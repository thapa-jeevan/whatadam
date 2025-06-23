import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from .adam import Adam


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14×14
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 7×7
            nn.Flatten(),

            nn.Linear(128 * 7 * 7, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)  # logits
        )

    def forward(self, x):
        return self.net(x)


epochs = 10
batch = 64
lr = 1e-3

device = torch.device("cuda")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch=batch, shuffle=True, num_workers=4, pin_memory=True)

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)

best_acc = 0.0
for epoch in range(1, epochs + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(train_loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    train_loss, train_acc = running_loss / total, 100.0 * correct / total
    print(f"Epoch: {epoch}/{epochs}] \t train_loss={train_loss:.4f}  train_acc={train_acc:.2f}% | ")
