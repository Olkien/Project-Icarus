import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn

torch.manual_seed(0)

transform = transforms.ToTensor()

train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

images, labels = next(iter(train_loader))

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    n = 0
    for x, y in loader:
        optimizer.zero_grad()
        predictions = model(x)
        loss = loss_fn(predictions,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
        n += len(x)
    return total_loss / n

@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x,y in loader:
        predictions = model(x)
        loss = loss_fn(predictions, y)
        total_loss += loss.item()*len(x)
        correct += (predictions.argmax(1) == y).sum().item()
        n+=len(x)
    return total_loss/n, correct/n

for epoch in range(12):
    tr_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
    te_loss, te_acc = evaluate(model, test_loader, loss_fn)
    print(f"epoch {epoch}: train {tr_loss:.4f} | test {te_loss:.4f} | acc {te_acc:.3f}")
