import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torch import nn

torch.manual_seed(0)

transform = transforms.ToTensor()

train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

small_train = Subset(train_data, range(500))
small_loader = DataLoader(small_train, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
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


def plot_history(history): #Plotting done by Claude
    best = history["test"].index(min(history["test"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(history["train"], label="train loss")
    ax1.plot(history["test"], label="test loss")
    ax1.axvline(best, color="gray", linestyle=":", label=f"lowest test loss (epoch {best})")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(history["acc"], color="green")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("test accuracy")
    ax2.set_title("Test accuracy")

    plt.tight_layout()
    plt.savefig("overfitting.png", dpi=150)
    plt.show()

history = {"train": [], "test": [], "acc": []}

for epoch in range(60):
    tr_loss = train_one_epoch(model, small_loader, loss_fn, optimizer)
    te_loss, te_acc = evaluate(model, test_loader, loss_fn)
    history["train"].append(tr_loss)
    history["test"].append(te_loss)
    history["acc"].append(te_acc)
    print(f"epoch {epoch}: train {tr_loss:.4f} | test {te_loss:.4f} | acc {te_acc:.3f}")

plot_history(history)