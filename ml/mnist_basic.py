import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(0)

transform = transforms.ToTensor()

train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

images, labels = next(iter(train_loader))
print(images.shape, labels.shape)