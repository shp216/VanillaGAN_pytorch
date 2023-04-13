from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
import torch

transform = []
transform.append(T.ToTensor())
transform.append(T.Normalize(mean=[0.5], std=[0.5]))
transform = T.Compose(transform)
dataset = datasets.MNIST(root="../data", train=True, transform=transform, download=True)

for i, data in enumerate(dataset):
    print(type(data[0]))
    print(type(data[1]))
    print(type(data))
    break

print(dataset.targets.shape)

