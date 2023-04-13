from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
import torch

class MNIST_Data(data.Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x,y

def load_mnist(mode):
    transform = []
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=[0.5], std=[0.5]))
    transform = T.Compose(transform)
    if mode == "train":
        dataset = datasets.MNIST(root="../data", train=True, transform=transform, download=True)
    elif mode == "test":
        dataset = datasets.MNIST(root="../data", train=False, transform=transform, download=True)
    
    x = dataset.data.float()
    y = dataset.targets

    return x,y

def get_loader(batch_size, mode, num_workers):
    x,y = load_mnist(mode)
    datasets = MNIST_Data(x,y)
    data_loader = DataLoader(dataset = datasets,
                            batch_size = batch_size,
                            shuffle = (mode=="train"),
                            num_workers = num_workers)
    return data_loader
        

        
