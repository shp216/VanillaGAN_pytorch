import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Discriminator(nn.Module):  
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(image_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        y = x.view(x.size(0), -1)
        y = self.layer(y)
        return y


class Generator(nn.Module):   
    def __init__(self, latent_size, image_size):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, image_size),
            nn.Tanh()
        )
        
    def forward(self, x):
        y = self.layer(x)
        y = y.view(x.size(0), 1, 28, 28)
        return y