from dataloader import get_loader
from solver import Solver
import os
import argparse
from torch.backends import cudnn

def main(config):
    # For fast training.
    cudnn.benchmark = True

    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir): 
        os.makedirs(config.result_dir)

    # Solver for training and testing VanillaGAN.
    solver = Solver(config)
    
    if config.mode == "train":
        solver.train()
    
    elif config.mode == "test":
        solver.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()        

    # Model Configuration
    parser.add_argument('--image_size', type=int, default=28*28, help='image_size')
    parser.add_argument('--latent_size', type=int, default=100, help='size of latent_vector used in Generator')

    # Miscellaneous Configuration
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'test'])
    parser.add_argument('--num_workers', type=int, default=1)

    # Directories.
    parser.add_argument('--model_save_dir', type=str, default='models')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--result_dir', type=str, default='results')

    #Training Configuration
    parser.add_argument('--batch_size', type=int, default=60, help='mini-batch_size')
    parser.add_argument('--num_epochs', type=int, default=100, help='total epochs')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate for D')

    config = parser.parse_args()
    print(config)
    main(config)
    
    
