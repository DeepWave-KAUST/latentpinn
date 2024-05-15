import os
import json
import math
import numpy as np 

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib
import seaborn as sns
plt.style.use("../science.mplstyle")

# Progress bar
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

# Torchvision
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser

from latentpinn.data import *
from latentpinn.model import *
from latentpinn.utils import *
from latentpinn.plots import *

def main():
    
    wandb.init(project='LatentPINNs-00-AEs')
    wandb.run.log_code(".")
    wandb_dir = wandb.run.dir
    wandb.config.update(args)
    
    CHECKPOINT_PATH=wandb_dir

    # Setting the seed
    pl.seed_everything(12315019)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    NUM_SAMPLES=36000

    # Create the custom dataset with the transformation
    reseam = torch.from_numpy(np.load('../data/smooth5mix_reseam.npy')[:,::2,::2]).to(device)
    marm = torch.from_numpy(np.load('../data/smooth5mix_marm.npy')[:,::2,::2]).to(device)
    
    train_set = NumpyDataset(reseam[:28000], normalize=True, water_depth=20)
    val_set = NumpyDataset(reseam[28000:29000], normalize=True, water_depth=20)
    test_set = NumpyDataset(reseam[29000:], normalize=True, water_depth=20)
    marm_set = NumpyDataset(marm, normalize=True, water_depth=20)

    # Create the DataLoader
    train_loader = data.DataLoader(train_set, batch_size=16, shuffle=False)#, drop_last=True, pin_memory=True, num_workers=20)
    val_loader = data.DataLoader(val_set, batch_size=160, shuffle=False)#, drop_last=False, num_workers=20)
    test_loader = data.DataLoader(test_set, batch_size=160, shuffle=False)#, drop_last=False, num_workers=20)
    marm_loader = data.DataLoader(marm_set, batch_size=160, shuffle=False)#, drop_last=False, num_workers=20)
    
    print(NUM_SAMPLES)

    print("Creating dataloader.")
    
    LATENT_DIM = args.latent_dim
    
    # print(next(iter(train_loader)).shape)

    train_imgs = get_images(4**2, train_set, delta=500)
    test_imgs = get_images(4**2, test_set, delta=10)
    marm_imgs = get_images(4**2, marm_set, delta=1)    
    print(marm_imgs.shape)
    
    plot_square_image((2+train_imgs.detach().cpu().numpy())*2, 4, 4, 'train_reseam_ae.pdf', '../saves/', wavefield=True)
    plot_square_image((2+test_imgs.detach().cpu().numpy())*2, 4, 4, 'test_reseam_ae.pdf', '../saves/', wavefield=True)
    
    print("Creating model.")
    
    wandb_logger = WandbLogger(log_model="all", project='LatentPINNs-00-AEs')

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=wandb_dir, 
        accelerator="gpu",
        devices=1,
        max_epochs=600, 
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(
                monitor='val_loss',  # Metric to monitor for saving the model
                filename='model-{epoch:02d}',  # Filename pattern for saved checkpoints
                save_top_k=-1,  # Save all checkpoints
                every_n_epochs=10  # Save every 2 epochs
            ),
            GenerateCallback(get_images(16, next(iter(test_loader)), delta=5), every_n_epochs=10, name='test'),
            GenerateCallback(get_images(16, next(iter(marm_loader)), delta=5), every_n_epochs=10, name='marm'),
            LearningRateMonitor("epoch")
        ]
    )
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    SAVED_PATH = ""

    if os.path.isfile(SAVED_PATH):
        print("Found pretrained model, loading...")
        model = AutoEncoder.load_from_checkpoint(SAVED_PATH)
    else:
        model = AutoEncoder(base_channel_size=128, latent_dim=LATENT_DIM)
        trainer.fit(model, train_loader, val_loader)
        
    train_latent = embed_imgs(model, train_loader)[1]
    marm_latent = embed_imgs(model, marm_loader)[1]
    test_latent = embed_imgs(model, test_loader)[1]
    
    np.save('../saves/train_latent_'+str(LATENT_DIM)+'.npy', train_latent)
    np.save('../saves/marm_latent_'+str(LATENT_DIM)+'.npy', marm_latent)
    np.save('../saves/test_latent_'+str(LATENT_DIM)+'.npy', test_latent)

if __name__ == '__main__':
    
    parser = ArgumentParser(description="LatentPINN training.")
    parser.add_argument(
            "--latent_dim",
            type=int,
            default=10,
            help="Latent dimension."
    )
    args = parser.parse_args()
    
    main()