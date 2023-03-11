from pl_bolts.models.autoencoders import VAE
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd # this module is useful to work with tabular data
import random # this module will be used to select random samples from a collection
import os # this module will be used just to create directories in the local filesystem
from tqdm import tqdm # this module is useful to plot progress bars
import plotly.io as pio
import torchvision.transforms.functional as F
import torchvision
import torch
import h5py

def generate_synth_lg(vae,num_samples,exp_id,img_size):
    X_fake = []
    y_fake = []
    vae.eval()
    for _ in range(num_samples):
        with torch.no_grad():
            z = torch.randn(1, vae.latent_dim)
            x_hat = vae.decoder(z)
            if x_hat.shape[0] != img_size:
                x_hat_r = F.resize(x_hat, img_size)
            else:
                x_hat_r = x_hat
            X_fake.append(x_hat_r.numpy())
            y_fake.append(-1)
            # x_hat = x_hat.cpu().squeeze().numpy()
            # x_hat = np.transpose(x_hat, (1,2,0))
            # plt.imshow(x_hat)
            # plt.show()
    X_ = np.concatenate(X_fake,axis=0)
    Y_ = np.array(y_fake)
    
    Xn = 255*(X_ - X_.min(axis=(1,2,3))[:,None,None,None])/(X_.max(axis=(1,2,3))[:,None,None,None] - X_.min(axis=(1,2,3))[:,None,None,None])
     
        
    
    data_set = {}
    data_set["images"] = Xn
    data_set["labels"] = Y_    
    
    h = h5py.File('./data/{}.hdf5'.format('synth_lg_ul_'+exp_id), 'w')
    for k, v in data_set.items():
        h.create_dataset(k, data=np.array(v))
        
def generate_reconst_lg(vae,eval_loader_, exp_id):
    all_x = []
    all_y = []
    val_loss = 0.0
    vae.eval()
    for batch_idx, (_,data, target) in enumerate(eval_loader_):
        with torch.no_grad():
        
            # Run VAE
            recon_batch = vae(data)
            # Compute loss
            all_y.append(target)
            loss = vae.step((_,data, target),0)[1]['loss']
            val_loss += loss
            # Plot reconstructions
            # n = min(data.size(0), 8)
            # print(recon_batch.shape)
            # comparison = torch.cat([data[:n], recon_batch.view(recon_batch.shape[0], -1, 64, 64)[:n]])
            # comparison = torchvision.utils.make_grid(comparison)
            # print("Reconstructions: ")
            # plt.imshow(comparison.numpy().transpose(1,2,0))
            # plt.show()
            # recon_batch = F.resize(recon_batch, 64)
            all_x.append(recon_batch.numpy())
            
    print('Total reconstruction loss = ', val_loss / len(eval_loader_.dataset))
    n = min(data.size(0), 4)
    comparison = torch.cat([data[:n], recon_batch.view(recon_batch.shape[0], -1, 64, 64)[:n]])
    comparison = torchvision.utils.make_grid(comparison)
    print("Reconstruction samples: ")
    plt.imshow(comparison.numpy().transpose(1,2,0))
    plt.show()
            
    X_ = np.concatenate(all_x,axis=0)
    Y_ = np.concatenate(all_y,axis=0)
    
    Xn = 255*(X_ - X_.min(axis=(1,2,3))[:,None,None,None])/(X_.max(axis=(1,2,3))[:,None,None,None] - X_.min(axis=(1,2,3))[:,None,None,None])
    data_set = {}
    data_set["images"] = Xn
    data_set["labels"] = Y_
    
    

    h = h5py.File('./data/{}.hdf5'.format('reconst_lg_ul_'+exp_id), 'w')
    for k, v in data_set.items():
        h.create_dataset(k, data=np.array(v))
    h.close()