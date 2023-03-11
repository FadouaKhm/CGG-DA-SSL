import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        # out_width = (28+2-5)/2+1 = 27/2+1 = 13
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=4, padding=0)
        # out_width = (14-5)/2+1 = 5
        #self.drop1=nn.Dropout2d(p=0.3) 
        # 6 * 6 * 16 = 576
        self.linear1 = nn.Linear(4*4*32, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        # print(x.shape)
        # x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
class Decoder(nn.Module):
    
    def __init__(self, latent_dims):
        super().__init__()

        ### Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(128, 4 * 4 * 32),
            nn.ReLU(True)
        )

        ### Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 4, 4))

        ### Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(32, 16, 4, stride=4, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Second transposed convolution
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Third transposed convolution
            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        # print()
        # print(x.shape)
        # Apply linear layers
        x = self.decoder_lin(x)
        # print(x.shape)
        # Unflatten
        x = self.unflatten(x)
        # print(x.shape)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # print(x.shape)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        # x = torch.sigmoid(x)
        # print(x.shape)
        return x
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        # x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)

def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for _,x, _ in dataloader: 
        # Move tensor to the proper device
        x = x.to(device)
        x_hat = vae(x)
        # Evaluate loss
        loss = ((x - x_hat)**2).sum() + vae.encoder.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss+=loss.item()

    return train_loss / len(dataloader.dataset)

### Testing function
def test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    all_x =  []
    all_y =[]
    all_loss = []
    with torch.no_grad(): # No need to track the gradients
        for _,x, y in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            y = y.to(device)
            # Encode data
            encoded_data = vae.encoder(x)
            # Decode data
            x_hat = vae(x)
            all_x.append(x_hat.cpu().numpy())
            all_y.append(y.cpu().numpy())
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset), all_x, all_y

def plot_ae_outputs(val_data,device,encoder,decoder,n=5):
    plt.figure(figsize=(10,4.5))
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = val_data[i][1].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      img = img.cpu().squeeze().numpy()[0,:,:]
      # img = np.transpose(img, (1,2,0))
      plt.imshow(img, cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      rec_img = rec_img.cpu().squeeze().numpy()[0,:,:]
      # rec_img = np.transpose(rec_img, (1,2,0))
      plt.imshow(rec_img, cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()

import h5py
def generate_synth(vae,latent_dim, device, num_samples, exp_id):    
    X_fake = []
    y_fake = []
    vae.eval()
    for _ in range(num_samples):
        with torch.no_grad():
            z = torch.randn(1, latent_dim, device=device)
            x_hat = vae.decoder(z)
            X_fake.append(x_hat.cpu().numpy())
            y_fake.append(-1)
            x_hat = x_hat.cpu().squeeze().numpy()
            x_hat = np.transpose(x_hat, (1,2,0))
            # plt.imshow(x_hat)
            # plt.show()
    X_ = np.concatenate(X_fake,axis=0)
    Y_ = np.array(y_fake)
    
    Xn = 255*(X_ - X_.min(axis=(1,2,3))[:,None,None,None])/(X_.max(axis=(1,2,3))[:,None,None,None] - X_.min(axis=(1,2,3))[:,None,None,None])
    data_set = {}
    data_set["images"] = Xn
    data_set["labels"] = Y_    
    
    h = h5py.File('./data/VAE/{}.hdf5'.format('synth_ul_'+exp_id), 'w')
    for k, v in data_set.items():
        h.create_dataset(k, data=np.array(v))
    h.close()

def generate_reconst(eval_loader_,vae, device, exp_id):
    _,X,Y = test_epoch(vae, device, eval_loader_)
    X_ = np.concatenate(X,axis=0)
    Y_ = np.concatenate(Y,axis=0)
    print(X_.min(), X_.max())
    
    Xn = 255*(X_ - X_.min(axis=(1,2,3))[:,None,None,None])/(X_.max(axis=(1,2,3))[:,None,None,None] - X_.min(axis=(1,2,3))[:,None,None,None])
    data_set = {}
    data_set["images"] = Xn
    data_set["labels"] = Y_
    print(Xn.shape, Y_.shape)
    
    h = h5py.File('./data/VAE/{}.hdf5'.format('reconst_ul_'+exp_id), 'w')
    for k, v in data_set.items():
        h.create_dataset(k, data=np.array(v))
    h.close()
