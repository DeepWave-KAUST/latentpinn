import torch
import numpy as np
import scipy.io
import time
import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import init
from torch.autograd import Variable, grad
import pytorch_lightning as pl
import torchvision
import wandb

from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.autograd import Variable, grad
from torch.optim import Adam, LBFGS, RMSprop
from torch.nn import Parameter

from latentpinn.utils import *
from latentpinn.model import *

from tqdm import tqdm

from latentpinn.plots import *

class Basicblock(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(Basicblock, self).__init__()
        self.layer1 = nn.Linear(in_planes,out_planes)

    def forward(self, x):
        out = torch.tanh(self.layer1(x))
        return out
    
def regloss(x, y, sx, v0, f, du_pred_real, du_pred_imag, radii=1e3):
    
    factor_d = F.relu(0.01*(v0 * 3.14/f)**2-(sx-x)**2-(y-0.025)**2)*radii*f
    loss_reg = torch.sum(factor_d*torch.pow(du_pred_real,2)) + torch.sum(factor_d*torch.pow(du_pred_imag,2))
                                                                                                              
    return loss_reg / (x.shape[0] * 2)

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear')!=-1:
        init.xavier_normal_(m.weight,gain=1)
        init.constant_(m.bias, 0)
        #init.orthogonal_(m.weight)
    if classname.find('Conv')!=-1:
        init.xavier_normal_(m.weight,gain=1)
        init.constant_(m.bias,0)

def gradient(y,x,grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad_ = grad(y,[x],grad_outputs=grad_outputs,create_graph=True)[0]
    return grad_

def divergence(y,x):
    div = 0
    for i in range(y.shape[-1]):
        div += grad(y[...,i],x,torch.ones_like(y[...,i]),create_graph=True)[0][...,i:i+1]
    return div

def laplace(y,x):
    grad_ = gradient(y,x)
    return divergence(grad_,x)

class PhysicsInformedNN(nn.Module):
    def __init__(self, layers):
        super(PhysicsInformedNN, self).__init__()
        self.layers = layers
        self.in_planes = self.layers[0]
        self.layer1 = self._make_layer(Basicblock,self.layers[1:len(layers)-1])
        self.linear = nn.Linear(layers[-2],layers[-1])

    def _make_layer(self, block, layers):
        layers_net = []
        for layer in layers:
            layers_net.append(block(self.in_planes,layer))
            self.in_planes = layer
        return nn.Sequential(*layers_net)

    def forward(self,x):
        out = self.layer1(x)
        out = self.linear(out)
        return out[:,0:1], out[:,1:2]
    
class NN(nn.Module):    
    def __init__(self, layers):
        super().__init__()
        
        self.layers = layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
    def forward(self, x):       
        a = x.float()
        for i in range(0, len(self.layers)-2):
            z = self.linears[i](a)
            a = torch.tanh(z)
        out = self.linears[-1](a)
        return out[:,0:1], out[:,1:2]

def pinn_train(model,x_train_all,y_train_all,sx_train_all,
               u0_real_train_all,u0_imag_train_all,
               lb_all,ub_all,m_all,m0_all,omega,
               niter,embedding,f,
               x_bc, z_bc, sx_bc, dur_bc, dui_bc, bc_type,
               num_vel=1,latent=None,alpha=None,radii=1,single_id=0, 
               latent_dim=12, save_dir='./', use_wandb='n', grid_size=101):
    start_time = time.time()
    m_all = torch.FloatTensor(m_all).cuda()
    m0_all = torch.FloatTensor(m0_all).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(),lr=1e-3, 
        weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.8, 
        patience=50, 
        min_lr=5e-5
    )
    # optimizer = Adam(model.parameters(),lr=1e-3)
    misfit = []
    model.train()

    print('Using '+bc_type+' type.')
    print('Using '+str(alpha)+' alpha.')

    for it in range(niter):
        
        v_id = torch.randint(num_vel, (1,), device='cuda')[0]
        
        m, m0 = m_all[v_id].cuda(), m0_all[v_id].cuda()
        
        v0 = torch.unique(torch.sqrt(1/m0))
        
        lb, ub = lb_all[v_id], ub_all[v_id]
        
        optimizer.zero_grad()
        
        x = x_train_all[v_id].clone().detach().requires_grad_(True)
        y = y_train_all[v_id].clone().detach().requires_grad_(True)
        sx = sx_train_all[v_id].clone().detach().requires_grad_(True)
        
        x_input = torch.cat((x,y,sx),1)
        x_input = 2.0 *(x_input-lb)/(ub - lb) - 1.0
        x_input = embedding(x_input)
        
        # if latent is not None:
        #     print(x_input.shape, latent[v_id,:].shape)
            
        # x_input = torch.cat((x_input, latent[v_id,:].repeat(40000).view(-1,latent_dim)),1)
        x_input = torch.cat((x_input, latent[v_id,:].repeat(40000).reshape(-1,latent_dim).requires_grad_(False)),1)
        
        dU_real_pred, dU_imag_pred = model(x_input)

        du_real_xx = laplace(dU_real_pred,x)
        du_imag_xx = laplace(dU_imag_pred,x)
        du_real_yy = laplace(dU_real_pred,y)
        du_imag_yy = laplace(dU_imag_pred,y)

        f_real_pred = omega*omega*m*dU_real_pred + du_real_xx + du_real_yy + omega*omega*(m-m0)*u0_real_train_all[v_id]
        f_imag_pred = omega*omega*m*dU_imag_pred + du_imag_xx + du_imag_yy + omega*omega*(m-m0)*u0_imag_train_all[v_id]

        if bc_type=='top-bottom':
            X_bc = embedding(2.0 *(torch.cat((x_bc[:,:,[0,-1]].reshape(-1,1), z_bc[:,:,[0,-1]].reshape(-1,1), sx_bc[:,:,[0,-1]].reshape(-1,1)),1)-0)/(2.5) - 1.0)
            X_bc = torch.cat((X_bc, latent[v_id,:].repeat(1818).view(-1,latent_dim).requires_grad_(False)),1)
            du_real_bc, du_imag_bc = model(X_bc)
            boundary = ((torch.sum(torch.square(du_real_bc-dur_bc[v_id,:,:,[0,-1]].reshape(-1,1))) + torch.sum(torch.square(du_imag_bc-dui_bc[v_id,:,:,[0,-1]].reshape(-1,1))))/(x.shape [0] * 2))
        elif bc_type=='top':
            X_bc = embedding(2.0 *(torch.cat((x_bc[:,:,0].reshape(-1,1), z_bc[:,:,0].reshape(-1,1), sx_bc[:,:,0].reshape(-1,1)),1)-0)/(2.5) - 1.0)
            X_bc = torch.cat((X_bc, latent[v_id,:].repeat(grid_size*9).view(-1,latent_dim).requires_grad_(False)),1)
            du_real_bc, du_imag_bc = model(X_bc)
            boundary = ((torch.sum(torch.square(du_real_bc-dur_bc[v_id,:,:,0].reshape(-1,1))) + torch.sum(torch.square(du_imag_bc-dui_bc[v_id,:,:,0].reshape(-1,1))))/(x.shape [0] * 2))
        elif bc_type=='source-top':
            X_bc = embedding(2.0 *(torch.cat((x_bc[:,:,20].reshape(-1,1), z_bc[:,:,20].reshape(-1,1), sx_bc[:,:,20].reshape(-1,1)),1)-0)/(2.5) - 1.0)
            X_bc = torch.cat((X_bc, latent[v_id,:].repeat(grid_size*9).view(-1,latent_dim).requires_grad_(False)),1)
            du_real_bc, du_imag_bc = model(X_bc)
            boundary = (
                (torch.sum(torch.square(du_real_bc-dur_bc[v_id,:,:,20].reshape(-1,1))) + torch.sum(torch.square(du_imag_bc-dui_bc[v_id,:,:,20].reshape(-1,1))))/(x.shape [0] * 2)
                + alpha*regloss(x, y, sx, v0, f, dU_real_pred, dU_imag_pred, radii)    
            )
        elif bc_type=='source-side':
            X_bc = embedding(2.0 *(torch.cat((x_bc[:,20,:].reshape(-1,1), z_bc[:,20,:].reshape(-1,1), sx_bc[:,20,:].reshape(-1,1)),1)-0)/(2.5) - 1.0)
            X_bc = torch.cat((X_bc, latent[v_id,:].repeat(grid_size*9).view(-1,latent_dim).requires_grad_(False)),1)
            du_real_bc, du_imag_bc = model(X_bc)
            boundary = (
                (torch.sum(torch.square(du_real_bc-dur_bc[v_id,:,20,:].reshape(-1,1))) + torch.sum(torch.square(du_imag_bc-dui_bc[v_id,:,20,:].reshape(-1,1))))/(x.shape [0] * 2)
                + alpha*regloss(x, y, sx, v0, f, dU_real_pred, dU_imag_pred, radii)    
            )
        elif bc_type=='all':
            X_bc = embedding(2.0 *(torch.cat((torch.cat((x_bc[:,:,[0,-1]].reshape(-1,1), x_bc[:,[0,-1],:].reshape(-1,1)), 0), torch.cat((z_bc[:,:,[0,-1]].reshape(-1,1), z_bc[:,[0,-1],:].reshape(-1,1)), 0), torch.cat((sx_bc[:,:,[0,-1]].reshape(-1,1), sx_bc[:,[0,-1],:].reshape(-1,1)),0)),1)-0)/(2.5) - 1.0)
            X_bc = torch.cat((X_bc, latent[v_id,:].repeat(grid_size*9*4).view(-1,latent_dim).requires_grad_(False)),1)
            du_real_bc, du_imag_bc = model(X_bc)
            boundary = ((torch.sum(torch.square(du_real_bc-torch.cat((dur_bc[v_id,:,:,[0,-1]].reshape(-1,1), dur_bc[v_id,:,[0,-1],:].reshape(-1,1)), 0))) + torch.sum(torch.square(du_imag_bc-torch.cat((dui_bc[v_id,:,:,[0,-1]].reshape(-1,1), dui_bc[v_id,:,[0,-1],:].reshape(-1,1)), 0))))/(x.shape [0] * 2))
        elif bc_type=='source':
            boundary = alpha*regloss(x, y, sx, v0, f, dU_real_pred, dU_imag_pred, radii)
        elif bc_type=='source-all':
            X_bc = embedding(2.0 *(torch.cat((torch.cat((x_bc[:,:,[0,-1]].reshape(-1,1), x_bc[:,[0,-1],:].reshape(-1,1)), 0), torch.cat((z_bc[:,:,[0,-1]].reshape(-1,1), z_bc[:,[0,-1],:].reshape(-1,1)), 0), torch.cat((sx_bc[:,:,[0,-1]].reshape(-1,1), sx_bc[:,[0,-1],:].reshape(-1,1)),0)),1)-0)/(2.5) - 1.0)
            X_bc = torch.cat((X_bc, latent[v_id,:].repeat(grid_size*9*4).view(-1,latent_dim).requires_grad_(False)),1)
            du_real_bc, du_imag_bc = model(X_bc)
            boundary = (
                (torch.sum(torch.square(du_real_bc-torch.cat((dur_bc[v_id,:,:,[0,-1]].reshape(-1,1), dur_bc[v_id,:,[0,-1],:].reshape(-1,1)), 0))) 
                 + torch.sum(torch.square(du_imag_bc-torch.cat((dui_bc[v_id,:,:,[0,-1]].reshape(-1,1), dui_bc[v_id,:,[0,-1],:].reshape(-1,1)), 0))))/(x.shape [0] * 2)
                 + alpha*regloss(x, y, sx, v0, f, dU_real_pred, dU_imag_pred, radii)
                )
        else:
            boundary = 0.
                        
        loss_value = ((torch.sum(torch.square(f_real_pred)) + torch.sum(torch.square(f_imag_pred)))/(x.shape [0] * 2)
                      + alpha*boundary)
        if use_wandb=='y':
            wandb.log({'loss':loss_value})
        loss_value.backward()
        optimizer.step()
        misfit.append(loss_value.item())

        if it%(10*num_vel) == 0:
            elapsed = time.time() - start_time
            print('It: %d, Loss: %.3e, Time: %.2f' %(it,\
                loss_value,elapsed))
            start_time = time.time()
                        #save model
            state = {
                'net':model.module
            }
            torch.save(state,save_dir+'model_'+str(it)+'.tz')
            
    del misfit

def pinn_predict(model, x_star, z_star, sx_star,lb,ub,embedding,num_vel=1,latent=None,latent_dim=12,gpt=False,grid_size=101):
    model.eval()
    x_star = x_star.cuda()
    z_star = z_star.cuda()
    sx_star= sx_star.cuda()
    x_input = torch.cat((x_star,z_star,sx_star),1)
    x_input = 2.0 *(x_input-lb)/(ub-lb) - 1.0
    x_input = embedding(x_input)
    x_input = torch.cat((x_input, latent.repeat(9*grid_size**2).view(-1,latent_dim)),1)
    with torch.no_grad():
        if gpt:
            dU_real_star, dU_imag_star = model.forward(test_data=x_input)
        else:
            dU_real_star, dU_imag_star = model(x_input)
    return dU_real_star, dU_imag_star

def get_train_images(num):
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)

class Encoder(nn.Module):
    
    def __init__(self, 
                 num_input_channels : int, 
                 base_channel_size : int, 
                 latent_dim : int, 
                 act_fn : object = nn.GELU):
        """
        Inputs: 
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 128 => 64
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 64 => 32
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 32 => 16
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16 => 4
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16 => 4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            # nn.Linear(2*16**2*c_hid, latent_dim)
            nn.Linear(2*4**2*c_hid, latent_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Decoder(nn.Module):
    
    def __init__(self, 
                 num_input_channels : int, 
                 base_channel_size : int, 
                 latent_dim : int, 
                 act_fn : object = nn.GELU):
        """
        Inputs: 
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            # nn.Linear(latent_dim, 2*16**2*c_hid),
            nn.Linear(latent_dim, 2*4**2*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4 => 8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8 => 16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 16 => 32
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 32 => 64
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 64 => 128
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )
    
    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x
    
class AutoEncoder(pl.LightningModule):
    
    def __init__(self, 
                 base_channel_size: int, 
                 latent_dim: int, 
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 num_input_channels: int = 1, 
                 width: int = 128, 
                 height: int = 128):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters() 
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)
        
    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x = batch # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='min', 
                                                         factor=0.5, 
                                                         patience=30, 
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
    
    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)                             
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)
    
    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)
        
    def encode(self, x):
        """
        The forward function takes in an image and returns the latent variables
        """
        z = self.encoder(x)
        # x_hat = self.decoder(z)
        return z
    
    def encode(self, z):
        """
        The forward function takes in a latent variables and returns the reconstructed image
        """
        x_hat = self.decoder(z)
        return x_hat
    
# class Encoder(nn.Module):
    
#     def __init__(self, 
#                  num_input_channels : int, 
#                  base_channel_size : int, 
#                  latent_dim : int, 
#                  act_fn : object = nn.GELU):
#         """
#         Inputs: 
#             - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
#             - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
#             - latent_dim : Dimensionality of latent representation z
#             - act_fn : Activation function used throughout the encoder network
#         """
#         super().__init__()
#         c_hid = base_channel_size
#         self.net = nn.Sequential(
#             nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=4), # 256 => 128
#             act_fn(),
#             nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
#             act_fn(),
#             nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=4), # 128 => 64
#             act_fn(),
#             nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
#             act_fn(),
#             nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=4), # 64 => 32
#             act_fn(),
#             # nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
#             # act_fn(),
#             # nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 32 => 16
#             # act_fn(),
#             # nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
#             # act_fn(),
#             # nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16 => 8
#             # act_fn(),
#             # nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
#             # act_fn(),
#             # nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8 => 4
#             # act_fn(),
#             nn.Flatten(), # Image grid to single feature vector
#             # nn.Linear(2*32**2*c_hid, 2*4**2*c_hid),
#             nn.Linear(2*4**2*c_hid, latent_dim)
#         )
    
#     def forward(self, x):
#         return self.net(x)
    
# class Decoder(nn.Module):
    
#     def __init__(self, 
#                  num_input_channels : int, 
#                  base_channel_size : int, 
#                  latent_dim : int, 
#                  act_fn : object = nn.GELU):
#         """
#         Inputs: 
#             - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
#             - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
#             - latent_dim : Dimensionality of latent representation z
#             - act_fn : Activation function used throughout the decoder network
#         """
#         super().__init__()
#         c_hid = base_channel_size
#         self.linear = nn.Sequential(
#             # nn.Linear(latent_dim, 2*16**2*c_hid),
#             nn.Linear(latent_dim, 2*4**2*c_hid),
#             # nn.Linear(2*4**2*c_hid, 2*32**2*c_hid),
#             act_fn()
#         )
#         self.net = nn.Sequential(
#             nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=5, output_padding=1, padding=1, stride=4), # 4 => 8
#             act_fn(),
#             nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
#             act_fn(),
#             nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=5, output_padding=1, padding=1, stride=4), # 8 => 16
#             act_fn(),
#             nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
#             act_fn(),
            
#             # nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 16 => 32
#             # act_fn(),
#             # nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
#             # act_fn(),
#             # nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 32 => 64
#             # act_fn(),
#             # nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
#             # act_fn(),
#             # nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 64 => 128
#             # act_fn(),
#             # nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
#             # act_fn(),
            
#             nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=5, output_padding=1, padding=1, stride=4), # 128 => 256
#             nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
#         )
    
#     def forward(self, x):
#         x = self.linear(x)
#         x = x.reshape(x.shape[0], -1, 4, 4)
#         x = self.net(x)
#         return x
    
# class AutoEncoder(pl.LightningModule):
    
#     def __init__(self, 
#                  base_channel_size: int, 
#                  latent_dim: int, 
#                  encoder_class : object = Encoder,
#                  decoder_class : object = Decoder,
#                  num_input_channels: int = 1, 
#                  width: int =256, 
#                  height: int = 256):
#         super().__init__()
#         # Saving hyperparameters of autoencoder
#         self.save_hyperparameters() 
#         # Creating encoder and decoder
#         self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
#         self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
#         # Example input array needed for visualizing the graph of the network
#         self.example_input_array = torch.zeros(2, num_input_channels, width, height)
        
#     def forward(self, x):
#         """
#         The forward function takes in an image and returns the reconstructed image
#         """
#         z = self.encoder(x)
#         x_hat = self.decoder(z)
#         # print(x_hat.shape)
#         return x_hat
    
#     def _get_reconstruction_loss(self, batch):
#         """
#         Given a batch of images, this function returns the reconstruction loss (MSE in our case)
#         """
#         x = batch # We do not need the labels
#         x_hat = self.forward(x)
#         loss = F.mse_loss(x, x_hat, reduction="none")
#         loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
#         return loss
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
#         # Using a scheduler is optional but can be helpful.
#         # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
#                                                          mode='min', 
#                                                          factor=0.5, 
#                                                          patience=20, 
#                                                          min_lr=5e-5)
#         return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
    
#     def training_step(self, batch, batch_idx):
#         loss = self._get_reconstruction_loss(batch)                             
#         self.log('train_loss', loss)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         loss = self._get_reconstruction_loss(batch)
#         self.log('val_loss', loss)
    
#     def test_step(self, batch, batch_idx):
#         loss = self._get_reconstruction_loss(batch)
#         self.log('test_loss', loss)
        
#     def encode(self, x):
#         """
#         The forward function takes in an image and returns the latent variables
#         """
#         z = self.encoder(x)
#         # x_hat = self.decoder(z)
#         return z
    
#     def encode(self, z):
#         """
#         The forward function takes in a latent variables and returns the reconstructed image
#         """
#         x_hat = self.decoder(z)
#         return x_hat

class GenerateCallback(pl.Callback):
    
    def __init__(self, input_imgs, every_n_epochs=1, name='test'):
        super().__init__()
        self.input_imgs = input_imgs # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.name = name
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
            plot_square_image((2+reconst_imgs.detach().cpu().numpy())*2, 4, 4, 
                              'recon_'+self.name+'_'+str(trainer.current_epoch)+'.pdf', '../saves/', wavefield=True)
            # trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)
            trainer.logger.log_image(key="samples", images=[grid])
            
def embed_imgs(model, data_loader):
    # Encode all images in the data_laoder using model, and return both images and encodings
    img_list, embed_list, dec_list = [], [], []
    model.eval()
    for imgs in tqdm(data_loader, desc="Encoding images", leave=False):
        with torch.no_grad():
            z = model.encoder(imgs.to(model.device))
            dec_imgs = model(imgs.to(model.device))
        img_list.append(imgs)
        embed_list.append(z)
        dec_list.append(dec_imgs)
    return (torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0), torch.cat(dec_list, dim=0))


class FullyConnectedNetwork(nn.Module):
    def __init__(
        self,
        num_input,
        num_output,
        n_hidden=[16, 32],
        lay="linear",
        act="tanh",
        last_act=None,
        last_multiplier=1,
        last_abs=False
    ):
        super(FullyConnectedNetwork, self).__init__()
        self.lay = lay
        self.act = act
        self.last_multiplier = last_multiplier
        self.last_act = last_act
        self.last_abs = last_abs

        act = activation(act)
        lay = layer(lay)
        if last_act == "sigmoid":
            self.model = nn.Sequential(
                nn.Sequential(lay(num_input, n_hidden[0]), act),
                *[
                    nn.Sequential(lay(n_hidden[i], n_hidden[i + 1]), act)
                    for i in range(len(n_hidden) - 1)
                ],
                lay(n_hidden[-1], num_output),
                nn.Sigmoid(),
            )
        elif last_act == "relu":
            self.model = nn.Sequential(
                nn.Sequential(lay(num_input, n_hidden[0]), act),
                *[
                    nn.Sequential(lay(n_hidden[i], n_hidden[i + 1]), act)
                    for i in range(len(n_hidden) - 1)
                ],
                lay(n_hidden[-1], num_output),
                nn.ReLU(),
            )
        elif last_act == "tanh":
            self.model = nn.Sequential(
                nn.Sequential(lay(num_input, n_hidden[0]), act),
                *[
                    nn.Sequential(lay(n_hidden[i], n_hidden[i + 1]), act)
                    for i in range(len(n_hidden) - 1)
                ],
                lay(n_hidden[-1], num_output),
                nn.Tanh(),
            )
        else:
            self.model = nn.Sequential(
                nn.Sequential(lay(num_input, n_hidden[0]), act),
                *[
                    nn.Sequential(lay(n_hidden[i], n_hidden[i + 1]), act)
                    for i in range(len(n_hidden) - 1)
                ],
                lay(n_hidden[-1], num_output),
            )

    def forward(self, x):
        x = self.model(x)  # / (1-0.9999)
        if self.last_abs:
            x = torch.abs(x)
        return x * self.last_multiplier
    
def activation(act_fun="leakyrelu"):

    if isinstance(act_fun, str):
        if act_fun == "leakyrelu":
            return nn.LeakyReLU(0.2, inplace="y")
        elif act_fun == "elu":
            return nn.ELU()
        elif act_fun == "relu":
            return nn.ReLU()
        elif act_fun == "tanh":
            return nn.Tanh()
        elif act_fun == "swish":
            return Swish()
        else:
            raise ValueError(f"{act_fun} is not an activation function...")
    else:
        return act_fun


def layer(lay="linear"):
    if isinstance(lay, str):
        if lay == "linear":
            return lambda x, y: nn.Linear(x, y)

        elif lay == "adaptive":
            return lambda x, y: AdaptiveLinear(
                x, y, adaptive_rate=0.1, adaptive_rate_scaler=10.0
            )
        else:
            raise ValueError(f"{lay} is not a layer type...")
    else:
        return lay