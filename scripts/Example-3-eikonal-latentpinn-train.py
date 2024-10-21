import torch
import numpy as np
import scipy.io
import time
import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import wandb

from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.autograd import Variable, grad
from torch.optim import Adam, LBFGS, RMSprop
from torch.nn import Parameter
from scipy.io import loadmat
from skimage.transform import resize

from latentpinn.utils import *
from latentpinn.model import *
from latentpinn.modeling2d import *
from latentpinn.plots import *

plt.style.use("../science.mplstyle")

from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from ptflops import get_model_complexity_info
import re

import os 

def main():
    
    dict_args = vars(args)
    
    np.random.seed(12345678)
    torch.manual_seed(12345678)

    train_latent = torch.from_numpy(np.load('../saves/train_latent_12.npy')).cuda()
    train_latent = train_latent/abs(train_latent).max()

    train_vel = loadmat('../data/reseam_train_vel.mat')['vel_train']

    train_vel = torch.from_numpy(resize(train_vel[:,:,:], (34000,128,128)))
    
    print(train_vel.shape)
    
    # Setup
    if args.use_wandb=='y':
        wandb.init(project='LatentPINNs-03-Eikonal')
        wandb.run.log_code(".")
        wandb_dir = wandb.run.dir + '/'
        wandb.config.update(args)
    else:
        wandb_dir = './LatentPINN-03-eikonal-latentpinn-try'
        try: 
            os.mkdir(wandb_dir) 
        except OSError as error: 
            print(error) 
        
    dict_args['save_folder'] = wandb_dir

    seed = 12315019

    set_seed(seed)
    device = set_device()
    
    print(dict_args)

    # Computational model parameters
    zmin = 0; zmax = 1;
    xmin = 0.; xmax = 5;

    z = np.linspace(zmin,zmax,128)
    nz = z.size

    x = np.linspace(xmin,xmax,128)
    nx = x.size

    Z,X = np.meshgrid(z,x,indexing='ij')

    deltax, deltaz = x[1]-x[0], z[1]-z[0]

    # Number of training points
            
    # Sources indices
    num_vel = args.num_vel

    idx_all = np.arange(X.size).reshape(X.shape)
    id_sou = idx_all[len(z)//2, len(x)//2].reshape(-1)

    sz = Z.reshape(-1)[id_sou]
    sx = X.reshape(-1)[id_sou]

    SX, SZ = sx*np.ones_like(X), sz*np.ones_like(Z)

    vel2d = train_vel[0]

    # Extending the velocity model in thirs dimension byy repeatin the array
    velmodel = np.repeat(vel2d[...,np.newaxis], sx.size, axis=2)

    # ZX plane after
    plot_section(vel2d, 'v_true_zx.pdf', vmin=np.nanmin(velmodel)+0.1, 
                vmax=np.nanmax(velmodel)-0.5, save_dir=wandb_dir, aspect='auto',
                xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, 
                sx=X.reshape(-1)[id_sou],sz=Z.reshape(-1)[id_sou])

    vs = 3 #km/s

    T0 = np.sqrt((Z-SZ)**2 + (X-SX)**2)/vs;
    px0 = np.divide(X-SX, T0*vs**2, out=np.zeros_like(T0), where=T0!=0)
    pz0 = np.divide(Z-SZ, T0*vs**2, out=np.zeros_like(T0), where=T0!=0)

    # Locate source boolean
    sids = id_sou

    # Locate source boolean
    isource = np.ones_like(X).reshape(-1,).astype(bool)
    isource[sids] = False

    velmodel = vel2d.reshape(-1,1)
    px0 = px0.reshape(-1,1)
    pz0 = pz0.reshape(-1,1)
    T0 = T0.reshape(-1,1)
    index = idx_all.reshape(-1,1)

    perm_id = np.random.permutation(X.size-sx.size)

    # input_unnorm = [X, Z, SX, SZ, T0, px0, pz0, index]
    # input_wsrc = [i/(X.max()-X.min()) for i in input_unnorm]
    input_wsrc = [X, Z, SX, SZ, T0, px0, pz0, index]
    input_wosrc = [i.ravel()[isource.reshape(-1)][perm_id] for i in input_wsrc]

    # Network
    lay = 'linear'
    opttype = 'adam'
    lr = args.learning_rate

    torch.manual_seed(seed)
    tau_model = FullyConnectedNetwork(4+train_latent.shape[1], 1, [args.num_neurons]*args.num_layers, 
                                    last_act='tanh', act='elu', 
                                    lay=lay, last_multiplier=1,
                                    last_abs=True)
    tau_model.to(device)
    # tau_model.apply(lambda m: init_weights(m, init_type=ini, bias=bias, mean=mean, std=std))

    npoints = int(X.size * 0.6)
    ipermute = np.random.permutation(np.arange(X.size))[:npoints]

    # Compute traveltime with randomly initialized network
    pde_loader, ic = create_dataloader2dmodeling([i.ravel() for i in input_wsrc], sx, sz,
                                    shuffle=False, batch_size=Z.size//400, fast_loader=True, perm_id=ipermute)

    v_init = evaluate_velocity2d(tau_model, train_latent[0].to(device), pde_loader, Z.size, batch_size=Z.size//400, device=device).detach().cpu().numpy()

    # Optimizer
    if opttype == 'adam':
        optimizer = torch.optim.Adam(list(tau_model.parameters()), lr=lr, betas=(0.9, 0.999), eps=1e-5)
    elif opttype == 'lbfgs':
        optimizer = torch.optim.LBFGS(list(tau_model.parameters()), line_search_fn="strong_wolfe")
        
    # ZX plane after
    plot_section(v_init.reshape(X.shape), 'v_init_zx.pdf', vmin=np.nanmin(velmodel)+0.1, 
                vmax=np.nanmax(velmodel)-0.5, save_dir=wandb_dir, aspect='auto',
                xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, 
                sx=X.reshape(-1)[id_sou],sz=Z.reshape(-1)[id_sou])

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,500,800], gamma=0.1)
    
    # Compute complexity
    macs, params = get_model_complexity_info(
        tau_model, (3, 4+train_latent.shape[1]), 
        as_strings=True, print_per_layer_stat=True, verbose=True
    )
    # Extract the numerical value
    flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
    # Extract the unit
    flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]

    print('Computational complexity: {:<8}'.format(macs))
    print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
    print('Number of parameters: {:<8}'.format(params))
    
    # Training
    import time
    start_time = time.time()

    loss_history = \
        training_loop2d(
        input_wosrc, sx, sz,
        tau_model, train_vel, train_latent, 
        optimizer, Z.size, npoints,
        out_epochs=args.num_epochs, 
        batch_size=Z.size//150, device=device, scheduler=scheduler, 
        fast_loader=True, args=dict_args, num_vel=num_vel
    )
    elapsed = time.time() - start_time
    print('Training time: %.2f minutes' %(elapsed/60.))

    # Convergence history plot for verification
    fig = plt.figure()
    ax = plt.axes()
    ax.semilogy(loss_history)
    ax.set_xlabel('Epochs',fontsize=14)
    plt.xticks(fontsize=11)
    ax.set_ylabel('Loss',fontsize=14)
    plt.yticks(fontsize=11)
    plt.grid()
    plt.savefig(os.path.join(wandb_dir, "loss.pdf"), format='pdf', bbox_inches="tight")

    # Save model
    torch.save({
            'tau_model_state_dict': tau_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_history
    }, wandb_dir+'/saved_model')
    
    # Set sample index
    test_latent = torch.from_numpy(np.load('../saves/test_latent_12.npy')).cuda()
    test_latent = test_latent/abs(test_latent).max()

    test_vel = loadmat('../data/reseam_test_vel.mat')['vel_train']

    test_vel = torch.from_numpy(resize(test_vel[:,:,:], (1000,128,128)))
    sim_idx = torch.arange(64*10)[::10].reshape(8,8) #torch.load(wandb_dir+'/reseam_similar_idx.pt').to(int)[:100,:8]

    v_preds = np.zeros((8,8,128,128))
    T_preds = np.zeros((8,8,128,128))
    T_datas = np.zeros((8,8,128,128))
    
    import time
    start_time = time.time()

    for i in range(sim_idx.shape[0]):
        
        for j in range(sim_idx.shape[1]):
        
            vel_idx = sim_idx[i,j] #5788 #30205
            vel = test_vel[vel_idx]
            
            # Augment a 2d velocity volume from 2D data
            vel2d = test_vel[vel_idx]

            # To load
            checkpoint = torch.load(wandb_dir+'/saved_model')
            tau_model.load_state_dict(checkpoint['tau_model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Prediction
            pde_loader, _ = create_dataloader2dmodeling([i.ravel() for i in input_wsrc], sx, sz,
                                            shuffle=False, batch_size=Z.size, fast_loader=True, perm_id=ipermute)

            v_pred = evaluate_velocity2d(tau_model, test_latent[vel_idx].to(device), pde_loader, Z.size, batch_size=Z.size, device=device)

            tau_pred = evaluate_tau2d(tau_model, test_latent[vel_idx].to(device), pde_loader, Z.size, batch_size=Z.size, device=device)

            v_preds[i,j,:,:] = v_pred.detach().cpu().numpy().reshape(Z.shape)

            T_datas[i,j,:,:] = numerical_traveltime2d(
                vel2d,
                nx,
                1,
                nz,
                len(id_sou),
                xmin,
                xmin,
                zmin,
                deltax,
                deltax,
                deltaz,
                np.array([64]),
                np.array([0]),
                np.array([64]),
            )[:,0,:,0]

            # tau_data = np.divide(T_data, T0.reshape(X.shape), out=np.ones_like(T0.reshape(X.shape)), where=T0.reshape(X.shape)!=0)

            T_preds[i,j,:,:] = tau_pred.reshape(Z.shape).detach().cpu() * T0.reshape(Z.shape)
    elapsed = time.time() - start_time
    print('Testing time: %.2f minutes' %(elapsed/60.))
        
    v_trues = np.zeros_like(v_preds)
    for i in range(sim_idx.shape[0]):
        
        for j in range(sim_idx.shape[1]):
            
            v_trues[i,j,:,:] = test_vel[sim_idx[i,j],:,:]
            
    
    print(v_trues.shape)
            
    plot_square_image(v_trues, 8,8, save_dir=wandb_dir, name='v_true_8x8.pdf')
    
    plot_square_contour(T_preds, T_datas, 8,8, save_dir=wandb_dir, name='t_pred_8x8.pdf')
        
    plot_square_image(v_preds, 8,8, save_dir=wandb_dir, name='v_pred_8x8.pdf')
    
    np.save(wandb_dir+'v_true.npy', v_trues)
    np.save(wandb_dir+'v_pred.npy', v_preds)
    np.save(wandb_dir+'T_pred.npy', T_preds)
    
    # Evalutaion metrics
    v_rmse = []
    v_ssim = []
    
    T_rmse = []
    T_ssim = []
    
    # Remove NaNs
    v_preds[:,:,64,64] = 0
    v_trues[:,:,64,64] = 0
    
    T_preds[:,:,64,64] = 0
    T_datas[:,:,64,64] = 0
    
    for i in range(8):
        for j in range(8):
            v_rmse.append(rmse(org_img=v_trues[i,j].reshape(1,128,128), pred_img=v_preds[i,j].reshape(1,128,128)))
            v_ssim.append(ssim(org_img=v_trues[i,j].reshape(1,128,128), pred_img=v_preds[i,j].reshape(1,128,128)))
            
            T_rmse.append(rmse(org_img=T_trues[i,j].reshape(1,128,128), pred_img=T_preds[i,j].reshape(1,128,128)))
            T_ssim.append(ssim(org_img=T_trues[i,j].reshape(1,128,128), pred_img=T_preds[i,j].reshape(1,128,128)))
    
    print('Max RMSE-T: ',  np.array([T_rmse]).max())
    print('Min RMSE-T: ',  np.array([T_rmse]).min())
    print('Max SSIM-T: ',  np.array([T_ssim]).max())
    print('Min SSIM-T: ',  np.array([T_ssim]).min())
    
    np.save(wandb_dir+'v_rmse_ssim.npy', np.array([v_rmse, v_ssim]))
    np.save(wandb_dir+'T_rmse_ssim.npy', np.array([T_rmse, T_ssim]))
    
    if args.use_wandb=="y":
        wandb.log({
            "Max v_rmse": np.array([v_rmse]).max(),
            "Min v_rmse": np.array([v_rmse]).min(),
            "Max v_ssim": np.array([v_ssim]).max(),
            "Min v_ssim": np.array([v_ssim]).min(),
            "Max T_rmse": np.array([T_rmse]).max(),
            "Min T_rmse": np.array([T_rmse]).min(),
            "Max T_ssim": np.array([T_ssim]).max(),
            "Min T_ssim": np.array([T_ssim]).min(),
            "gpt-macs": args.meta_neurons * macs,
            "gpt-flops": args.meta_neurons * flops,
            "gpt-params": args.meta_neurons * params,
            "macs":macs,
            "flops":flops,
            "params":params
        })
                

if __name__ == '__main__':
    
    parser = ArgumentParser(description="LatentPINN training.")
    parser.add_argument(
            "--alpha",
            type=float,
            default=1e4,
            help="Regularization weight."
    )
    parser.add_argument(
            "--boundary",
            type=str,
            default='all',
            help="Type of boundary information used."
    )
    parser.add_argument(
        "--num_neurons",
        type=int,
        default=20,
        help="Neurons width.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Neurons width.",
    )
    parser.add_argument(
        "--num_vel",
        type=int,
        default=100,
        help="Neurons width.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=10,
        help="Layers depth.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--use_wandb",
        type=str,
        default="n",
        help="Whether we use weight and biases to keep track of the experiments.",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default='./',
        help="Whether we use weight and biases to keep track of the experiments.",
    )
    args = parser.parse_args()
    
    main()
