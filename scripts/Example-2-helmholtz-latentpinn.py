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
import h5py

from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.autograd import Variable, grad
from torch.optim import Adam, LBFGS, RMSprop
from torch.nn import Parameter
from scipy.io import loadmat

from latentpinn.utils import *
from latentpinn.model import *

plt.style.use("../science.mplstyle")

from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser

def main():
    
    LATENT_DIM = args.latent_dim
    
    np.random.seed(12345678)
    torch.manual_seed(12345678)

    train_latent = torch.from_numpy(np.load('../saves/marine_128/latent_'+str(LATENT_DIM)+'/train_latent_'+str(LATENT_DIM)+'.npy')).cuda()
    train_latent = train_latent/abs(train_latent).max()

    # Parameters setting
    NUM_TRAIN_VEL = 2000
    fre = 4.0 # Hz
    PI = 3.14159
    omega = 2.0 * PI * fre
    radius = [10e2]
    alphas = [args.alpha]
    NUM_TEST_VEL= 1000 #00
    niter = 1000000

    layers = [27+LATENT_DIM, 64, 64, 32, 32, 16, 16, 8, 8, 2]

    # Load analytical wavefield solutions
    test_data = h5py.File('../scripts/smooth5mix_reseam_4Hz_test_data_NS40000_NV2000_128.mat')
    
    dU_real_star = test_data['dU_real_star'][()].T.reshape(NUM_TRAIN_VEL,-1,1)
    dU_imag_star = test_data['dU_imag_star'][()].T.reshape(NUM_TRAIN_VEL,-1,1)
    v_star = test_data['v_star'][()].T.reshape(NUM_TRAIN_VEL,-1,1)

    x_star = test_data['x_star'][()].T.reshape(-1,1)
    z_star = test_data['z_star'][()].T.reshape(-1,1)
    sx_star = test_data['sx_star'][()].T.reshape(-1,1)

    train_data = h5py.File('../scripts/smooth5mix_reseam_4Hz_train_data_NS40000_NV2000_128.mat')

    u0_real_train_all = train_data['U0_real_train'][()].T.reshape(NUM_TRAIN_VEL,-1,1)
    u0_imag_train_all = train_data['U0_imag_train'][()].T.reshape(NUM_TRAIN_VEL,-1,1)
    x_train_all = train_data['x_train'][()].T.reshape(NUM_TRAIN_VEL,-1,1)
    y_train_all = train_data['z_train'][()].T.reshape(NUM_TRAIN_VEL,-1,1)
    sx_train_all = train_data['sx_train'][()].T.reshape(NUM_TRAIN_VEL,-1,1)
    m_train_all = train_data['m_train'][()].T.reshape(NUM_TRAIN_VEL,-1,1)
    m0_train_all = train_data['m0_train'][()].T.reshape(NUM_TRAIN_VEL,-1,1)

    embedding_fn_g, _ = get_embedder1(4,3,0)

    np.random.seed(1234)
    torch.manual_seed(1234)

    model = PhysicsInformedNN(layers)
    model.apply(weight_init)
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0])
    cudnn.benchmark = True

    if args.use_wandb=='y':
        wandb.init(project='LatentPINNs-02-ScatteredHelmholtz')
        wandb.run.log_code(".")
        wandb_dir = wandb.run.dir + '/'
        wandb.config.update(args)
    else:      
        wandb_dir = './wandb/latest-run/'

    print('Saved at: ' + wandb_dir)

    for radii in radius:

        for alpha in alphas:

            if(1):
                model = PhysicsInformedNN(layers)
                model.apply(weight_init)
                model.cuda()
                model = torch.nn.DataParallel(model, device_ids=[0])
                cudnn.benchmark = True

                # Load data
                X = np.concatenate([x_train_all,y_train_all,sx_train_all],2)
                lb_all = torch.FloatTensor(X.min(1)).cuda()
                ub_all = torch.FloatTensor(X.max(1)).cuda()
                X = torch.FloatTensor(X)
                x_train = torch.FloatTensor(X[:,:,0:1]).cuda()
                y_train = torch.FloatTensor(X[:,:,1:2]).cuda()
                sx_train =torch.FloatTensor(X[:,:,2:3]).cuda()
                u0_real_train = torch.FloatTensor(u0_real_train_all).cuda()
                u0_imag_train = torch.FloatTensor(u0_imag_train_all).cuda()
                
                x_bc = torch.FloatTensor(x_star.reshape(9,128,128)).cuda()
                z_bc = torch.FloatTensor(z_star.reshape(9,128,128)).cuda()
                sx_bc = torch.FloatTensor(sx_star.reshape(9,128,128)).cuda()
                dur_bc = torch.FloatTensor(test_data['dU_real_star'][()].T.reshape(-1,9,128,128)).cuda()
                dui_bc = torch.FloatTensor(test_data['dU_imag_star'][()].T.reshape(-1,9,128,128)).cuda()

                pinn_train(model,x_train,y_train,sx_train, 
                           u0_real_train, u0_imag_train,
                           lb_all,ub_all,
                           m_train_all,m0_train_all,
                           omega,niter,embedding_fn_g,fre,
                           x_bc, z_bc, sx_bc, dur_bc, dui_bc, args.boundary,
                           NUM_TEST_VEL,train_latent,alpha,radii,0,args.latent_dim,wandb_dir,
                           grid_size=128, use_wandb=args.use_wandb
                           )

                # Save model
                state = {
                    'net':model.module,
                    'lb':lb_all.cpu().numpy(),
                    'ub':ub_all.cpu().numpy(),
                }
                torch.save(state,wandb_dir+'lpe_pinnmodel_all_n30'+str(alpha)+str(radii)+'.tz')
                
     # Inference
    NUM_TEST_VEL = 80

    # Load analytical wavefield solutions
    marm_data = h5py.File('../scripts/smooth5mix_marm_4Hz_test_data_NS40000_NV80_128.mat')

    dU_real_star = marm_data['dU_real_star'][()].T.reshape(NUM_TEST_VEL,-1,1)
    dU_imag_star = marm_data['dU_imag_star'][()].T.reshape(NUM_TEST_VEL,-1,1)
    v_star = marm_data['v_star'][()].T.reshape(NUM_TEST_VEL,-1,1)

    x_star = marm_data['x_star'][()].T.reshape(-1,1)
    z_star = marm_data['z_star'][()].T.reshape(-1,1)
    sx_star = marm_data['sx_star'][()].T.reshape(-1,1)
    s_idxs = [0,2,4,8,0,2,4,8,0,2,4,8,0,2,4,8]
    idxs = np.arange(16)

    v_test = v_star.reshape(-1,9,128,128)[idxs,s_idxs,:,:]#[-25:]
    dU_real_test = dU_real_star.reshape(-1,9,128,128)[idxs,s_idxs,:,:]#[-25:]
    dU_imag_test = dU_imag_star.reshape(-1,9,128,128)[idxs,s_idxs,:,:]#[-25:]
    dU_real_pred_pinn, dU_imag_pred_pinn, dU_real_error_pinn, dU_imag_error_pinn = np.ones_like(v_test),np.ones_like(v_test),np.ones_like(v_test),np.ones_like(v_test)

    lb = torch.FloatTensor(torch.load('../saves/result.tz')['lb']).cuda()
    ub = torch.FloatTensor(torch.load('../saves/result.tz')['ub']).cuda()

    marm_latent = torch.from_numpy(np.load('../saves/marine_128/latent_'+str(LATENT_DIM)+'/marm_latent_'+str(LATENT_DIM)+'.npy')).cuda()
    marm_latent = train_latent/abs(marm_latent).max()
    
    for idx in range(len(idxs)):

        # s_idx = np.random.randint(3)*4

        temp_real, temp_imag = pinn_predict(
            model,torch.FloatTensor(x_star).cuda(),
            torch.FloatTensor(z_star).cuda(), 
            torch.FloatTensor(sx_star).cuda(),
            lb[idxs[idx]],ub[idxs[idx]],
            embedding_fn_g,1,
            marm_latent[idxs[idx]],
            grid_size=128
        )
        dU_real_pred_pinn[idx], dU_imag_pred_pinn[idx] = temp_real.reshape(9,128,128)[s_idxs[idx],:,:].detach().cpu().numpy(), temp_imag.reshape(9,128,128)[s_idxs[idx],:,:].detach().cpu().numpy()        
        dU_real_error_pinn[idx] = (dU_real_test[idx]-dU_real_pred_pinn[idx])#/np.linalg.norm(dU_real_star[idx],2)
        dU_imag_error_pinn[idx] = (dU_imag_test[idx]-dU_imag_pred_pinn[idx])#/np.linalg.norm(dU_imag_star[idx],2)
        
    plot_square_wavefield(
        dU_real_error_pinn,4,4, save_dir=wandb_dir, 
        name='real_error_pinn.pdf', vmin=-1.2, vmax=1.2, cmap='jet',
        extent=[0, 2.5, 0, 2.5]
    )
    plot_square_wavefield(
        dU_real_test,4,4, save_dir=wandb_dir, 
        name='real_test.pdf', vmin=-1.2, vmax=1.2, cmap='jet',
        extent=[0, 2.5, 0, 2.5]
    )
    plot_square_wavefield(
        dU_real_pred_pinn,4,4, save_dir=wandb_dir, 
        name='real_pred_pinn.pdf', vmin=-1.2, vmax=1.2, cmap='jet',
        extent=[0, 2.5, 0, 2.5]
    )
    
    print('Max Error Real PINN: ', dU_real_error_pinn.max())
    print('Min Error Real PINN: ', dU_real_error_pinn.max())
    print('Mean Error Real PINN: ', dU_real_error_pinn.mean())
    
    np.save(wandb_dir+'real_pred_pinn.npy', dU_real_pred_pinn)
    np.save(wandb_dir+'real_error_pinn.npy', dU_real_error_pinn)
    
    if args.use_wandb=="y":
        wandb.log({
            'Max Error Real PINN: ': dU_real_error_pinn.max(),
            'Min Error Real PINN: ': dU_real_error_pinn.max(),
            'Mean Error Real PINN: ': dU_real_error_pinn.mean(),
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
        "--latent_dim",
        type=int,
        default=12,
        help="Neurons width.",
    )
    parser.add_argument(
        "--use_wandb",
        type=str,
        default="n",
        help="Whether we use weight and biases to keep track of the experiments.",
    )

    args = parser.parse_args()
    
    main()
