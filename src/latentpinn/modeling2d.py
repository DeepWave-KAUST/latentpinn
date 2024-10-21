import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import time
import random
import os
import wandb

from torch.optim.lr_scheduler import ReduceLROnPlateau
from latentpinn.utils import create_dataloader2dmodeling

# Load style @hatsyim
# plt.style.use("~/science.mplstyle")

# Training functions
import random
import numpy as np
from zmq import device
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def numerical_traveltime2d(
    vel,
    nx,
    ny,
    nz,
    ns,
    xmin,
    ymin,
    zmin,
    deltax,
    deltay,
    deltaz,
    id_sou_x,
    id_sou_y,
    id_sou_z,
):

    import pykonal

    T_data_surf = np.zeros((nz, ny, nx, ns))

    for i in range(ns):

        solver = pykonal.EikonalSolver(coord_sys="cartesian")
        solver.velocity.min_coords = zmin, ymin, xmin
        solver.velocity.node_intervals = deltaz, deltay, deltax
        solver.velocity.npts = nz, ny, nx
        solver.velocity.values = vel.reshape(nz, ny, nx)

        src_idx = id_sou_z[i], id_sou_y[i], id_sou_x[i]

        solver.traveltime.values[src_idx] = 0
        solver.unknown[src_idx] = False
        solver.trial.push(*src_idx)

        solver.solve()

        T_data_surf[:, :, :, i] = solver.traveltime.values

    return T_data_surf


def train2d(
    input_wosrc,
    sx,
    sz,
    tau_model,
    v_sample,
    latent,
    optimizer,
    grid_size,
    num_pts,
    batch_size,
    vscaler,
    scheduler,
    fast_loader,
    device,
    args,
    num_vel,
    transfer=False
):
    tau_model.train()
    loss = []

    # Whether a full or subgrid size is used
    # if grid_size==num_pts:
    #     ipermute = torch.randperm(grid_size)[:num_pts]
    # else:
    #     ipermute = None

    data_loader, ic = create_dataloader2dmodeling(
        input_wosrc,
        sx,
        sz,
        batch_size,
        shuffle=True,
        device=device,
        fast_loader=fast_loader,
        perm_id=None,
    )

    sid = torch.arange(sx.size).float().to(device)

    latent_dim = latent.shape[0]

    for xz, sx, sz, t0, t0_dx, t0_dz, idx in data_loader:
        
        if transfer:
            v_idx = torch.randint(num_vel, (1,), device=device)[0]+1000
        else:
            v_idx = torch.randint(num_vel, (1,), device=device)[0]#+100
        # print(v_idx)

        # Latent
        z = latent[v_idx].to(device).repeat(sx.size()[0]).view(-1, latent.shape[1])

        # Input for the data network
        xz.requires_grad = True
        xzic = torch.cat([xz, ic.view(1, -1)])
        s = torch.hstack((sx.view(-1, 1).view(-1, 1), sz.view(-1, 1)))
        sic = torch.cat([s, ic.view(1, -1)])
        xzsic = torch.hstack((xz, s, z))

        # Compute T
        tau = tau_model(xzsic).view(-1)

        # Compute v
        v = (
            v_sample[v_idx]
            .to(device)
            .view(128, -1, 128)
            .view(-1)[idx.long()]
        )

        # Gradients
        gradient = torch.autograd.grad(
            tau, xzsic, torch.ones_like(tau), create_graph=True
        )[0]

        tau_dx = gradient[:, 0]
        tau_dz = gradient[:, 1]
        T_dx = tau_dx * t0 + tau * t0_dx
        T_dz = tau_dz * t0 + tau * t0_dz
        pde_lhs = T_dx**2 + T_dz**2 #+ 1e-8
        
        pde = torch.mean(abs(torch.sqrt(torch.where(1/pde_lhs==torch.inf,v,1/pde_lhs)) - v))
        # pde = torch.mean(abs(pde_lhs - 1 / (v ** 2)))
        
        # T0 = torch.sqrt((xz[:,0]-sx)**2 + (xz[:,1]-sz)**2)
        # T1 = (t0**2)*(tau_dx**2 + tau_dz**2)
        # T2 = 2*tau*(tau_dx*(xz[:,0]-sx) + tau_dz*(xz[:,1]-sz))
        # T3 = tau**2
        # pde_lhs = torch.sqrt(1/(T1+T2+T3))
        # pde = torch.mean(abs(pde_lhs-v)) + torch.sum(torch.abs(tau[-1]-1/v_sample[64,64]))
        # pde = torch.mean(abs(torch.sqrt(1 / pde_lhs) - v))

        # bc = torch.mean((tau[-1] - 1) ** 2)
        # bc = torch.mean((tau_model(xzs).view(-1) - 1) ** 2)
        # bc = abs((tau_model(sic).view(-1)-1)/tau_model(sic).view(-1))

        ls = pde #+ bc
        loss.append(ls.item())
        ls.backward()
        optimizer.step()
        optimizer.zero_grad()

        del (
            idx,
            z,
            s,
            xz,
            xzic,
            sic,
            xzsic,
            t0,
            t0_dx,
            t0_dz,
            ls,
            v,
            tau,
            tau_dx,
            tau_dz,
            T_dx,
            T_dz,
            gradient,
            pde,
        )

    mean_loss = np.sum(loss) / len(data_loader)

    return mean_loss


def evaluate_tau2d(tau_model, latent, grid_loader, num_pts, batch_size, device):
    tau_model.eval()

    with torch.no_grad():
        T = torch.empty(num_pts, device=device)
        for i, X in enumerate(grid_loader, 0):

            batch_end = (
                (i + 1) * batch_size
                if (i + 1) * batch_size < num_pts
                else i * batch_size + X[0].shape[0]
            )

            # Latent
            z = latent.repeat(X[1].size()[0]).view(-1, latent.shape[0])

            xzs = torch.hstack((X[0], X[1].view(-1, 1), X[2].view(-1, 1), z))
            # xzs = torch.hstack((X[0], X[1].view(-1,1), X[2].view(-1,1), X[3].view(-1,1)))

            # Compute T
            T[i * batch_size : batch_end] = tau_model(xzs).view(-1)

    return T


def evaluate_velocity2d(tau_model, latent, grid_loader, num_pts, batch_size, device):

    tau_model.eval()

    # Prepare input
    # with torch.no_grad():
    V = torch.empty(num_pts, device=device)
    for i, X in enumerate(grid_loader):

        # Compute v
        batch_end = (
            (i + 1) * batch_size
            if (i + 1) * batch_size < num_pts
            else i * batch_size + X[0].shape[0]
        )

        # Latent
        z = latent.repeat(X[1].size()[0]).view(-1, latent.shape[0])

        xzs = torch.hstack((X[0], X[1].view(-1, 1), X[2].view(-1, 1), z))
        # xzs = torch.hstack((X[0], X[1].view(-1,1), X[2].view(-1,1), X[3].view(-1,1)))

        xzs.requires_grad = True

        # Compute T
        tau = tau_model(xzs).view(-1)

        # Gradients
        gradient = torch.autograd.grad(
            tau, xzs, torch.ones_like(tau), create_graph=True
        )[0]

        tau_dx = gradient[:, 0]
        tau_dz = gradient[:, 1]

        T_dx = tau_dx * X[3] + tau * X[4]
        T_dz = tau_dz * X[3] + tau * X[5]

        pde_lhs = T_dx**2 + T_dz**2

        V[i * batch_size : batch_end] = torch.sqrt(1 / pde_lhs)

    return V

def training_loop2d(
    input_wosrc,
    sx,
    sz,
    tau_model,
    v_sample,
    latent,
    optimizer,
    grid_size,
    num_pts,
    out_epochs,
    in_epochs=100,
    batch_size=200**3,
    vscaler=1.0,
    scheduler=None,
    fast_loader="n",
    device="cuda",
    wandb=None,
    args=None,
    num_vel=1,
    transfer=False,
):

    loss_history = []

    #     for vi in range(num_vel):

    #         v_sample_i, latent_i = v_sample[vi].to(device), latent[vi].to(device)

    #         print(f'Velocity #{vi+1}')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=30, verbose=True
    )

    for epoch in range(out_epochs):

        # Train step
        mean_loss = train2d(
            input_wosrc,
            sx,
            sz,
            tau_model,
            v_sample,
            latent,
            optimizer,
            grid_size,
            num_pts,
            batch_size,
            vscaler,
            scheduler,
            fast_loader,
            device,
            args,
            num_vel,
            transfer
        )
        if wandb is not None:
            wandb.log({"loss": mean_loss})
        loss_history.append(mean_loss)

        if epoch % (out_epochs//20) == 0:
            print(f"Epoch {epoch}, Loss {mean_loss:.7f}")
            # Save model
            torch.save({
                    'tau_model_state_dict': tau_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss':loss_history
            }, args['save_folder']+'/saved_model_epoch_'+str(epoch))

        if scheduler is not None:
            scheduler.step(mean_loss)

    return loss_history
