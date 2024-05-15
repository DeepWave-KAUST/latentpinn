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

    def _make_layer(self, block, layers,):
        layers_net = []
        for layer in layers:
            layers_net.append(block(self.in_planes,layer))
            self.in_planes = layer
        return nn.Sequential(*layers_net)

    def forward(self,x):
        out = self.layer1(x)
        out = self.linear(out)
        return out[:,0:1], out[:,1:2]

def pinn_train(model,x_train_all,y_train_all,sx_train_all,
               u0_real_train_all,u0_imag_train_all,
               lb_all,ub_all,m_all,m0_all,omega,
               niter,embedding,f,
               x_bc, z_bc, sx_bc, dur_bc, dui_bc, bc_type,
               num_vel=1,latent=None,alpha=None,radii=1,single_id=0):
    start_time = time.time()
    m_all = torch.FloatTensor(m_all).cuda()
    m0_all = torch.FloatTensor(m0_all).cuda()
    optimizer = Adam(model.parameters(),lr=1e-3,weight_decay=5e-4)
    #optimizer = Adam(model.parameters(),lr=1e-3)

    model.train()

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
            
        # x_input = torch.cat((x_input, latent[v_id,:].repeat(40000).view(-1,96)),1)
        x_input = torch.cat((x_input, latent[v_id,:].repeat(40000).view(-1,96).requires_grad_(False)),1)
        
        dU_real_pred, dU_imag_pred = model(x_input)

        du_real_xx = laplace(dU_real_pred,x)
        du_imag_xx = laplace(dU_imag_pred,x)
        du_real_yy = laplace(dU_real_pred,y)
        du_imag_yy = laplace(dU_imag_pred,y)

        f_real_pred = omega*omega*m*dU_real_pred + du_real_xx + du_real_yy + omega*omega*(m-m0)*u0_real_train_all[v_id]
        f_imag_pred = omega*omega*m*dU_imag_pred + du_imag_xx + du_imag_yy + omega*omega*(m-m0)*u0_imag_train_all[v_id]

        if bc_type=='top-bottom':
            X_bc = embedding(2.0 *(torch.cat((x_bc[:,:,[0,-1]].reshape(-1,1), z_bc[:,:,[0,-1]].reshape(-1,1), sx_bc[:,:,[0,-1]].reshape(-1,1)),1)-0)/(2.5) - 1.0)
            X_bc = torch.cat((X_bc, latent[v_id,:].repeat(1818).view(-1,96).requires_grad_(False)),1)
            du_real_bc, du_imag_bc = model(X_bc)
            boundary = ((torch.sum(torch.square(du_real_bc-dur_bc[v_id,:,:,[0,-1]].reshape(-1,1))) + torch.sum(torch.square(du_imag_bc-dui_bc[v_id,:,:,[0,-1]].reshape(-1,1))))/(x.shape [0] * 2))
        elif bc_type=='top':
            X_bc = embedding(2.0 *(torch.cat((x_bc[:,:,0].reshape(-1,1), z_bc[:,:,0].reshape(-1,1), sx_bc[:,:,0].reshape(-1,1)),1)-0)/(2.5) - 1.0)
            X_bc = torch.cat((X_bc, latent[v_id,:].repeat(909).view(-1,96).requires_grad_(False)),1)
            du_real_bc, du_imag_bc = model(X_bc)
            boundary = ((torch.sum(torch.square(du_real_bc-dur_bc[v_id,:,:,0].reshape(-1,1))) + torch.sum(torch.square(du_imag_bc-dui_bc[v_id,:,:,0].reshape(-1,1))))/(x.shape [0] * 2))
        elif bc_type=='all':
            X_bc = embedding(2.0 *(torch.cat((torch.cat((x_bc[:,:,[0,-1]].reshape(-1,1), x_bc[:,[0,-1],:].reshape(-1,1)), 0), torch.cat((z_bc[:,:,[0,-1]].reshape(-1,1), z_bc[:,[0,-1],:].reshape(-1,1)), 0), torch.cat((sx_bc[:,:,[0,-1]].reshape(-1,1), sx_bc[:,[0,-1],:].reshape(-1,1)),0)),1)-0)/(2.5) - 1.0)
            X_bc = torch.cat((X_bc, latent[v_id,:].repeat(3636).view(-1,96).requires_grad_(False)),1)
            du_real_bc, du_imag_bc = model(X_bc)
            boundary = ((torch.sum(torch.square(du_real_bc-torch.cat((dur_bc[v_id,:,:,[0,-1]].reshape(-1,1), dur_bc[v_id,:,[0,-1],:].reshape(-1,1)), 0))) + torch.sum(torch.square(du_imag_bc-torch.cat((dui_bc[v_id,:,:,[0,-1]].reshape(-1,1), dui_bc[v_id,:,[0,-1],:].reshape(-1,1)), 0))))/(x.shape [0] * 2))
        else:
            boundary = 0.
                        
        loss_value = ((torch.sum(torch.square(f_real_pred)) + torch.sum(torch.square(f_imag_pred)))/(x.shape [0] * 2)
                      + alpha*boundary)
        loss_value.backward()
        optimizer.step()
        misfit.append(loss_value.item())

        if it%(100*num_vel) == 0:
            elapsed = time.time() - start_time
            print('It: %d, Loss: %.3e, Time: %.2f' %(it,\
                loss_value,elapsed))
            start_time = time.time()
                        #save model
            state = {
                'net':model.module
            }
            torch.save(state,sw.log_dir+'/model_'+str(it)+'.tz')
            
    del loss_value

def pinn_predict(model, x_star, z_star, sx_star,lb,ub,embedding,num_vel=1,latent=None):
    model.eval()
    x_star = x_star.cuda()
    z_star = z_star.cuda()
    sx_star= sx_star.cuda()
    x_input = torch.cat((x_star,z_star,sx_star),1)
    x_input = 2.0 *(x_input-lb)/(ub-lb) - 1.0
    x_input = embedding(x_input)
    x_input = torch.cat((x_input, latent.repeat(9*101**2).view(-1,96)),1)
    with torch.no_grad():
        dU_real_star, dU_imag_star = model(x_input)
    return dU_real_star, dU_imag_star