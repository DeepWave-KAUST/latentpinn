import torch
import sys
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import random, os
import torch.nn.functional as F
import math

from torch.autograd import grad
from mpl_toolkits.axes_grid1 import make_axes_locatable

def outloss(epoch,num_epochs,batch_idx,batch_len,loss,elapsed):
    '''
    print the loss
    '''
    sys.stdout.write('\r')
    sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\tElapsed: %.4f\tLoss: %.4f'
                                        %(epoch, num_epochs, batch_idx+1, batch_len, elapsed, loss.item()))
    sys.stdout.flush()

def plot_results(du_real_pred,du_imag_pred,du_real_star, du_imag_star, epoch,source_number,freq):
    # Error
    error_du_real = np.linalg.norm(du_real_star-du_real_pred,2)/np.linalg.norm(du_real_star,2)
    error_du_imag = np.linalg.norm(du_imag_star-du_imag_pred,2)/np.linalg.norm(du_imag_star,2)

    print('Error u_real: %e, Error u_imag: %e' % (error_du_real,error_du_imag))
    scipy.io.savemat('du_real_pred_atan-{}.mat'.format(cf.fre),{'du_real_pred':du_real_pred})
    scipy.io.savemat('du_imag_pred_atan-{}.mat'.format(cf.fre),{'du_imag_pred':du_imag_pred})

    scipy.io.savemat('du_real_star-{}.mat'.format(cf.fre),{'du_real_star':du_real_star})
    scipy.io.savemat('du_imag_star-{}.mat'.format(cf.fre),{'du_imag_star':du_imag_star})

    ## plot the real parts of the scattered wavefield for i th source
    #source_number = 8 ## 1-9
    a = (source_number-1)*nx*nz
    b = (source_number)*nx*nz
    du_real_star_is = du_imag_star[a:b:1]
    du_real_pred_is = du_imag_pred[a:b:1]
    du_real_star_is2D = np.reshape(np.array(du_real_star_is), (nx, nz))
    du_real_pred_is2D = np.reshape(np.array(du_real_pred_is), (nx, nz))
    du_real_dif2D = du_real_star_is2D - du_real_pred_is2D

    error_du_imag = np.linalg.norm(du_real_star_is-du_real_pred_is,2)/np.linalg.norm(du_real_star_is,2)

    print('Error for shot 4 u_imag: %e' % (error_du_imag))

    plt.figure(figsize=(20,60))
    plt.subplot(3, 1, 1)
    ax = plt.gca()
    im = ax.imshow(du_real_star_is2D.T, vmin=cf.vmin,vmax=cf.vmax,extent=[0, cf.axisx, cf.axisz,0],aspect=1, cmap="jet")
    #im = ax.imshow(du_real_star_is2D.T, extent=[0, 2.5, 2.5,0],aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.ylabel('Depth (km)', fontsize=14)
    plt.title('Numerical solution')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    plt.subplot(3, 1, 2)
    ax = plt.gca()
    im = ax.imshow(du_real_pred_is2D.T, vmin=cf.vmin,vmax=cf.vmax,extent=[0, cf.axisx, cf.axisz,0],aspect=1, cmap="jet")
    #im = ax.imshow(du_real_pred_is2D.T,extent=[0, 2.5, 2.5,0],aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.title('PINN solution')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    plt.subplot(3, 1, 3)
    ax = plt.gca()
    im = ax.imshow(du_real_dif2D.T, vmin=cf.vmin,vmax=cf.vmax,extent=[0, cf.axisx, cf.axisz, 0], aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.title('Difference')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Amplitude')
    #plt.show()
    plt.savefig(cf.model_name+'Epoch'+str(epoch)+'-'+str(freq)+'result-imag.png')

    du_real_star_is = du_real_star[a:b:1]
    du_real_pred_is = du_real_pred[a:b:1]
    du_real_star_is2D = np.reshape(np.array(du_real_star_is), (nx, nz))
    du_real_pred_is2D = np.reshape(np.array(du_real_pred_is), (nx, nz))
    du_real_dif2D = du_real_star_is2D - du_real_pred_is2D

    error_du_imag = np.linalg.norm(du_real_star_is-du_real_pred_is,2)/np.linalg.norm(du_real_star_is,2)

    print('Error for shot 4 u_real: %e' % (error_du_imag))

    plt.figure(figsize=(20,60))
    plt.subplot(3, 1, 1)
    ax = plt.gca()
    im = ax.imshow(du_real_star_is2D.T, vmin=cf.vmin,vmax=cf.vmax,extent=[0, cf.axisx, cf.axisz,0],aspect=1, cmap="jet")
    #im = ax.imshow(du_real_star_is2D.T, extent=[0, 2.5, 2.5,0],aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.ylabel('Depth (km)', fontsize=14)
    plt.title('Numerical solution')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    plt.subplot(3, 1, 2)
    ax = plt.gca()
    im = ax.imshow(du_real_pred_is2D.T, vmin=cf.vmin,vmax=cf.vmax,extent=[0, cf.axisx, cf.axisz,0],aspect=1, cmap="jet")
    #im = ax.imshow(du_real_pred_is2D.T,extent=[0, 2.5, 2.5,0],aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.title('PINN solution')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    plt.subplot(3, 1, 3)
    ax = plt.gca()
    im = ax.imshow(du_real_dif2D.T, vmin=cf.vmin,vmax=cf.vmax,extent=[0, cf.axisx, cf.axisz, 0], aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.title('Difference')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Amplitude')
    #plt.show()
    plt.savefig(cf.model_name+'Epoch'+str(epoch)+'-'+str(freq)+'result-real.png')

# Positional encoding
class Embedder1:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder1(multires,embed_dim, i=0):

    if i == -1:
        return torch.nn.identity, embed_dim

    embed_kwargs = {
        'include_input': True,
        'input_dims': embed_dim,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder1(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

def _l2_normalize(d):
    #d_reshaped = d.view(d.shape[0],-1,*(1 for _ in range(d.dim() - 2)))
    #d /= torch.norm(d_reshaped,dim=1,keepdim=True) + 1e-8
    d /= torch.norm(d,keepdim=True) + 1e-8
    return d.cuda()

def plot(d):
    fig = plt.figure()
    plt.imshow(-1.0*np.sum(d[:,:,:],axis=0).T,extent=(0,12.5,4.0,0.0))
    plt.colorbar()
    return fig


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=200**3, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def torch_to_numpy(x, nx=None, nz=None, ns=None):

    if (nx is not None) & (nz is not None) & (ns is not None):
        return x.detach().cpu().numpy().reshape(nz, nx, ns)
    else:
        return x.detach().cpu().numpy()


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

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

def normalizer(x, lb, ub):
    return 2.0 * (x - lb) / (ub - lb) - 1.0

class helmholtz_loss_pde(torch.nn.Module):
    def __init__(self,  dx, dy, lb, ub, regularized=1.0,causal=False,pde_loss_penelty=0.0, v_background=1.5, device='cuda:0'):
        super(helmholtz_loss_pde, self).__init__()
        self.dx = dx
        self.dy = dy
        self.ub = ub
        self.lb = lb
        self.device = device
        self.regularized = regularized
        self.pde_loss_penelty = pde_loss_penelty
        self.causal = causal
        self.v_background = v_background / 2.0
    
    def regloss(self, x, y, sx, f, du_pred_out):
        factor_d = F.relu((self.v_background * 3.14/f)**2-(sx-x)**2-(y-0.025)**2)*10e7*f
        loss_reg = torch.sum(factor_d*torch.pow(du_pred_out[:,0:1],2)) +  torch.sum(factor_d*torch.pow(du_pred_out[:,1:2],2))
        return loss_reg / (x.shape[0] * 2)
    
    def _pde_res_penelty(self, x, y, pde_loss):
        return torch.sum(torch.pow(gradient(pde_loss, x), 2)) + torch.sum(torch.pow(gradient(pde_loss, y) , 2)) 
    
    def _laplace(self, u_left, u_right, u , u_top, u_down):
        return (u_left + u_right - 2.0 * u) / (self.dx**2) + (u_top + u_down - 2.0 * u) / (self.dy**2)
    
    def _query(self, x, y, sx, net, embedding_fn):
        x_input = torch.cat((x,y,sx),1)
        x_input = 2.0 * (x_input - self.lb) / (self.ub - self.lb) - 1.0
        return net(embedding_fn(x_input))
    
    def causal_loss(self, f_real_pred, f_imag_pred, x, y, sx, interval=0.025, max_r_step=142, epsilon=1e-7):
        loss_temp = 0.0
        lr = ((torch.pow(f_real_pred, 2)) + (torch.pow(f_imag_pred, 2)))
        distance_s_r = (sx-x)**2 + (y- 0.025) **2
        loss = torch.sum(lr[ distance_s_r <= interval**2])
        for iz in range(1,max_r_step):
            loss += math.exp(-1.0*epsilon*loss_temp) * torch.sum(lr[((distance_s_r<=((iz+1)*interval)**2) & (distance_s_r>(iz*interval)**2))])
            loss_temp  +=  torch.sum(lr[((distance_s_r<=((iz+1)*interval)**2) & (distance_s_r>(iz*interval)**2))]).item()
        loss = 1.0/max_r_step *loss
        return loss, loss_temp
        
    def forward(self, x, y, sx, omega, m_train, m0_train, u0_real_train, u0_imag_train, du_pred_out, net, embedding_fn, derivate_type='ad', epsilon=1e-7):
        if derivate_type == 'fd':
            with torch.no_grad():
                du_pred_out_left = self._query(x - self.dx, y, sx, net, embedding_fn)
                du_pred_out_right = self._query(x + self.dx, y, sx, net, embedding_fn)
                du_pred_out_top = self._query(x, y-self.dy, sx, net, embedding_fn)
                du_pred_out_down = self._query(x, y+self.dy, sx, net, embedding_fn)
            du_laplace_real = self._laplace(du_pred_out_left[:,0:1], du_pred_out_right[:,0:1], du_pred_out[:,0:1], du_pred_out_top[:,0:1], du_pred_out_down[:,0:1])
            du_laplace_imag = self._laplace(du_pred_out_left[:,1:2], du_pred_out_right[:,1:2], du_pred_out[:,1:2], du_pred_out_top[:,1:2], du_pred_out_down[:,1:2])
        elif derivate_type == 'ad':
            du_real_xx = laplace(du_pred_out[:,0:1], x)
            du_imag_xx = laplace(du_pred_out[:,1:2], x)
            du_real_yy = laplace(du_pred_out[:,0:1], y)
            du_imag_yy = laplace(du_pred_out[:,1:2], y)
            du_laplace_real = du_real_xx + du_real_yy
            du_laplace_imag = du_imag_xx + du_imag_yy 
        f_real_pred = omega**2 * m_train*du_pred_out[:,0:1] + du_laplace_real + omega**2 * (m_train-m0_train) * u0_real_train
        f_imag_pred = omega**2 * m_train*du_pred_out[:,1:2] + du_laplace_imag + omega**2 * (m_train-m0_train) * u0_imag_train
        if self.pde_loss_penelty > 0:
            pde_loss_pen = self._pde_res_penelty(x, y, f_real_pred) + self._pde_res_penelty(x, y, f_imag_pred)
        else:
            pde_loss_pen = 0.0
        if self.regularized != 'None':
            loss_reg = self.regloss(x, y, sx, omega, du_pred_out)
            loss_pde =  (torch.sum(torch.pow(f_real_pred,2)) + torch.sum(torch.pow(f_imag_pred,2))) / (x.shape[0] * 2) 
            return loss_pde + self.regularized * loss_reg + self.pde_loss_penelty * pde_loss_pen , loss_pde, loss_reg, f_real_pred, f_imag_pred
        elif self.causal:
            loss_pde, loss_pde_wo_weight = self.causal_loss(f_real_pred, f_imag_pred, x, y, sx, interval=0.025, max_r_step=142, epsilon=epsilon) 
            return loss_pde, loss_pde_wo_weight/(x.shape[0]*2), f_real_pred, f_imag_pred
        else:
            return (torch.sum(torch.pow(f_real_pred,2)) + torch.sum(torch.pow(f_imag_pred,2))) / (x.shape[0] * 2) + self.pde_loss_penelty * pde_loss_pen, f_real_pred, f_imag_pred

def error_l2(du_real, du_imag, du_real_ref, du_imag_ref):
    error_du_real = np.linalg.norm(du_real-du_real_ref,2)/np.linalg.norm(du_real_ref,2)
    error_du_imag = np.linalg.norm(du_imag-du_imag_ref,2)/np.linalg.norm(du_imag_ref,2)
    return error_du_real, error_du_imag