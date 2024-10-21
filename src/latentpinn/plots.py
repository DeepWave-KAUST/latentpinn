import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable

from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_square_images(array, height, width=4, name=None, save_dir=None, unit='km/s', **kwargs):
    
    fig, ax = plt.subplots(height, width, sharex=True, sharey=True, figsize=(12,4), layout="constrained")

    axs = ax.ravel()
    
    idx = 0
    for i in range(width*height):
        im = axs[idx].imshow(array[i,:,:], aspect='auto', **kwargs)
        axs[idx].set_xticklabels([])
        axs[idx].set_yticklabels([])
        idx+=1
        
    fig.supxlabel('Lateral Location (km)', fontsize=20)
    fig.supylabel('Depth (km)', fontsize=20)
        
    cbar = fig.colorbar(im, ax=ax[-1])
    cbar.set_label(unit, size=20)
    cbar.ax.tick_params(labelsize=20) 
    if name is not None:
        plt.savefig(save_dir+name)
        
    plt.show()
    
def plot_square_curves(trues, preds, height, width=4, name=None, save_dir=None, titles=None, **kwargs):
    
    fig, ax = plt.subplots(height, width, sharex=True, sharey=True, figsize=(12,4), layout="constrained")

    axs = ax.ravel()
    
    idx = 0
    for i in range(width*height):
        if i != (width*height-1):
            axs[idx].plot(trues[i], label='True', **kwargs)
            axs[idx].plot(preds[i], label='Prediction', **kwargs)
        else:
            axs[idx].plot(trues[0]-preds[0], label='Real', **kwargs)
            axs[idx].plot(trues[1]-preds[1], label='Imaginary', **kwargs)
        axs[idx].set_xticklabels([])
        axs[idx].set_yticklabels([])
        axs[idx].legend()
        axs[idx].set_title(titles[idx])
        idx+=1
        
    fig.supxlabel('Lateral Location (km)', fontsize=20)
    fig.supylabel('Amplitude (unitless)', fontsize=20)
        
    if name is not None:
        plt.savefig(save_dir+name)
        
    plt.show()

# def plot_square_image(array, height, width, name=None, save_dir=None, unit=None, wavefield=False):
    
#     fig, ax = plt.subplots(height, width, sharex=True, sharey=True, figsize=(int(1.5*height), int(1.5*height)), layout="constrained")

#     axs = ax.ravel()
#     idx = 0
#     for i in range(height):
#         for j in range(width):
#             if wavefield:
#                 im = axs[idx].imshow(array[idx,0,:,:], cmap='terrain', extent=[0,1,0,1], aspect='auto',vmin=2,vmax=5.5) # Wavefield
#             else:   
#                 im = axs[idx].imshow(array[i,j,:,:], cmap='terrain', extent=[0,1,0,1], aspect='auto',vmin=2,vmax=5.5) # Wavefield
#             axs[idx].set_xticklabels([])
#             axs[idx].set_yticklabels([])
#             idx+=1
        
#     # plt.subplots_adjust(wspace=0, hspace=0)
        
#     fig.supxlabel('Lateral Location (km)', fontsize=20)
#     fig.supylabel('Depth (km)', fontsize=20)
        
#     cbar = fig.colorbar(im, ax=ax[:, -1])#, ax=ax[-1, :], orientation="horizontal", shrink=0.6, location='bottom', pad=2)
#     cbar.set_label('km/s', size=20)
#     cbar.ax.tick_params(labelsize=20) 
#     if name is not None:
#         plt.savefig(save_dir+'/'+name)
        
#     # plt.tight_layout()
    
#     plt.show()
    

def plot_square_wavefield(array, height, width=4, name=None, save_dir=None, unit=None, **kwargs):
    
    fig, ax = plt.subplots(height, width, sharex=True, sharey=True, figsize=(int(1.5*height), int(1.5*height)), layout="constrained")

    axs = ax.ravel()
    
    idx = 0
    # for i in [6,7,9,16,21,28,29,43]:
    # for i in [6,7,9,29,43]:
    for i in range(width*height):
        im = axs[idx].imshow(array[i,:,:].T, aspect='auto', **kwargs)
        axs[idx].set_xticklabels([])
        axs[idx].set_yticklabels([])
        idx+=1
        
    fig.supxlabel('Lateral Location (km)', fontsize=20)
    fig.supylabel('Depth (km)', fontsize=20)
        
    cbar = fig.colorbar(im, ax=ax[:, -1])#, ax=ax[-1, :], orientation="horizontal", shrink=0.6, location='bottom', pad=2)
    cbar.set_label('km/s', size=20)
    cbar.ax.tick_params(labelsize=20) 
    if name is not None:
        plt.savefig(save_dir+'/'+name)
        
    # plt.tight_layout()
    
    plt.show()

    
def plot_square_contour(T_preds, T_datas, height, width=4, name=None, save_dir=None, unit=None, label1="Prediction",label2="True"):
    
    fig, ax = plt.subplots(height, width, sharex=True, sharey=True, figsize=(int(1.5*height), int(1.5*height)), layout="constrained")

    axs = ax.ravel()
    
    idx = 0
    for i in range(height):
    # for i in [6,7,9,16,21,28,29,43]:
        for j in range(width):
            c_p = axs[idx].contour(
                T_preds[i,j,:,:],
                5,
                colors="k",
                extent=[0,1,0,1]
            )
            c_t = axs[idx].contour(
                T_datas[i,j,:,:],
                5,
                colors="y",
                linestyles="dashed",
                extent=[0,1,0,1]                
            )
            h1, _ = c_p.legend_elements()
            h2, _ = c_t.legend_elements()
            # h3, _ = c_i.legend_elements()

            # axs[idx].legend([h1[0], h2[0]], ["Prediction", "True"])
            axs[idx].scatter(0.5, 0.5, s=200, marker="*", color="k")

            idx+=1
        
    fig.supxlabel('Lateral Location (km)')
    fig.supylabel('Depth (km)')
    
    plt.legend([h1[0], h2[0]], [label1, label2],loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True)
    
    if name is not None:
        plt.savefig(save_dir+'/'+name)
        
    # plt.tight_layout()
            
    plt.show()

def plot_contour(
    pred,
    true,
    init,
    idx,
    nx,
    nz,
    ns,
    sx,
    sz,
    x,
    z,
    fig_name=None,
    save_dir="./",
    title=None,
):
    plt.figure()
    c_p = plt.contour(
        pred.reshape(nz, nx, ns)[:, :, idx],
        5,
        colors="k",
        extent=(x[0], x[-1], z[0], z[-1]),
    )
    c_t = plt.contour(
        true.reshape(nz, nx, ns)[:, :, idx],
        5,
        colors="y",
        linestyles="dashed",
        extent=(x[0], x[-1], z[0], z[-1]),
    )
    c_i = plt.contour(
        init.reshape(nz, nx, ns)[:, :, idx],
        5,
        colors="b",
        linestyles="dashed",
        extent=(x[0], x[-1], z[0], z[-1]),
    )

    h1, _ = c_p.legend_elements()
    h2, _ = c_t.legend_elements()
    h3, _ = c_i.legend_elements()

    plt.legend([h1[0], h2[0], h3[0]], ["Prediction", "True", "Initial"])

    plt.scatter(sx[idx], sz[idx], s=200, marker="*", color="k")
    if title is not None:
        plt.title("Traveltime Contour")
    plt.xlabel("X (km)")
    plt.ylabel("Z (km)")
    plt.axis("tight")

    if fig_name is not None:
        plt.savefig(os.path.join(save_dir, fig_name), format="pdf", bbox_inches="tight")


def plot_section(
    data,
    fig_name,
    data_type="km/s",
    vmin=None,
    vmax=None,
    cmap="terrain",
    save_dir="./",
    aspect="equal",
    xmin=0,
    xmax=1,
    zmin=0,
    zmax=1,
    sx=None,
    sz=None,
    rx=None,
    rz=None,
    xtop=None,
    ztop=None,
):
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(
        data,
        extent=[xmin, xmax, zmax, zmin],
        cmap=cmap,
        aspect=aspect,
        vmin=vmin,
        vmax=vmax,
        interpolation="kaiser",
    )

    if sx is not None:
        plt.scatter(sx, sz, 25, "white", marker="*")

    if rx is not None:
        plt.scatter(rx, rz, 25, "black", marker="v")

    if xtop is not None:
        plt.scatter((xtop - xtop.min()), ztop, 2, "black", marker="o")

    plt.xlabel("Lateral Location (km)", fontsize=14)
    plt.xticks(fontsize=11)
    plt.ylabel("Depth (km)", fontsize=14)
    plt.yticks(fontsize=11)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax)

    cbar.set_label(data_type, size=10)

    if fig_name is not None:
        plt.savefig(os.path.join(save_dir, fig_name), format="pdf", bbox_inches="tight")


def plot_depth(
    data,
    fig_name,
    data_type="km/s",
    vmin=None,
    vmax=None,
    cmap="terrain",
    save_dir="./",
    aspect="equal",
    xmin=0,
    xmax=1,
    zmin=0,
    zmax=1,
    sx=None,
    sz=None,
    rx=None,
    rz=None,
):
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(
        data,
        extent=[xmin, xmax, zmax, zmin],
        cmap=cmap,
        aspect=aspect,
        vmin=vmin,
        vmax=vmax,
        interpolation="kaiser",
    )

    if sx is not None:
        plt.scatter(sx, sz, 25, "white", marker="*")

    if rx is not None:
        plt.scatter(rx, rz, 25, "black", marker="v")

    plt.xlabel("Lateral Location (km)", fontsize=14)
    plt.xticks(fontsize=11)
    plt.ylabel("Lateral Location (km)", fontsize=14)
    plt.yticks(fontsize=11)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax)

    cbar.set_label(data_type, size=10)

    if fig_name is not None:
        plt.savefig(os.path.join(save_dir, fig_name), format="pdf", bbox_inches="tight")


def plot_trace(init, true, pred, trace_id, x, z, fig_name=None, save_dir="./"):
    plt.figure(figsize=(3, 5))
    plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.labelbottom"] = False
    plt.rcParams["xtick.top"] = plt.rcParams["xtick.labeltop"] = True

    ax = plt.gca()

    plt.plot(init[:, trace_id], z, "b:")
    plt.plot(true[:, trace_id], z, "k")
    plt.plot(pred[:, trace_id], z, "r--")

    ax.set_title("Velocity (km/s)", fontsize=14)

    plt.xticks(fontsize=11)
    plt.ylabel("Depth (km)", fontsize=14)
    plt.xlabel("Lateral Location " + str(x[trace_id].round(3)) + " (km)", fontsize=14)
    plt.yticks(fontsize=11)
    plt.gca().invert_yaxis()
    plt.legend(["Initial", "True", "Inverted"], fontsize=11)
    plt.grid()

    if fig_name is not None:
        plt.savefig(os.path.join(save_dir, fig_name), format="pdf", bbox_inches="tight")


def plot_horizontal(
    trace1,
    trace2,
    x,
    title,
    ylabel,
    fig_name,
    label1,
    label2,
    save_dir="./",
    id_rec_x=None,
    id_rec_z=None,
):
    plt.figure(figsize=(5, 3))

    ax = plt.gca()

    plt.plot(x, trace1, "b")
    plt.plot(x, trace2, "r:")

    if id_rec_x is not None:
        plt.scatter(x[id_rec_x], trace1[id_rec_x], 25, "y", marker="v")

    ax.set_title(title, fontsize=14)

    plt.xticks(fontsize=11)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel("Lateral Location (km)", fontsize=14)
    plt.yticks(fontsize=11)
    plt.gca().invert_yaxis()
    plt.legend([label1, label2], fontsize=11)
    plt.grid()

    if fig_name != None:
        plt.savefig(os.path.join(save_dir, fig_name), format="pdf", bbox_inches="tight")
