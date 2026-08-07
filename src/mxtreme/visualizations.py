'''
Basic, useful visualizations.
- MEA
- raster plots

'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.animation as animation
import matplotlib.image
from glob import glob

from mxtreme import constants

def MEA(ax, channelmap, stim_elecs, title="MEA Channel Layout"):

    mid_point = (constants.CHIP_WIDTH / 2) * constants.ELEC_SIZE # This is the x mid point

    chip_ht_um = constants.CHIP_HEIGHT * constants.ELEC_SIZE

    # channel_map[:,3] are the x locations, channel_map[:,4] are the y locations
    num_chans_left = np.sum(channelmap[:, 3] < mid_point)
    num_chans_right = np.sum(channelmap[:, 3] > mid_point)
    print(f'Left: {num_chans_left}, Right: {num_chans_right}')

    # Plot the channel map and stimulation electrodes
    ax.scatter(channelmap[:, 3], chip_ht_um-channelmap[:, 4], marker=".", facecolors='none', edgecolors='k')
    ax.plot([mid_point, mid_point], [0, chip_ht_um], 'b--', label='Midpoint')
    ax.set_title(title)
    ax.set_xlabel("x direction (µm)")
    ax.set_ylabel("y direction (µm)")

    # Plot stimulation electrodes
    stim_coords = channelmap[np.isin(channelmap[:, 2], stim_elecs), 3:5] # channel_map[:,2] is the electrode ID 

    # Coordinates for lightning bolt emojis
    stim_coords = channelmap[np.isin(channelmap[:, 2], stim_elecs), 3:5]

    # Plot lightning bolt emojis
    for x, y in stim_coords:
        txt = ax.text(x, chip_ht_um-y, "⚡", fontsize=16, ha='center', va='center', color='gold') #, fontweight='bold'
        # Add black outline
        txt.set_path_effects([
            path_effects.Stroke(linewidth=1, foreground="black"),
            path_effects.Normal()
        ])


def plot_asdr(spike_bin, title=None, zoom=None, save_path=None, savefilename=None, annotate_bursts=False, bursts=None, burst_threshold=None):
    # ASDR = array-wide spike detection rate
    # Get average activity across channels
    asdr = spike_bin.mean(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(12,3))
    
    ax.plot(asdr)
    ax.set_ylabel("ASDR", fontsize=10)
    ax.set_xlabel("Time (bins)", fontsize=10)
    ax.set_title(title, fontsize=15)

    if annotate_bursts and bursts is not None:

        # when bursts was a list of burst objects instead of a dataframe 
        # heights = [x.peak+0.02 for x in bursts] # +0.02 for annotation offset
        # t = [x.t_peak_bin for x in bursts] 

        heights = bursts['peak_bin_amp']+0.02
        t = bursts['t_peak_bin']

        # Annotate peaks with upside-down red triangles
        ax.scatter(t, heights, marker='v', color='red', s=10)

    if burst_threshold is not None:
        ax.hlines(burst_threshold, 0, len(asdr), colors='g', linestyles='dashed')

    if zoom:
        ax.set_xlim(left=zoom[0], right=zoom[1])

    if zoom:
        ax.set_xlim(zoom)

    if save_path is not None:
        plt.savefig(save_path/savefilename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def raster_plot(spike_data, zoom, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(12,3))

    spike_dat_slice = spike_data[(spike_data['frameno']<zoom[1])&(spike_data['frameno']>zoom[0])]

    spike_mat = np.zeros(shape=(zoom[1]-zoom[0], np.max(spike_data['channel'])),dtype=int)
    
    for i in spike_dat_slice:
        frame = i['frameno']-zoom[0]
        spike_mat[frame, i['channel']]=1
    
    # imshow = ax.imshow(spike_bin[:,150000:150050], cmap='binary', aspect='auto') # zoomed
    imshow = ax.imshow(spike_mat, cmap='binary', aspect='auto')
    cbar = plt.colorbar(imshow, ax=ax)
    cbar.set_label(label='Activity',size=15)

    ax.set_ylabel("Channels", fontsize=10)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_title(title, fontsize=15)
    
    if zoom:
        ax.set_xlim(zoom)

    plt.show()

def plot_asdr_raster(spike_bin, title=None, zoom=None, save_path=None, savefilename=None, annotate_bursts=False, bursts=None, show=True, return_ax=False):

    asdr = spike_bin.mean(axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,7))

    # ASDRs
    ax1.plot(asdr)
    ax1.set_ylabel("ASDR", fontsize=10)
    ax1.set_xlabel("Time (bins)", fontsize=10)
    ax1.set_title(title, fontsize=15)

    if annotate_bursts and bursts is not None:
        heights = [x.peak+0.02 for x in bursts] # +0.02 for annotation offset
        t = [x.t_peak_bin for x in bursts] 

        # Annotate peaks with upside-down red triangles
        ax1.scatter(t, heights, marker='v', color='red', s=10)

    if zoom:
        ax1.set_xlim(left=zoom[0], right=zoom[1])

    # Raster
    ax2.imshow(spike_bin, cmap='binary', aspect='auto')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    ax2.set_ylabel("Channels", fontsize=10)
    ax2.set_xlabel("Time (bins)", fontsize=10)

    if zoom:
        ax2.set_xlim(left=zoom[0], right=zoom[1])

    if save_path is not None:
        plt.savefig(save_path+savefilename, dpi=300, bbox_inches='tight')
        plt.close()
    
    if show:
        plt.show()

    if return_ax:
        return ax1, ax2

def histogram(x, xlabel=None, ylabel='Frequency', title=None, save_path=None, savefilename=None):
        
    plt.figure(figsize=(8, 5))
    
    plt.hist(x, color='steelblue', edgecolor='black', alpha=0.75)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path+savefilename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def make_gif(image_files, savepath, savename='output.png', labels=None, label_loc=(200,200), label_color='black', label_fontsize=10, title='', fps=1):

    images = [matplotlib.image.imread(im) for im in image_files]

    fig, ax = plt.subplots()
    im = ax.imshow(images[0])
    ax.axis("off")
    ax.set_title(title)

    annotation = ax.text(label_loc[0], label_loc[1], "", color=label_color, fontsize=label_fontsize, weight="bold")

    def update(frame):
        im.set_array(images[frame])
        if labels is not None:
            annotation.set_text(labels[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(images), interval=200, blit=True)

    fig.tight_layout()
    ani.save(f"{savepath}{savename}.gif", writer="pillow", fps=fps, dpi=300)
    plt.close(fig)




