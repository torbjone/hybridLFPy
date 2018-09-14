#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.style
matplotlib.use("AGG")
matplotlib.style.use('classic')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os

import plotting_helpers as phlp
from hybridLFPy import CachedNetwork
import analysis_params

######################################
### OUTSIDE SCOPE DEFINITIONS      ###
######################################

from params_evoked_with_EEG import multicompartment_params


######################################
### IMPORT PANELS                  ###
######################################

from plot_methods import network_sketch, plotConnectivity, plot_population, plot_signal_sum



######################################
### FIGURE                         ###
######################################


def fig_intro(params, ana_params, fraction=0.05, rasterized=False):
    '''set up plot for introduction'''
    plt.close("all")
    ana_params.set_PLOS_2column_fig_style(ratio=0.5)
    
    #load spike as database
    networkSim = CachedNetwork(**params.networkSimParams)
    if analysis_params.bw:
        networkSim.colors = phlp.get_colors(len(networkSim.X))

    #set up figure and subplots
    fig = plt.figure(figsize=[6, 2.7])
    gs = gridspec.GridSpec(3, 5)

    fig.subplots_adjust(left=0.03, right=0.98, wspace=0.7, hspace=0.)

    #population
    ax2 = fig.add_subplot(gs[:, 2], frameon=False)
    ax2.xaxis.set_ticks([])
    ax2.yaxis.set_ticks([])
    plot_population(ax2, params, isometricangle=np.pi/24, plot_somas=False,
                    plot_morphos=True, num_unitsE=1, num_unitsI=1,
                    clip_dendrites=True, main_pops=True, title='',
                    rasterized=rasterized)
    ax2.set_title('multicompartment\nneurons', va='bottom', fontweight='normal')
    # phlp.annotate_subplot(ax2, ncols=4, nrows=1, letter='C', linear_offset=0.065)

    #population
    ax_4s = fig.add_subplot(gs[0, 4], frameon=False)
    ax_4s.xaxis.set_ticks([])
    ax_4s.yaxis.set_ticks([])
    plot_foursphere_to_ax(ax_4s, params)

    ax_4s.set_title('four-sphere\nhead model', va='bottom', fontweight='normal')
    # phlp.annotate_subplot(ax_4s, ncols=4, nrows=1, letter='E', linear_offset=0.065)


    #network diagram
    ax0_1 = fig.add_subplot(gs[:, 0], frameon=False)
    ax0_1.set_title('point-neuron network', va='bottom')

    network_sketch(ax0_1, yscaling=1.6)
    ax0_1.xaxis.set_ticks([])
    ax0_1.yaxis.set_ticks([])
    # phlp.annotate_subplot(ax0_1, ncols=4, nrows=1, letter='A', linear_offset=0.065)
   
    #network raster
    ax1 = fig.add_subplot(gs[:, 1], frameon=True)
    ax1.set_ylabel('')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax1.set_title('spiking activity', va='bottom')
    phlp.remove_axis_junk(ax1)
    # phlp.annotate_subplot(ax1, ncols=4, nrows=1, letter='B', linear_offset=0.065)


    # Plot EEG at top of head
    ax_top_EEG = fig.add_subplot(gs[2, 4], frameon=True, title="EEG at top\nof head", ylim=[-0.8, .3])
    ax_top_EEG.xaxis.set_major_locator(plt.MaxNLocator(4))

    ax_top_EEG.set_ylabel("$\mu$V", labelpad=-4)
    summed_top_EEG = np.load(os.path.join(params.savefolder, "summed_EEG.npy"))

    ax_top_EEG.plot(summed_top_EEG - np.average(summed_top_EEG), lw=0.7, c='b')
    phlp.remove_axis_junk(ax_top_EEG)

    #LFP traces in all channels
    ax3 = fig.add_subplot(gs[:, 3], frameon=True)
    phlp.remove_axis_junk(ax3)
    ax3.set_title('LFP', va='bottom')
    ax3.xaxis.set_major_locator(plt.MaxNLocator(4))


    # phlp.annotate_subplot(ax3, ncols=4, nrows=1, letter='D', linear_offset=0.065)

    #draw some arrows:
    ax = plt.gca()
    ax.annotate("", xy=(0.27, 0.5), xytext=(.24, 0.5),
                xycoords="figure fraction",
            arrowprops=dict(facecolor='black', arrowstyle='simple'),
            )
    ax.annotate("", xy=(0.52, 0.5), xytext=(.49, 0.5),
                xycoords="figure fraction",
            arrowprops=dict(facecolor='black', arrowstyle='simple'),
            )
    ax.annotate("", xy=(0.78, 0.5), xytext=(.75, 0.5),
                xycoords="figure fraction",
            arrowprops=dict(facecolor='black', arrowstyle='simple'),
            )

    for t_idx in np.arange(40, 2200, 2):

        T = [t_idx, t_idx + 75]

        ax_top_EEG.set_xlim(T)

        x, y = networkSim.get_xy(T, fraction=fraction)
        # networkSim.plot_raster(ax1, T, x, y, markersize=0.1, alpha=1.,legend=False, pop_names=True)
        networkSim.plot_raster(ax1, T, x, y, markersize=0.2, marker='_',
                               alpha=1.,legend=False, pop_names=True, rasterized=rasterized)

        # a = ax1.axis()
        # try:
        #     ax1.vlines(x['TC'][0], a[2], a[3], 'k', lw=0.25)
        # except IndexError:
        #     pass
        ax3.clear()
        #LFP traces in all channels
        ax3 = fig.add_subplot(gs[:, 3], frameon=True)
        phlp.remove_axis_junk(ax3)
        ax3.set_title('LFP', va='bottom')
        ax3.xaxis.set_major_locator(plt.MaxNLocator(4))
        # phlp.annotate_subplot(ax3, ncols=4, nrows=1, letter='D', linear_offset=0.065)



        plot_signal_sum(ax3, params, fname=os.path.join(params.savefolder, 'LFPsum.h5'),
                    unit='mV', vlimround=0.8,
                    T=T, ylim=[ax2.axis()[2], ax2.axis()[3]],
                    rasterized=False)
        # ax3.set_xticks(ax_top_EEG.get_xticks())
        # ax3.set_xlim(T)
        # a = ax3.axis()
        # try:
        #     ax3.vlines(x['TC'][0], a[2], a[3], 'k', lw=0.25)
        # except IndexError:
        #     pass


        fig.savefig(os.path.join("EEG_anim2", 'hybrid_with_EEG_{:04d}.png'.format(t_idx)),
                        dpi=300,
                    #    bbox_inches='tight',
                    #pad_inches=0
                    )


    

def plot_foursphere_to_ax(ax, params):

    # four_sphere properties
    radii = [79000., 80000., 85000., 90000.]
    radii_name = ["Cortex", "CSF", "Skull", "Scalp"]

    xlim = [-7000, 7000]
    ylim = [radii[0]-6000, radii[-1] + 100]

    max_angle = np.abs(np.rad2deg(np.arcsin(xlim[0] / ylim[0])))


    pop_bottom = radii[0] - 1500
    pop_top = radii[0]

    isometricangle = np.pi/24
    r = 564

    theta0 = np.linspace(0, np.pi, 20)
    theta1 = np.linspace(np.pi, 2*np.pi, 20)

    outline_params = dict(color='red', lw=0.5, zorder=50)
    ax.plot(r*np.cos(theta0), r*np.sin(theta0)*np.sin(isometricangle)+pop_bottom,
            **outline_params)
    ax.plot(r*np.cos(theta1), r*np.sin(theta1)*np.sin(isometricangle)+pop_bottom,
            **outline_params)
    ax.plot(r*np.cos(theta0), r*np.sin(theta0)*np.sin(isometricangle)+pop_top,
            **outline_params)
    ax.plot(r*np.cos(theta1), r*np.sin(theta1)*np.sin(isometricangle)+pop_top,
            **outline_params)

    ax.plot([-r, -r], [pop_bottom, pop_top], **outline_params)
    ax.plot([r, r], [pop_bottom, pop_top], **outline_params)

    angle = np.linspace(-max_angle, max_angle, 100)
    for b_idx in range(len(radii)):
        x_ = radii[b_idx] * np.sin(np.deg2rad(angle))
        z_ = radii[b_idx] * np.cos(np.deg2rad(angle))
        l_curved, = ax.plot(x_, z_, c="k", lw=1)
        z_shift = -500 if b_idx == 0 else 0
        ax.text(x_[0], z_[0] + z_shift, radii_name[b_idx], fontsize=6,
                va="top", ha="right", color="k", rotation=5)

    ax.plot([2500, 7500], [radii[0] - 3000, radii[0] - 3000], c='gray', lw=3)
    ax.text(2500, radii[0] - 2500, "5 mm", color="gray")

    ax.plot([-300, 300], [radii[-1], radii[-1]], 'b', lw=2)
    ax.text(0, radii[-1] + 500, "EEG electrode", ha="center")



def plot_EEG_sphere_to_ax(ax, params):
    pass

if __name__ == '__main__':
    plt.close('all')

    params = multicompartment_params()
    ana_params = analysis_params.params()

    
    savefolders = ['evoked_cdm']


    for i, savefolder in enumerate(savefolders):
        # path to simulation files

        params.savefolder = os.path.join(os.path.split(params.savefolder)[0],
                                         savefolder)
        params.figures_path = os.path.join(params.savefolder, 'figures')
        params.spike_output_path = os.path.join(params.savefolder,
                                                'processed_nest_output')
        params.networkSimParams['spike_output_path'] = params.spike_output_path


        fig_intro(params, ana_params, fraction=1.)
        


        # fig.savefig('figure_01.eps',
        #             bbox_inches='tight', pad_inches=0.01)
    # plt.show()
