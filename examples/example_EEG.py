#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example plot for LFPy: Single-synapse contribution to the LFP

Copyright (C) 2017 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""
import LFPy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colorbar

def plot_cell_to_ax(cell, ax, synidxs):
    for idx in range(cell.totnsegs):
        if idx == 0:
            ax.plot(cell.xmid[idx], cell.zmid[idx], 'ko')
        else:
            ax.plot([cell.xstart[idx], cell.xend[idx]],
                    [cell.zstart[idx], cell.zend[idx]], c='k')

    for synidx in synidxs:
        l, = ax.plot(cell.xmid[synidx], cell.zmid[synidx], '*',
                c="r", ms=10)
    ax.legend([l], ["Synapse"], frameon=False, bbox_to_anchor=[1, -0.1])

def plot_EEG_sphere(fig, eeg, x_eeg, y_eeg, z_eeg):
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(322, projection='3d',
                         title="Max EEG potential\nat 4-sphere surface")
    vmax = 6
    vmin = -vmax
    clr = lambda phi: plt.cm.PRGn((phi - vmin) / (vmax - vmin))
    clrs = clr(eeg)
    surf = ax.plot_surface(x_eeg.reshape(num_theta, num_phi),
                           y_eeg.reshape(num_theta, num_phi),
                           z_eeg.reshape(num_theta, num_phi),
                           rstride=1, cstride=1, facecolors=clrs,
                           linewidth=0, antialiased=False)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim3d(-65000, 65000)
    ax.set_ylim3d(-65000, 65000)
    ax.set_zlim3d(-65000, 65000)
    ax.view_init(10, 0)

    # colorbar
    cax = fig.add_axes([0.65, 0.75, 0.25, 0.01])
    m = plt.cm.ScalarMappable(cmap=plt.cm.PRGn)
    ticks = np.linspace(vmin, vmax, 5) # global normalization
    m.set_array(ticks)
    cbar = fig.colorbar(m, cax=cax, #format='%3.6f',
                        extend='both', orientation='horizontal')
    cbar.outline.set_visible(False)
    cbar.set_ticks(ticks)
    #unit is pV
    # cax.set_xticklabels([format(t, '3.1f') for t in ticks])
    cbar.set_label(r'$\phi$ (pV)', labelpad=1.)


def plot_EEG_electrodes(fig, x_eeg, y_eeg, z_eeg, eeg_elec_clr):
    ax = fig.add_subplot(322, xlabel="x", ylabel="z", aspect="equal",
                         title="EEG electrode positions")

    [ax.plot(x_eeg[idx], z_eeg[idx], '.', c=eeg_elec_clr(idx)) for idx in range(len(x_eeg))]

    ax.set_aspect('equal')




if __name__ == '__main__':

    # four_sphere properties
    radii = [79000., 80000., 85000., 90000.]
    sigmas = [0.3, 1.5, 0.015, 0.3]
    rad_tol = 1e-2

    # simulate cell
    syn_loc = (0, 0, 1000)

    cell_params = {'morphology': 'morphologies/ex.hoc',
                   'tstart': 0.,
                   'tstop': 40
                   }

    synapse_params = {'e': 0.,  # reversal potential
                      'syntype': 'ExpSyn',  # exponential synapse
                      'tau': 5.,  # synapse time constant
                      'weight': 0.001,  # 0.001, # synapse weight
                      'record_current': True  # record synapse current
                     }
    # create cell with parameters in dictionary
    cell = LFPy.Cell(**cell_params)
    pos = syn_loc
    synapse_params['idx'] = cell.get_closest_idx(x=pos[0], y=pos[1], z=pos[2])
    synapse = LFPy.Synapse(cell, **synapse_params)
    synapse.set_spike_times(np.array([5.]))

    cell.simulate(rec_imem=True,
                  rec_current_dipole_moment=True)

    # compute dipole
    P = cell.current_dipole_moment

    somapos = np.array([0., 0., 77500])
    # cell.set_pos(x=somapos[0], y=somapos[1], z=somapos[2])
    r_soma_syns = [cell.get_intersegment_vector(idx0=0, idx1=i) for i in cell.synidx]
    r_mid = np.average(r_soma_syns, axis=0)
    r_mid = somapos + r_mid/2.


    #measurement points
    # for nice plot use theta_step = 1 and phi_step = 1. NB: Long computation time.
    theta_step = 20
    phi_step = 90
    theta = np.linspace(-45, 45, 7)
    num_theta = theta.shape[0]
    num_tsteps = len(cell.tvec)
    theta = theta.flatten()

    theta_r = np.deg2rad(theta)

    x_eeg = (radii[3] - rad_tol) * np.sin(theta_r)
    y_eeg = np.zeros(x_eeg.shape)
    z_eeg = (radii[3] - rad_tol) * np.cos(theta_r)
    eeg_coords = np.vstack((x_eeg, y_eeg, z_eeg)).T

    # potential in 4S with db
    time_max = np.argmax(np.linalg.norm(P, axis=1))
    p = P[time_max, None]
    four_sphere = LFPy.FourSphereVolumeConductor(radii, sigmas, eeg_coords, r_mid)
    eeg = four_sphere.calc_potential(P)*1e3# from mV to uV

    # eeg = pot_db_4s.reshape(num_theta, num_tsteps)
    print np.max(np.abs(eeg))

    plt.close('all')
    fig = plt.figure(figsize=[5, 8])
    fig.subplots_adjust(left=0.15, hspace=0.5, right=0.98, top=0.95)

    morph_ax = fig.add_subplot(321, aspect=1, frameon=False,
                               title="Cell and synapse",
                               xticks=[], yticks=[])
    plot_cell_to_ax(cell, morph_ax, cell.synidx)

    eeg_elec_clr = lambda idx: plt.cm.jet(1.0 * idx / (num_theta + 1))

    plot_EEG_electrodes(fig, x_eeg, y_eeg, z_eeg, eeg_elec_clr)

    ax_p = fig.add_subplot(312, title="Current dipole moment", ylabel="nA$\cdot\mu$m")
    ax_p.plot(cell.tvec, P[:, 0], label="P$_x$")
    ax_p.plot(cell.tvec, P[:, 1], label="P$_y$")
    ax_p.plot(cell.tvec, P[:, 2], label="P$_z$")
    ax_p.axvline(cell.tvec[time_max], c='gray', ls='--')
    ax_p.legend(frameon=False, ncol=4, bbox_to_anchor=[1, -0.1])

    ax_eeg = fig.add_subplot(313, title="EEG at top", xlabel="Time (ms)",
                             ylabel='nV')
    [ax_eeg.plot(cell.tvec, eeg[idx, :], c=eeg_elec_clr(idx)) for idx in range(eeg.shape[0])]

    ax_eeg.axvline(cell.tvec[time_max], c='gray', ls='--')

    plt.savefig('example_EEG.pdf')