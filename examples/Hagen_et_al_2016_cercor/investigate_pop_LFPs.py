import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import LFPy
import h5py

sim_folder = "evoked_cdm"

populations = [f for f in os.listdir(join(sim_folder, "cdm"))
               if os.path.isdir(join(sim_folder, "cdm", f))]

# four_sphere properties
radii = [79000., 80000., 85000., 90000.]
sigmas = [0.3, 1.5, 0.015, 0.3]
rad_tol = 1e-2

eeg_coords_top = np.array([[0., 0., radii[3] - rad_tol]])
four_sphere_top = LFPy.FourSphereVolumeConductor(radii, sigmas, eeg_coords_top)

pop_clrs = lambda idx: plt.cm.jet(idx / (len(populations) - 1))
pop_clrs_list = [pop_clrs(pidx) for pidx in range(len(populations))]

for pidx, pop in enumerate(populations):
    LFP_file = h5py.File(join(sim_folder, "populations", "{}_population_LFP.h5".format(pop)))
    LFP = np.array(LFP_file["data"])
    LFP = LFP[:, :] - LFP[:, 0, None]


    pos_file = join(sim_folder, "populations",
                    "{}_population_somapos.gdf".format(pop))
    positions_file = open(pos_file, 'r')
    positions = np.array([pos.split()
                          for pos in positions_file.readlines()], dtype=float)
    positions_file.close()
    positions[:, 2] += radii[0]
    summed_cdm = np.load(join(sim_folder, "cdm", "summed_cdm_{}.npy".format(pop)))

    eeg_pop_dipole = np.array(four_sphere_top.calc_potential(summed_cdm,
                     np.average(positions, axis=0))) * 1e6  # from mV to nV

    plt.close("all")
    fig = plt.figure(figsize=[9, 9])
    fig.subplots_adjust(hspace=0.4)
    ax1 = fig.add_axes([0.1, 0.85, 0.78, 0.1], title="EEG", ylabel="nV", xlabel="Time (ms)",
                          xlim=[660, 750])
    ax1.plot(eeg_pop_dipole[0, :] - np.average(eeg_pop_dipole[0, :]),
             c=pop_clrs_list[pidx], lw=1., label=pop)

    ax2 = fig.add_axes([0.1, 0.05, 0.78, 0.7], title="LFP", xlabel="Time (ms)",
                       ylabel="$\mu$m",
                       xlim=[660, 750])
    for idx in range(LFP.shape[0]):
        ax2.plot(LFP[idx, 201:]/np.max(np.abs(LFP))*4 - idx, c='k')
    ax2.set_yticks(-np.arange(16))
    ax2.set_yticklabels(-np.arange(16) * 100)
    plt.savefig(join(sim_folder, "LFP_and_EEG_{}.png".format(pop)))

