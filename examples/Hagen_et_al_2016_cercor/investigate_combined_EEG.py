import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import LFPy

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

plt.close("all")
fig = plt.figure(figsize=[9, 9])
fig.subplots_adjust(hspace=0.4)
ax1 = fig.add_subplot(211, title="EEG", ylabel="nV", xlabel="Time (ms)",
                      xlim=[670, 750])
summed_eeg = np.zeros(1000)
summed_pop_cdm = np.zeros((1000, 3))

pop_rmid = np.array([0, 0, radii[0] - 1000])

dominating_pops = ["p5(L56)", "p5(L23)",
                   "p6(L4)", "p6(L56)",
                   "p4",
                   "p23"
                   ]


for pidx, pop in enumerate(populations):
    pos_file = join(sim_folder, "populations",
                    "{}_population_somapos.gdf".format(pop))
    positions_file = open(pos_file, 'r')
    positions = np.array([pos.split()
                          for pos in positions_file.readlines()], dtype=float)
    positions_file.close()
    positions[:, 2] += radii[0]
    summed_cdm = np.load(join(sim_folder, "cdm", "summed_cdm_{}.npy".format(pop)))
    if pop in dominating_pops:
        print("Adding {} to summed cdm".format(pop))
        summed_pop_cdm[:, 2] += summed_cdm[:, 2]
    # cdm_folder = join(sim_folder, "cdm", pop)
    # files = os.listdir(cdm_folder)
    # print(pop, len(files))
    # if not len(files) == len(positions):
    #     raise RuntimeError("Missmatch!")
    # summed_cdm = np.zeros((1000, 3))
    # summed_eeg = np.zeros(1000)
    #
    #
    # for idx, f in enumerate(files):
    #     cdm = np.load(join(cdm_folder, f))[201:, :]
    #     r_mid = positions[idx]
    #     eeg_top = np.array(four_sphere_top.calc_potential(cdm, r_mid)) * 1e6  # from mV to nV
    #     summed_cdm += cdm
    #     summed_eeg += eeg_top[0, :]
    #     # if idx < 100:
    #     #     ax1.plot(eeg_top[0, :], c='gray', lw=0.5)

    eeg_pop_dipole = np.array(four_sphere_top.calc_potential(summed_cdm,
                     np.average(positions, axis=0))) * 1e6  # from mV to nV
    summed_eeg += eeg_pop_dipole[0, :]
    ax1.plot(eeg_pop_dipole[0, :] - np.average(eeg_pop_dipole[0, :]),
             c=pop_clrs_list[pidx], lw=1., label=pop)

ax1.plot(summed_eeg - np.average(summed_eeg[0]),
         c="k", lw=2., label="Sum")

simple_eeg = np.array(four_sphere_top.calc_potential(summed_pop_cdm,
                     pop_rmid)) * 1e6  # from mV to nV

ax1.plot(simple_eeg[0, :] - np.average(simple_eeg[0, :]), ":",
         c="gray", lw=3., label="Simple")


fig.legend(frameon=False, ncol=8)
plt.savefig(join(sim_folder, "combined_eeg.png"))

