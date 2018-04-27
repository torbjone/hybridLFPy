import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import LFPy
from plotting_convention import simplify_axes

sim_folder = "evoked_cdm"

populations = [f for f in os.listdir(join(sim_folder, "cdm"))
               if os.path.isdir(join(sim_folder, "cdm", f))]

# four_sphere properties
radii = [79000., 80000., 85000., 90000.]
sigmas = [0.3, 1.5, 0.015, 0.3]
rad_tol = 1e-2

eeg_coords_top = np.array([[0., 0., radii[3] - rad_tol]])
four_sphere_top = LFPy.FourSphereVolumeConductor(radii, sigmas, eeg_coords_top)


plt.close("all")
fig = plt.figure(figsize=[4, 4])
fig.subplots_adjust(hspace=0.4, top=0.8, left=0.2, bottom=0.15, right=0.98)
ax1 = fig.add_subplot(111, xlabel="Time (ms)",
                      xlim=[875, 950], ylim=[-0.7, 0.7])
ax1.set_ylabel("$\mu$V", labelpad=-3)
ax1.axvline(900, color='gray', lw=0.2)
dominating_pops = ["p5(L56)", "p5(L23)",
                   "p6(L4)", "p6(L56)",
                   "p4",
                   "p23"
                   ]
sub_pop_groups_dict = {"L5E": ["p5(L56)", "p5(L23)"],
                       "L4E": ["p4", "ss4(L4)", "ss4(L23)"],
                       "L6E": ["p6(L4)", "p6(L56)"],
                       "L23E": ["p23"],
                       "L5I": ["b5", "nb5"],
                       "L4I": ["b4", "nb4"],
                       "L6I": ["b6", "nb6"],
                       "L23I": ["b23", "nb23"],
                       }
pop_clrs = lambda idx: plt.cm.jet(idx / (len(sub_pop_groups_dict.keys()) - 1))
pop_clrs_list = {pop_name: pop_clrs(pidx) for pidx, pop_name in
                 enumerate(sub_pop_groups_dict.keys())}


summed_eeg = np.zeros(1201)
summed_pop_cdm = np.zeros((1201, 3))
pop_rmid = np.array([0, 0, radii[0] - 1000])

for pop_name, subpops in sub_pop_groups_dict.items():
    pop_eeg = np.zeros(1201)
    for subpop in subpops:
        summed_cdm = np.zeros((1201, 3))
        print(subpop)
        cdm_folder = join(sim_folder, "cdm", "{}".format(subpop))
        files = os.listdir(cdm_folder)
        pos_file = join(sim_folder, "populations",
                        "{}_population_somapos.gdf".format(subpop))
        positions_file = open(pos_file, 'r')
        positions = np.array([pos.split()
                              for pos in positions_file.readlines()], dtype=float)
        positions_file.close()
        positions[:, 2] += radii[0]
        if not len(files) == len(positions):
            raise RuntimeError("Missmatch!")

        for idx, f in enumerate(files):
            cdm = np.load(join(cdm_folder, f))[:, :]
            r_mid = positions[idx]
            eeg_top = np.array(four_sphere_top.calc_potential(cdm, r_mid)) * 1e3  # from mV to uV
            summed_cdm += cdm
            summed_eeg += eeg_top[0, :]
            pop_eeg += eeg_top[0, :]
            # if idx < 100:
            #     ax1.plot(eeg_top[0, :], c='gray', lw=0.5)

        if subpop in dominating_pops:
            print("Adding {} to summed cdm".format(subpop))
            summed_pop_cdm[:, 2] += summed_cdm[:, 2]

    # eeg_pop_dipole = np.array(four_sphere_top.calc_potential(summed_cdm,
    #                  np.average(positions, axis=0))) * 1e3  # from mV to uV
    # summed_eeg += eeg_pop_dipole[0, :]
    np.save(join(sim_folder, "EEG_{}.npy".format(pop_name)), pop_eeg)
    ax1.plot(pop_eeg - np.average(pop_eeg),
             c=pop_clrs_list[pop_name], lw=2., label=pop_name)

simple_eeg = np.array(four_sphere_top.calc_potential(summed_pop_cdm,
                     pop_rmid)) * 1e3  # from mV to uV

np.save(join(sim_folder, "summed_EEG.npy"), summed_eeg)
np.save(join(sim_folder, "simple_EEG.npy"), simple_eeg)


ax1.plot(summed_eeg - np.average(summed_eeg),
         c="k", lw=2., label="Sum")

ax1.plot(simple_eeg[0, :] - np.average(simple_eeg[0, :]), ":",
         c="gray", lw=2., label="Simple")
simplify_axes(ax1)
fig.legend(frameon=False, ncol=3, fontsize=8)
plt.savefig(join(sim_folder, "Figure_combined_EEG.png"))
plt.savefig(join(sim_folder, "Figure_combined_EEG.pdf"))

