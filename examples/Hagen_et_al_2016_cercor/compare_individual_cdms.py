import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from plotting_convention import mark_subplots, simplify_axes

sim_folder = "evoked_cdm_2"
num_tsteps = 16801
dt = 2**-2

populations = [f for f in os.listdir(join(sim_folder, "cdm"))
               if os.path.isdir(join(sim_folder, "cdm", f))]

ax_dict = dict(ylim=[-200, 200], xlim=[200, num_tsteps * dt])


sub_pop_groups_dict = {"L5E": ["p5(L56)", "p5(L23)"],
                       "L4E": ["p4", "ss4(L4)", "ss4(L23)"], # SHOULD WE HAVE SS4 HERE???
                       "L6E": ["p6(L4)", "p6(L56)"],
                       "L23E": ["p23"],
                       "L5I": ["b5", "nb5"],
                       "L4I": ["b4", "nb4"],
                       "L6I": ["b6", "nb6"],
                       "L23I": ["b23", "nb23"],
                       }

def plot_and_return_subgroup_cdms(subpops):
    ax1.set_ylabel("nA$\cdot\mu$m", labelpad=-3)
    ax2.set_ylabel("nA$\cdot\mu$m", labelpad=-3)
    ax3.set_ylabel("nA$\cdot\mu$m", labelpad=-3)

    tvec = np.arange(num_tsteps) * dt
    summed_cdm = np.zeros((num_tsteps, 3))
    for subpop in subpops:
        cdm_folder = join(sim_folder, "cdm", "{}".format(subpop))
        files = os.listdir(cdm_folder)

        print(pop_name, subpop, len(files))

        for idx, f in enumerate(files):
            cdm = np.load(join(cdm_folder, f))[:, :]
            summed_cdm += cdm
            if idx < 100:
                ax1.plot(tvec, cdm[:, 0], lw=0.5, c="0.7")
                ax2.plot(tvec, cdm[:, 1], lw=0.5, c="0.7")
                ax3.plot(tvec, cdm[:, 2], lw=0.5, c="0.7")

    ax1.plot(tvec, summed_cdm[:, 0] / len(files), lw=2, c="k")
    ax2.plot(tvec, summed_cdm[:, 1] / len(files), lw=2, c="k")
    ax3.plot(tvec, summed_cdm[:, 2] / len(files), lw=2, c="k")
    return summed_cdm

for pop_name, subpops in sub_pop_groups_dict.items():

    plt.close("all")
    fig = plt.figure(figsize=[18, 9], )
    fig.suptitle("{}".format(pop_name))
    fig.subplots_adjust(hspace=0.5, left=0.3)

    ax1 = fig.add_subplot(311, title="$P_x$", **ax_dict)
    ax2 = fig.add_subplot(312, title="$P_y$", **ax_dict)
    ax3 = fig.add_subplot(313, title="$P_z$", xlabel="Time (ms)", **ax_dict)
    # ax1.axvline(900, color="gray", zorder=0, lw=0.5)
    # ax2.axvline(900, color="gray", zorder=0, lw=0.5)
    # ax3.axvline(900, color="gray", zorder=0, lw=0.5)
    summed_cdm = plot_and_return_subgroup_cdms(subpops)

    simplify_axes([ax1, ax2, ax3])
    plt.savefig(join(sim_folder, "cdm_{}.png".format(pop_name)))
    plt.savefig(join(sim_folder, "cdm_{}.pdf".format(pop_name)))

    np.save(join(sim_folder, "cdm", "summed_cdm_{}.npy".format(pop_name)), summed_cdm)
