import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

sim_folder = "evoked_cdm"

populations = [f for f in os.listdir(join(sim_folder, "cdm")) if os.path.isdir(join(sim_folder, "cdm", f))]

for pop in populations:

    cdm_folder = join(sim_folder, "cdm", pop)

    files = os.listdir(cdm_folder)
    print(pop, len(files))

    summed_cdm = np.zeros((1000, 3))
    plt.close("all")
    fig = plt.figure(figsize=[18, 9], )
    fig.suptitle("{}: {} cells".format(pop, len(files)))
    fig.subplots_adjust(hspace=0.5)
    ax1 = fig.add_subplot(311, title="P$_x$", xlabel="Time (ms)", ylim=[-200, 200])
    ax2 = fig.add_subplot(312, title="P$_y$", xlabel="Time (ms)", ylim=[-200, 200])
    ax3 = fig.add_subplot(313, title="P$_z$", xlabel="Time (ms)", ylim=[-200, 200])

    for idx, f in enumerate(files):
        cdm = np.load(join(cdm_folder, f))[201:, :]
        summed_cdm += cdm
        if idx < 100:
            ax1.plot(cdm[:, 0], lw=1, c="gray")
            ax2.plot(cdm[:, 1], lw=1, c="gray")
            ax3.plot(cdm[:, 2], lw=1, c="gray")

    np.save(join(sim_folder, "cdm", "summed_cdm_{}.npy".format(pop)), summed_cdm)

    ax1.plot(summed_cdm[:, 0] / len(files), lw=2, c="k")
    ax2.plot(summed_cdm[:, 1] / len(files), lw=2, c="k")
    ax3.plot(summed_cdm[:, 2] / len(files), lw=2, c="k")

    plt.savefig(join(sim_folder, "cdm_{}.png".format(pop)))
