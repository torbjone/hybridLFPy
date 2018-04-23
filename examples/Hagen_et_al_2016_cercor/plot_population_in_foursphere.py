import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import LFPy

sim_folder = "evoked_cdm"

populations = [f for f in os.listdir(join(sim_folder, "cdm")) if os.path.isdir(join(sim_folder, "cdm", f))]

# four_sphere properties
radii = [79000., 80000., 85000., 90000.]
radii_name = ["Brain", "CSF", "Skull", "Scalp"]
sigmas = [0.3, 1.5, 0.015, 0.3]
rad_tol = 1e-2
xlim = [-8000, 8000]
ylim = [radii[0]-6000, radii[-1] + 100]

max_angle = np.abs(np.rad2deg(np.arcsin(xlim[0] / ylim[0])))

plt.close("all")
fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot(111, aspect=1)
#     plt.scatter(positions[:, 0], positions[:, 2])
#     plt.subplot(122, aspect=1, xlim=[-600, 600], ylim=[-600, 600])
#     plt.scatter(positions[:, 0], positions[:, 1])
#     plt.savefig(join(sim_folder, "positions_{}.png".format(pop)))

angle = np.linspace(-max_angle, max_angle, 100)
for b_idx in range(len(radii)):
    x_ = radii[b_idx] * np.sin(np.deg2rad(angle))
    z_ = radii[b_idx] * np.cos(np.deg2rad(angle))
    l_curved, = ax.plot(x_, z_, ':', c="k")
    ax.text(xlim[1], z_[0] - 100, radii_name[b_idx], va="top", ha="right", color="k")

#
#
for pop in populations:
    pos_file = join(sim_folder, "populations",
                    "{}_population_somapos.gdf".format(pop))
    positions_file = open(pos_file, 'r')
    positions = np.array([pos.split() for pos in positions_file.readlines()], dtype=float)
    positions_file.close()
    positions[:, 2] += radii[0]
    plt.scatter(positions[:, 0], positions[:, 2], s=1)

plt.savefig(join(sim_folder, "4s_head_model.png"))
