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
radii_name = ["Cortex", "CSF", "Skull", "Scalp"]
sigmas = [0.3, 1.5, 0.015, 0.3]
rad_tol = 1e-2
xlim = [-7000, 7000]
ylim = [radii[0]-6000, radii[-1] + 100]

max_angle = np.abs(np.rad2deg(np.arcsin(xlim[0] / ylim[0])))

plt.close("all")
fig = plt.figure(figsize=[3, 3])
ax = fig.add_subplot(111, aspect=1, frameon=False, xticks=[], yticks=[])
fig.subplots_adjust(top=0.99, bottom=0.02, left=0.02, right=0.99)
#     plt.scatter(positions[:, 0], positions[:, 2])
#     plt.subplot(122, aspect=1, xlim=[-600, 600], ylim=[-600, 600])
#     plt.scatter(positions[:, 0], positions[:, 1])
#     plt.savefig(join(sim_folder, "positions_{}.png".format(pop)))


pop_bottom = radii[0] - 1500
pop_top = radii[0]
pop_radii = 564

isometricangle=np.pi/24
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
    l_curved, = ax.plot(x_, z_, c="k", lw=2)
    ax.text(x_[-1], z_[0] - 50, radii_name[b_idx], fontsize=8,
            va="top", ha="right", color="k", rotation=-5)

ax.plot([1000, 6000], [radii[0] - 3000, radii[0] - 3000], c='gray', lw=3)
ax.text(2500, radii[0] - 2700, "5 mm", color="gray")
# for pop in populations:
#     pos_file = join(sim_folder, "populations",
#                     "{}_population_somapos.gdf".format(pop))
#     positions_file = open(pos_file, 'r')
#     positions = np.array([pos.split() for pos in positions_file.readlines()], dtype=float)
#     positions_file.close()
#     positions[:, 2] += radii[0]
    # plt.scatter(positions[:, 0], positions[:, 2], marker='.', s=0.1, color='red')

plt.savefig(join(sim_folder, "4s_head_model.png"))
