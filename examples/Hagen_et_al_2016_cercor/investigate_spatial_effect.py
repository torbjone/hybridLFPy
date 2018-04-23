import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import LFPy

sim_folder = "evoked_cdm"

populations = [f for f in os.listdir(join(sim_folder, "cdm")) if os.path.isdir(join(sim_folder, "cdm", f))]

# four_sphere properties
radii = [79000., 80000., 85000., 90000.]
sigmas = [0.3, 1.5, 0.015, 0.3]
rad_tol = 1e-2

eeg_coords_top = np.array([[0., 0., radii[3] - rad_tol]])
four_sphere_top = LFPy.FourSphereVolumeConductor(radii, sigmas, eeg_coords_top)

def plot_EEG_sphere(fig, eeg, x_eeg, y_eeg, z_eeg):
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(122, projection='3d',
                         title="Max EEG potential\nat 4-sphere surface")
    vmax = np.max(np.abs(eeg))


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
    cax = fig.add_axes([0.6, 0.15, 0.25, 0.01])
    m = plt.cm.ScalarMappable(cmap=plt.cm.PRGn)
    ticks = np.linspace(vmin, vmax, 5) # global normalization
    m.set_array(ticks)
    cbar = fig.colorbar(m, cax=cax,
                        extend='both', orientation='horizontal')
    cbar.outline.set_visible(False)
    cbar.set_ticks(ticks)
    cbar.set_label(r'$\phi$ (nV)', labelpad=1.)



for pop in populations:
    pos_file = join(sim_folder, "populations",
                    "{}_population_somapos.gdf".format(pop))
    positions_file = open(pos_file, 'r')
    positions = np.array([pos.split() for pos in positions_file.readlines()], dtype=float)
    positions_file.close()
    positions[:, 2] += radii[0]

    cdm_folder = join(sim_folder, "cdm", pop)
    files = os.listdir(cdm_folder)
    print(pop, len(files))
    if not len(files) == len(positions):
        raise RuntimeError("Missmatch!")
    summed_cdm = np.zeros((1000, 3))
    summed_eeg = np.zeros(1000)

    plt.close("all")
    fig = plt.figure(figsize=[18, 9])
    fig.subplots_adjust(hspace=0.4)
    ax1 = fig.add_subplot(221, title="EEG", ylabel="nV", xlabel="Time (ms)")
    ax2 = fig.add_subplot(223, title="Error", ylabel="nV", xlabel="Time (ms)")

    for idx, f in enumerate(files):
        cdm = np.load(join(cdm_folder, f))[201:, :]
        r_mid = positions[idx]
        eeg_top = np.array(four_sphere_top.calc_potential(cdm, r_mid)) * 1e6  # from mV to nV
        summed_cdm += cdm
        summed_eeg += eeg_top[0, :]
        # if idx < 100:
        #     ax1.plot(eeg_top[0, :], c='gray', lw=0.5)

    eeg_pop_dipole = np.array(four_sphere_top.calc_potential(summed_cdm,
                     np.average(positions, axis=0))) * 1e6  # from mV to nV

    ax1.plot(summed_eeg, 'k', lw=2, label="Summed EEG")
    ax1.plot(eeg_pop_dipole[0, :], 'r:', lw=1.5, label="Single population dipole")
    ax2.plot(eeg_pop_dipole[0, :] - summed_eeg, 'gray', lw=2, label="Difference")
    fig.legend(frameon=False)


    #measurement points
    # for nice plot use theta_step = 1 and phi_step = 1. NB: Long computation time.
    theta_step = 5
    phi_step = 5
    theta, phi_angle = np.mgrid[0.:180.:theta_step, 0.:360.+phi_step:phi_step]

    num_theta = theta.shape[0]
    num_phi = theta.shape[1]
    theta = theta.flatten()
    phi_angle = phi_angle.flatten()

    theta_r = np.deg2rad(theta)
    phi_angle_r = np.deg2rad(phi_angle)

    x_eeg = (radii[3] - rad_tol) * np.sin(theta_r) * np.cos(phi_angle_r)
    y_eeg = (radii[3] - rad_tol) * np.sin(theta_r) * np.sin(phi_angle_r)
    z_eeg = (radii[3] - rad_tol) * np.cos(theta_r)
    eeg_coords = np.vstack((x_eeg, y_eeg, z_eeg)).T

    four_sphere = LFPy.FourSphereVolumeConductor(radii, sigmas, eeg_coords)

    time_max = np.argmax(np.abs(summed_eeg))

    pot_db_4s = four_sphere.calc_potential(summed_cdm[time_max, None],
                     np.average(positions, axis=0))
    eeg = pot_db_4s.reshape(num_theta, num_phi)*1e6# from mV to nV
    plot_EEG_sphere(fig, eeg, x_eeg, y_eeg, z_eeg)

    plt.savefig(join(sim_folder, "eeg_{}.png".format(pop)))

