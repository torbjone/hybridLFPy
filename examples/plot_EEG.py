import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import LFPy
from plotting_convention import mark_subplots, simplify_axes

def sum_single_cell_signals(folder, population):
    all_files = glob.glob(os.path.join(folder, "{}_*.npy".format(population)))
    print "Summing {} signals from {}.".format(len(all_files), population)
    summed_signal = np.array([])
    for idx, file in enumerate(all_files):
        sig = np.load(file)
        if idx == 0:
            summed_signal = sig
        else:
            summed_signal += sig
    return summed_signal

def return_average_somapos(fname):

    f = file(fname, "r")
    data = np.array(f.read().split(), dtype=float)
    data = data.reshape(len(data)/3, 3)
    f.close()
    return np.average(data, axis=0)


def draw_head_to_ax(ax, radii, eeg_coords, eeg_clrs, pop_r_mid):

    for idx in range(eeg_coords.shape[0])[1:-1]:
        ax.plot(eeg_coords[idx, 0], eeg_coords[idx, 2], 'o',
                c=eeg_clrs(idx), ms=12, clip_on=False)

    ax.plot(pop_r_mid[0], pop_r_mid[2], 'g.')
    ax.text(pop_r_mid[0], pop_r_mid[2] - 2000, "Dipole",
            color='g', ha="center", va="top")

    circle_skalp = plt.Circle((0, 0), radii[-1], color='0.7', fill=True)
    circle_brain = plt.Circle((0, 0), radii[0], color='salmon', fill=True)
    ax.text(radii[0] - 10000, 0, "Brain", va="top",
            ha="right", rotation=-70, color="salmon")
    ax.text(radii[3], 0, "CSF\nSkull\nscalp", va="top",
            ha="center", rotation=-70, color="0.7")
    ax.text(0, radii[-1] + 5000, "EEG electrodes",
            va="bottom", ha="center", rotation=0, color="0")
    ax.add_artist(circle_skalp)
    ax.add_artist(circle_brain)

sim_dict = {"name": ""}

savefolder = 'simulation_output_example_brunel'
# savefolder = 'simulation_output_example_microcircuit'
figures_path = os.path.join('simulation_output_example_brunel', 'figures')

pop_r_mid_EX = return_average_somapos(os.path.join(savefolder, "populations",
                                                   "{}_population_somapos.gdf".format("EX")))
pop_r_mid_IN = return_average_somapos(os.path.join(savefolder, "populations",
                                                   "{}_population_somapos.gdf".format("IN")))

summed_EEG_EX = sum_single_cell_signals(os.path.join(savefolder, "EEGs"), "EX")
summed_EEG_IN = sum_single_cell_signals(os.path.join(savefolder, "EEGs"), "IN")

summed_CDM_EX = sum_single_cell_signals(os.path.join(savefolder, "CDMs"), "EX")
summed_CDM_IN = sum_single_cell_signals(os.path.join(savefolder, "CDMs"), "IN")

summed_CDM = summed_CDM_EX + summed_CDM_IN
summed_EEG = summed_EEG_EX + summed_EEG_IN

eeg_name = os.path.join(savefolder, "EEGsum.h5")
cdm_name = os.path.join(savefolder, "CDMsum.h5")

radii = [79000., 80000., 85000., 90000.]
rad_tol = 1e-2
phi_step = 90
theta = np.linspace(-45, 45, 7)
theta = theta.flatten()

theta_r = np.deg2rad(theta)

x_eeg = (radii[3] - rad_tol) * np.sin(theta_r)
y_eeg = np.zeros(x_eeg.shape)
z_eeg = (radii[3] - rad_tol) * np.cos(theta_r)
eeg_coords = np.vstack((x_eeg, y_eeg, z_eeg)).T

sigmas = [0.3, 1.5, 0.015, 0.3]

eeg_dict = dict(
    radii = radii,
    sigmas = sigmas,
    somapos_z = radii[0],
    eeg_coords = eeg_coords,)
P = summed_CDM
pop_r_mid = (pop_r_mid_EX + pop_r_mid_IN) / 2
pop_r_mid[2] += eeg_dict["somapos_z"] - 50
# potential in 4S with db
time_max = np.argmax(np.linalg.norm(P, axis=1))
p = P[time_max, None]
four_sphere = LFPy.FourSphereVolumeConductor(radii, sigmas, eeg_coords, pop_r_mid)
eeg_summed_dipole = four_sphere.calc_potential(P.T)*1e3# from mV to uV

#open file and get data, samplingrate
f = h5py.File(eeg_name)
eeg = f['data'].value
srate = f['srate'].value
max_eeg = np.max(np.abs(eeg))
#close file object
f.close()

f = h5py.File(cdm_name)
cdm = f['data'].value
# max_eeg = np.max(np.abs(eeg))
f.close()

eeg -= np.average(eeg, axis=-1)[:, None]
summed_EEG -= np.average(summed_EEG, axis=-1)[:, None]
eeg_summed_dipole -= np.average(eeg_summed_dipole, axis=-1)[:, None]

cdm_sum_error = np.max(np.abs(cdm - summed_CDM) / np.max(np.abs(cdm)))
eeg_sum_error = np.max(np.abs(eeg - summed_EEG) / np.max(np.abs(eeg)))
if cdm_sum_error > 1e-5:
    raise RuntimeError("Something wrong in summing of current dipole moments?")
if eeg_sum_error > 1e-5:
    raise RuntimeError("Something wrong in summing of EEGs?")

tvec = np.arange(eeg.shape[-1]) * 1000. / srate

fig = plt.figure(figsize=[18, 9])
fig.subplots_adjust(hspace=0.5, wspace=0.5, left=0.05, right=0.92)

xlim = [800, 1000]
ylim=[0.5, 5.5]

ax_head = fig.add_subplot(341, aspect=1, frameon=False,
                          xticks=[], yticks=[], ylim=[0, radii[-1] + 5000],
                          xlim=[-radii[-1] - 1000, radii[-1] + 1000])
ax_cdm = fig.add_subplot(345, title="Current dipole moment", xlim=xlim)
ax_cdm_dec = fig.add_subplot(349, title="Current dipole moment\nP$_z$ decomposed",
                             xlim=xlim)
ax_eeg = fig.add_subplot(142, title="EEG signal", xlim=xlim, frameon=False,
                         xticks=[], ylim=ylim)


ax_error = fig.add_subplot(143, xlim=xlim, frameon=False, xticks=[], ylim=ylim)
ax_eeg_dec = fig.add_subplot(144, title="EEG decomposed", xlim=xlim, frameon=False,
                             xticks=[], ylim=ylim)

eeg_clrs = lambda idx: plt.cm.jet(1.0 * idx / (eeg.shape[0] + 1))

draw_head_to_ax(ax_head, radii, eeg_coords, eeg_clrs, pop_r_mid)

lx, = ax_cdm.plot(tvec, cdm[0], 'g', lw=2)
ly, = ax_cdm.plot(tvec, cdm[1], 'brown', lw=2)
lz, = ax_cdm.plot(tvec, cdm[2], 'orange', lw=2)
ax_cdm.legend([lx, ly, lz], ["P$_x$", "P$_y$", "P$_z$"], frameon=False, ncol=3)

l_IN, = ax_cdm_dec.plot(tvec, summed_CDM_IN[2], c='b', lw=2)
l_EX, = ax_cdm_dec.plot(tvec, summed_CDM_EX[2], c='r', lw=2)

error = (summed_EEG - eeg_summed_dipole) / max_eeg
max_error = np.max(np.abs(error))
ax_error.set_title("EEG error\n(pop_sum - pop_dipole)/MAX|pop_sum|".format(max_error))
eeg_norm = max_eeg / 1.

for i in range(eeg.shape[0])[1:-1]:
    l_pop, = ax_eeg.plot(tvec, eeg[i] / eeg_norm + i, c=eeg_clrs(i), lw=3)
    l_dip, = ax_eeg.plot(tvec, eeg_summed_dipole[i] / eeg_norm + i, c="k", ls="--")
    l_IN, = ax_eeg_dec.plot(tvec, summed_EEG_IN[i] / eeg_norm + i, c='b')
    l_EX, = ax_eeg_dec.plot(tvec, summed_EEG_EX[i] / eeg_norm + i, c='r')

    ax_error.plot(tvec, error[i] / max_error + i, color=eeg_clrs(i), lw=2)


time_scalebar_length = (xlim[1] - xlim[0]) / 10
ax_eeg.plot([xlim[1] + 3 - time_scalebar_length, xlim[1] + 3], [0.5, 0.5], lw=3, c='k',
            clip_on=False)
ax_eeg.text(xlim[1], 0.3, "{} ms".format(time_scalebar_length), ha='right')
ax_eeg.plot([xlim[1] + 3, xlim[1] + 3], [0.5, 0.5 + 1], lw=3, c='k',
            clip_on=False)
ax_eeg.text(xlim[1] + 5, 1, "{:0.3f} $\mu$V".format(eeg_norm), ha='left')

ax_eeg_dec.plot([xlim[1] + 3 - time_scalebar_length, xlim[1] + 3], [0.5, 0.5], lw=3, c='k',
                clip_on=False)
ax_eeg_dec.text(xlim[1], 0.3, "{} ms".format(time_scalebar_length), ha='right')
ax_eeg_dec.plot([xlim[1] + 3, xlim[1] + 3], [0.5, 0.5 + 1], lw=3, c='k',
                clip_on=False)
ax_eeg_dec.text(xlim[1] + 5, 1, "{:0.3f} $\mu$V".format(eeg_norm), ha='left')

ax_error.plot([xlim[1] + 3 - time_scalebar_length, xlim[1] + 3], [0.5, 0.5], lw=3, c='k',
              clip_on=False)
ax_error.text(xlim[1], 0.3, "{} ms".format(time_scalebar_length), ha='right')
ax_error.plot([xlim[1] + 3, xlim[1] + 3], [0.5, 0.5 + 1], lw=3, c='k',
              clip_on=False)
ax_error.text(xlim[1] + 5, 1, "{:0.3e}".format(max_error), ha='left')

fig.legend([l_pop, l_dip, l_IN, l_EX],
           ["Population sum", "Population dipole", "Population IN", "Population EX"],
           frameon=False, ncol=4, bbox_to_anchor=[0.9, 0.05])

simplify_axes([ax_cdm, ax_cdm_dec])
mark_subplots(fig.axes, ypos=1.05)
fig.savefig(os.path.join(figures_path, 'EEG_2.pdf'), dpi=300)
plt.close(fig)
