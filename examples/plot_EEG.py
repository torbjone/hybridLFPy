import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import LFPy

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


savefolder = 'simulation_output_example_brunel'
figures_path = os.path.join('simulation_output_example_brunel', 'figures')

pop_r_mid_EX = return_average_somapos(os.path.join(savefolder, "populations", "{}_population_somapos.gdf".format("EX")))
pop_r_mid_IN = return_average_somapos(os.path.join(savefolder, "populations", "{}_population_somapos.gdf".format("IN")))




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

print eeg_coords

sigmas = [0.3, 1.5, 0.015, 0.3]

eeg_dict = dict(
    radii = radii,
    sigmas = sigmas,
    somapos_z = radii[0] - 1500,
    eeg_coords = eeg_coords,)
P = summed_CDM


pop_r_mid = (pop_r_mid_EX + pop_r_mid_IN) / 2
pop_r_mid[2] += eeg_dict["somapos_z"]
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


tvec = np.arange(eeg.shape[-1]) * 1000. / srate

z = [x for x in range(6)]

fig = plt.figure(figsize=[18, 9])

ax0 = fig.add_subplot(211, title="Current dipole moment")
ax1 = fig.add_subplot(212, title="EEG")


lx, = ax0.plot(tvec, cdm[0], 'g')
ly, = ax0.plot(tvec, cdm[1], 'b')
lz, = ax0.plot(tvec, cdm[2], 'r')

lx2, = ax0.plot(tvec, summed_CDM[0], c='lightgreen', ls='--')
ly2, = ax0.plot(tvec, summed_CDM[1], c='lightblue', ls='--')
lz2, = ax0.plot(tvec, summed_CDM[2], c='pink', ls='--')

ax0.legend([lx, ly, lz], ["P$_x$", "P$_y$", "P$_z$"], frameon=False)
for i, z in enumerate(z):
    ax1.plot(tvec, eeg[i] / max_eeg + z, c='k')
    ax1.plot(tvec, summed_EEG[i] / max_eeg + z, c='gray', ls="--")
    ax1.plot(tvec, eeg_summed_dipole[i] / max_eeg + z, c='olive', ls="--")

fig.savefig(os.path.join(figures_path, 'EEG.pdf'), dpi=300)
plt.close(fig)
