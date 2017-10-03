import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

# def sum_EEGs():


savefolder = 'simulation_output_example_brunel'
figures_path = os.path.join('simulation_output_example_brunel', 'figures')

eeg_name = os.path.join(savefolder, "EEGsum.h5")
cdm_name = os.path.join(savefolder, "CDMsum.h5")

#open file and get data, samplingrate
f = h5py.File(eeg_name)
data = f['data'].value
dataT = data.T - data.mean(axis=1)
eeg = dataT.T
srate = f['srate'].value
max_eeg = np.max(np.abs(eeg))
#close file object
f.close()

f = h5py.File(cdm_name)
cdm = f['data'].value
# max_eeg = np.max(np.abs(eeg))
f.close()


tvec = np.arange(data.shape[1]) * 1000. / srate

z = [x for x in range(6)]

fig = plt.figure(figsize=[18, 9])

ax0 = fig.add_subplot(211, title="Current dipole moment")
ax1 = fig.add_subplot(212, title="EEG")


lx, = ax0.plot(tvec, cdm[0])
ly, = ax0.plot(tvec, cdm[1])
lz, = ax0.plot(tvec, cdm[2])

ax0.legend([lx, ly, lz], ["P$_x$", "P$_y$", "P$_z$"], frameon=False)
for i, z in enumerate(z):
    ax1.plot(tvec, eeg[i] / max_eeg + z, c='k')

fig.savefig(os.path.join(figures_path, 'EEG.pdf'), dpi=300)
plt.close(fig)
