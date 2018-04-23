import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

sim_folder = "evoked_cdm"

name_mapping = {"p23": "L23E",
                "b23": "L23I",
                "nb23": "L23I",
                "p4": "L4E",
                "ss4(L23)": "L4E",
                "ss4(L4)": "L4E",
                "b4": "L4I",
                "nb4": "L4I",
                "p5(L23)": "L5E",
                "p5(L56)": "L5E",
                "b5": "L5I",
                "nb5": "L5I",
                "p6(L4)": "L6E",
                "p6(L56)": "L6E",
                "b6": "L6I",
                "nb6": "L6I",
                }

populations = [f for f in os.listdir(join(sim_folder, "cdm")) if os.path.isdir(join(sim_folder, "cdm", f))]

cdm_dict = {}

fig = plt.figure(figsize=[9, 9])
fig.subplots_adjust(bottom=0.3)
ax1 = fig.add_subplot(211, xlim=[600, 800], xlabel="Time (ms)")
ax2 = fig.add_subplot(212)

cdm_max_amplitude = []

pop_clrs = lambda idx: plt.cm.viridis(idx / (len(populations) - 1))
pop_clrs_list = [pop_clrs(pidx) for pidx in range(len(populations))]

for pidx, pop in enumerate(populations):

    cdm = np.load(join(sim_folder, "cdm", "summed_cdm_{}.npy".format(pop)))[:, 2]
    cdm -= np.average(cdm)
    cdm_max_amplitude.append(np.max(np.abs(cdm))/1000)
    cdm_dict[pop] = cdm
    ax1.plot(cdm / 1000, label=pop, c=pop_clrs(pidx))

p1 = ax2.bar(np.arange(len(populations)),
             cdm_max_amplitude, color=pop_clrs_list)
ax2.set_xticks(np.arange(len(populations)))
ax2.set_xticklabels(populations)
# ax2.pie(cdm_max_amplitude, labels=populations,
#             shadow=True, startangle=90)


fig.legend(loc="lower center", ncol=3, frameon=False)
plt.savefig(join(sim_folder, "pop_cdms.png"))
