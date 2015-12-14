#!/usr/bin/env python
'''test the same-sized network using different MPI pool sizes'''
import os
import matplotlib.pyplot as plt
import numpy as np
import brunel_alpha_nest_bench as BN


size = np.array([1, 2, 4, 8, 16, 32, 64])
ppn = 24

jobscript = '''#!/bin/bash
##################################################################
#SBATCH --job-name example_brunel_{}
#SBATCH --time {}
#SBATCH -o /homea/jinb33/jinb3323/sources/hybridLFPy/examples/example_brunel_{}.txt
#SBATCH -e /homea/jinb33/jinb3323/sources/hybridLFPy/examples/example_brunel_{}.txt
#SBATCH -N {}
#SBATCH --ntasks={}
#SBATCH --exclusive
##################################################################
# from here on we can run whatever command we want
cd /homea/jinb33/jinb3323/sources/hybridLFPy/examples
srun python example_brunel.py
'''
tottime = 640

if False:
    for s in size:
        jobname = 'example_brunel_{}.job'.format(s)
        job = file(jobname, 'w')
        job.write(jobscript.format(s, '{}:{}:00'.format(divmod(tottime, s*60)[0], divmod(tottime, s*60)[1]), s, s, s, s*ppn))
        job.close()
        os.system('sbatch {}'.format(jobname))
else:
    #gather data
    output = {}
    keys = ['network', 'spike caching', 'init pop. EX', 'run EX', 'collect EX', 'init pop. IN', 'run IN', 'collect IN', 'post processing']
    for key in keys:
        output[key] = []
    for s in size:
        simstats = np.loadtxt(os.path.join('simulation_output_example_brunel{}',
                                           'simstats.dat').format(s*ppn), dtype='object')
        for i, key in enumerate(keys):
            output[key] += [float(simstats[i, 1])]
    
    fig, ax = plt.subplots(1, figsize=(10,8))
    ax.set_color_cycle([plt.cm.rainbow(k) for k in np.linspace(0, 255, len(keys)).astype(int)])
    total = np.zeros(len(size))
    for key in keys:
        ax.loglog(size, output[key], 'o-', label=key)
        total += np.array(output[key])
    ax.loglog(size, total, lw=2, label='total')
    ax.set_xticks(size)
    ax.set_xticklabels(size*ppn)
    ax.set_xlabel('cores (-)')
    ax.set_ylabel('time (s)')
    ax.set_title('hybridLFPy scaling, Brunel network, {} neurons'.format(BN.order*5))
    ax.axis(ax.axis('tight'))
    plt.legend(loc='best', fontsize='x-small')
    plt.savefig('benchmark_example_brunel.png', dpi=200)
