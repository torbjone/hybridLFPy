#!/bin/bash
################################################################################
#SBATCH --job-name cellsim16pops_modified_spontan
#SBATCH --time 5:00:00
#SBATCH -o cellsim16pops_modified_spontan_o.txt
#SBATCH -e cellsim16pops_modified_spontan_e.txt
#SBATCH --mem-per-cpu=2000MB
#SBATCH --ntasks 400
#SBATCH --mail-type=ALL
################################################################################
unset DISPLAY # slurm appear to create a problem with too many displays
srun --mpi=pmi2 python cellsim16pops_modified_spontan.py