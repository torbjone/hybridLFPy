#!/bin/bash
################################################################################
#SBATCH --job-name cellsim16pops_modified_regular_input
#SBATCH --time 04:00:00
#SBATCH -o cellsim16pops_modified_regular_input_o.txt
#SBATCH -e cellsim16pops_modified_regular_input_e.txt
#SBATCH --mem-per-cpu=2000MB
#SBATCH --ntasks 40
#SBATCH --mail-type=ALL
################################################################################
unset DISPLAY # slurm appear to create a problem with too many displays
srun --mpi=pmi2 python cellsim16pops_modified_regular_input.py