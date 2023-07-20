#!/bin/bash
#
#SBATCH --job-name=nqp_exam
#SBATCH --comment="nqp exam: exact diagonalization of spin wheel"
#SBATCH --ntasks=1
#SBATCH --partition=cip
#SBATCH --mem-per-cpu=2048
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Simon.Krumm@physik.uni-muenchen.de
#SBATCH --chdir=/home/s/Simon.Krumm/Documents/nqp/nqp-exam/src
#SBATCH --output=/home/s/Simon.Krumm/Documents/nqp/nqp-exam/slurm/slurm.%j.%N.out
#SBATCH --error=/home/s/Simon.Krumm/Documents/nqp/nqp-exam/slurm/slurm.%j.%N.err.out
python3 main.py
