#!/bin/bash
# this says that this is a shell script
#SBATCH --nodes=1
# this specifies one node
#SBATCH --cpus-per-task=8
# this specifies that the process should use 8 CPU cores
#SBATCH --mem=32gb
# this saves 32 gigabytes of RAM for the process
#SBATCH --partition=msismall
# this specifies which partition to run on, which in this case isi msismall
#SBATCH -o week11.out
# this specifies to save everything to week11.out

cd ~/psy8712-week11/msi
mkdir -p ../out
module load R/4.3.0
Rscript week11-cluster.R