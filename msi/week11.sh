#!/bin/bash
# this says that this is a shell script
#SBATCH --nodes=1
# this specifies one node
#SBATCH --ntasks=16
# this specifies that the process should run 8 tasks
#SBATCH --mem=32gb
# this saves 32 gigabytes of RAM for the process
#SBATCH -t 2:00:00
#SBATCH --mail-user=ross0778@umn.edu
# specifies my email
#SBATCH --mail-type=ALL
#SBATCH -p msismall
# this specifies which partition to run on, which in this case is msismall

cd ~/psy8712-week11/msi
module load R/4.4.2-openblas-rocky8
Rscript week11-cluster.R