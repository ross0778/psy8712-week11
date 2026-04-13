#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=32gb
#SBATCH -t 2:00:00
#SBATCH --mail-user=ross0778@umn.edu
#SBATCH --mail-type=ALL
#SBATCH -p msismall

cd ~/psy8712-week11/msi
module load R/4.4.2-openblas-rocky8
Rscript week11-cluster.R