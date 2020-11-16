#!/bin/sh
#SBATCH --job-name=gradexp_LOR
#SBATCH --partition=gpuv100
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --array=100,200,400,800,1600
#SBATCH --mail-type=end
#SBATCH --mail-user=jtenegg1@jhu.edu
#SBATCH --time=24:0:0

module load python/3.7
module load cuda/10.1

python /home-2/jtenegg1@jhu.edu/work/repo/hshap/hshap/experiments/rsna/LOR/compute_LOR_drop_single_explainer.py "/home-2/jtenegg1@jhu.edu/work" 300 $SLURM_ARRAY_TASK_ID 10 0