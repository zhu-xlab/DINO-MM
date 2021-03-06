#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/B14_sup_rn50_1_%j.out
#SBATCH --error=srun_outputs/B14_sup_rn50_1_%j.err
#SBATCH --time=04:00:00
#SBATCH --job-name=B14_sup
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --partition=booster

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000


# load required modules
module load GCCcore/.9.3.0
module load Python
module load torchvision
module load OpenCV
module load scikit
module load TensorFlow

# activate virtual environment
source /p/project/hai_dm4eo/wang_yi/env1/bin/activate

# define available gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3

# run script as slurm job
srun python -u sup_rn_B14.py \
--lmdb_dir /p/scratch/hai_dm4eo/wang_yi/BigEarthNet_LMDB \
--bands B14 \
--checkpoints_dir /p/project/hai_dm4eo/wang_yi/ssl4eo-s1s2/src/checkpoints/sup/B14_rn50_1 \
--backbone resnet50 \
--train_frac 0.01 \
--batchsize 64 \
--lr 0.001 \
--optimizer AdamW \
--epochs 100 \
--num_workers 8 \
--seed 42 \
--dist_url $dist_url \
--cos \
#--schedule 60 80 \