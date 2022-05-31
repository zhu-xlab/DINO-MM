#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/B2_sup_rn50_100_%j.out
#SBATCH --error=srun_outputs/B2_sup_rn50_100_%j.err
#SBATCH --time=08:00:00
#SBATCH --job-name=B2_sup_rn50
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
srun python -u sup_rn_B2.py \
--lmdb_dir /p/scratch/hai_dm4eo/wang_yi/BigEarthNet_LMDB \
--bands B2 \
--checkpoints_dir /p/project/hai_dm4eo/wang_yi/ssl4eo-s1s2/src/checkpoints/sup/B2_rn50_100 \
--backbone resnet50 \
--train_frac 1 \
--batchsize 64 \
--lr 0.001 \
--optimizer AdamW \
--epochs 100 \
--num_workers 8 \
--seed 42 \
--dist_url $dist_url \
--cos \
#--schedule 60 80 \