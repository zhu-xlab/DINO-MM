#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/B2_lc_vit_s_8_crop_100_%j.out
#SBATCH --error=srun_outputs/B2_lc_vit_s_8_crop_100_%j.err
#SBATCH --time=08:00:00
#SBATCH --job-name=b2_lc
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
srun python -u linear_dino_B2.py \
--data_path /p/scratch/hai_dm4eo/wang_yi/BigEarthNet_LMDB \
--output_dir /p/project/hai_dm4eo/wang_yi/ssl4eo-s1s2/src/checkpoints/dino_lc/B2_vit_s_8_crop_100 \
--lmdb \
--train_frac 1 \
--arch vit_small \
--patch_size 8 \
--num_workers 8 \
--batch_size_per_gpu 64 \
--epochs 100 \
--lr 0.01 \
--num_labels 19 \
--is_slurm_job \
--dist_url $dist_url \
--pretrained_weights /p/project/hai_dm4eo/wang_yi/ssl4eo-s1s2/src/checkpoints/dino/B2_vit_s_8_crop/checkpoint.pth \