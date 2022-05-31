#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/B12_rand_vit_s_8_1_%j.out
#SBATCH --error=srun_outputs/B12_rand_vit_s_8_1_%j.err
#SBATCH --time=05:00:00
#SBATCH --job-name=s2_rand
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
srun python -u rand_vit_s2.py \
--data_path /p/scratch/hai_dm4eo/wang_yi/BigEarthNet_LMDB \
--output_dir /p/project/hai_dm4eo/wang_yi/ssl4eo-s1s2/src/checkpoints/rand/B12_vit_s_8_1 \
--lmdb \
--train_frac 0.01 \
--arch vit_small \
--patch_size 8 \
--num_workers 8 \
--batch_size_per_gpu 64 \
--epochs 100 \
--lr 0.1 \
--num_labels 19 \
--is_slurm_job \
--dist_url $dist_url \
#--pretrained_weights /p/project/hai_dm4eo/wang_yi/ssl4eo-vit/src/checkpoints/dino/vit_s_8_all/checkpoint.pth \