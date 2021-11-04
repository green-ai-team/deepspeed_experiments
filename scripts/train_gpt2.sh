#!/bin/bash
#SBATCH --job-name=GPT2
#SBATCH --partition gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
# SBATCH --mem=24G
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:3
# SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=1
# SBATCH --exclude=gn[07,11-12,14,22,24-25]
# SBATCH --exclusive
#SBATCH --output gpt2.log

module load compilers/gcc-8.3.0
module load compilers/cmake-3.20
module load gpu/cuda-11.1
module load python/anaconda3
source activate mark15

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib64:/trinity/home/d.cherniuk/llvm-9-dev/usr/lib/llvm-9/lib

# deepspeed --num_gpus=2 \
#           train_gpt2.py --epochs 1 --batch_size 4 \
#           --zero_stage 0

#GPUS_PER_NODE=1
#WORLD_SIZE=2 
COMM_PATH=/trinity/home/d.cherniuk/transformer_project/scripts/comm.txt \
deepspeed train_gpt2.py \
               --epochs 1 \
               --batch_size 6 \
               --zero_stage 0 \
               --wandb_entity "darayavaus" \
               --wandb_name "gpt2"