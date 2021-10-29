#!/bin/bash -l
#SBATCH --job-name=MULTINODE
#SBATCH --partition gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=1

module load gpu/cuda-11.1
conda activate /beegfs/home/e.konyagin/miniconda3/envs/gpt_train

CMD="train_multinode.py --epochs 1 --batch_size 10 --fp16 \
                    --zero_stage 2 \
                    --dalle_output_file_name ../models/dalle-birds \
                    --image_text_folder ../data/birds-merged/ \
                    --vae_path ../models/vae-birds-final.pt \
                    --distributed_backend deepspeed"

WORLD_SIZE=2 GPUS_PER_NODE=1 COMM_PATH=/trinity/home/e.konyagin/comm.txt srun python $CMD
