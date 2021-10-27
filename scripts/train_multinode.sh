#!/bin/bash
#SBATCH --job-name=MULTINODE
#SBATCH --partition gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
# SBATCH --mem=24G
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=1
# SBATCH --exclude=gn[07,11-12,14,22,24-25]
# SBATCH --exclusive
#SBATCH --output multinode.log

module load python/anaconda3 compilers/cmake-3.20 compilers/gcc-8.3.0 gpu/cuda-11.1
source activate mark15
# ntasks = nodes * gpus-per-node

#MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
#echo 'MASTER_ADDR:'
#echo $MASTER_ADDR
#MASTER_PORT=6000
# GPUS_PER_NODE=1
# NNODES=2

#             |    Node1  |   Node2    |
# ____________| p1 |  p2  |  p3  |  p4 |
# local_rank  | 0  |   1  |  0   |   1 |
# rank        | 0  |   1  |  2   |   4 |


nvidia-smi topo -m 

CMD="train_multinode.py --epochs 1 --batch_size 10 --fp16 \
                    --zero_stage 2 \
                    --dalle_output_file_name ../models/dalle-birds \
                    --image_text_folder ../data/birds-merged/ \
                    --vae_path ../models/vae-birds-final.pt \
                    --wandb_name dalle --wandb_entity darayavaus \
                    --distributed_backend deepspeed"

#CMD="hello_world.py"


#export LAUNCHER="python -u -m torch.distributed.launch \
#    --nproc_per_node $GPUS_PER_NODE \
#    --nnodes $NNODES \
#    --master_addr $MASTER_ADDR \
#    --master_port $MASTER_PORT \
#    --node_rank $SLURM_PROCID \
#    $CMD
#    "

#srun --jobid $SLURM_JOBID bash -c '$LAUNCHER'
#bash -c '$LAUNCHER'
#deepspeed --num_gpus=2 --num_nodes=2 \
#          train_dalle.py --epochs=1 --batch_size=20 --fp16 \
#                    --dalle_output_file_name=../models/dalle-birds \
#                    --image_text_folder=../data/birds-merged/ \
#                    --vae_path=../models/vae-birds-final.pt \
#                    --wandb_name=dalle --wandb_entity=darayavaus \
#                    --distributed_backend=deepspeed


WORLD_SIZE=2 GPUS_PER_NODE=1 COMM_PATH=/trinity/home/d.cherniuk/transformer_project/scripts/comm.txt srun python $CMD
