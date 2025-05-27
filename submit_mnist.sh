#!/bin/bash
#SBATCH --job-name=ds_mnist
#SBATCH --partition=hsu_gpu_priority
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1            # one launcher task per node
#SBATCH --gres=gpu:2                   # 2 GPUs per node
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.log

# pick one node as rendezvous host
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_ADDR
export MASTER_PORT=29500

# run DeepSpeed with Slurm launcher:
srun torchrun \
  --nnodes=${SLURM_JOB_NUM_NODES} \
  --nproc_per_node=2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --rdzv_id=${SLURM_JOB_ID} \
  train_mnist.py \
    --deepspeed \
    --deepspeed_config_yaml "ds_config.yml"
