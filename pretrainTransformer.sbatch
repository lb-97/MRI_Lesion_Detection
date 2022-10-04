#!/bin/bash 
#SBATCH --nodes=1          # requests 1 compute servers
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=3  # uses 3 compute core per task
#SBATCH --time=72:00:00
#SBATCH --mem=96GB
#SBATCH --job-name=transformer-pretraining
#SBATCH --output=transformerPretrain.out
#SBATCH --gres=gpu:v100:1

module purge

singularity exec --nv \
            --overlay /scratch/vb2183/hcp/overlay-25GB-500K2.ext3 \
            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh;python /scratch/vb2183/hcp/MRI_Lesion_Detection/pretrain_transformer.py --store_checkpoint_path /scratch/vb2183/hcp/MRI_Lesion_Detection/transformer_pretrained_checkpoints/pretrained --max_epochs 30 --cache_root /scratch/vb2183/hcp/MRI_Lesion_Detection/pretrain_transformer_cache_data --n_hidden 60 --cnn_checkpoint_path /scratch/vb2183/hcp/MRI_Lesion_Detection/cnn_pretrained_checkpoints/checkpoint_60_100.pt --n_worker_pretrain 2"
