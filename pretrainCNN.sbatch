#!/bin/bash 
#SBATCH --nodes=1          # requests 1 compute servers
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=3  # uses 3 compute core per task
#SBATCH --time=72:00:00
#SBATCH --mem=96GB
#SBATCH --job-name=CNN-pretraining
#SBATCH --output=CNNpretrain.out
#SBATCH --gres=gpu:rtx8000:1

module purge

singularity exec --nv \
            --overlay /scratch/vb2183/hcp/overlay-25GB-500K.ext3 \
            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh; python /scratch/vb2183/hcp/MRI_Lesion_Detection/CNN.py --hidden_size 60 --epochs 10 --alpha 0.05 --checkpoint_path /scratch/vb2183/hcp/MRI_Lesion_Detection/cnn_pretrained_checkpoints/current"