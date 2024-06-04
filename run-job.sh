#!/bin/bash
#SBATCH --job-name=tensorflow-test-gpu
#SBATCH --partition=ampere
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G

# conda commands
module load gcc/10.2.0 miniconda3/22.11.1-wqd7jjw cuda/11.1.0 cudnn/8.0.4.30-11.0-linux-x64
source $CONDA_PROFILE/conda.sh
#conda create -n tf-3 python=3.8
#activate environment; tf-2 does not include tensorflow-gpu
conda activate tf-3
#conda activate tf-2

#pip install matplotlib==3.6.3
#pip install scikit-learn
#pip install tqdm
#pip install pandas==1.1.5
#pip install numpy~=1.19.2
#pip install seaborn==0.11.2
#if cudnn works
#pip install tensorflow==2.5.0
#if cudnn is not compatible
#pip install tensorflow-gpu==2.5.0
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib


# Logging GPU stats
nvidia-smi --loop=300 > nvidia-smi.log &

function terminate_monitor_gpu_status()
{
  nvidia_smi_pid=$(pgrep -u $USER nvidia-smi)
  kill $nvidia_smi_pid
}

trap 'terminate_monitor_gpu_status' KILL TERM EXIT

# code to be executed
# python3 data_processing/SimCLR.py
python3 ./code/baselines/SimCLR/train_model.py
# python3 ./code/baselines/SimCLR/finetune_model.py
#python3 ./code/baselines/SimCLR/train_model_freeze_alternatives.py