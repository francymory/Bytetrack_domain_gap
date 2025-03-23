#!/bin/bash
#SBATCH --job-name=YOLOX_train
#SBATCH --output="logs/slurm_logs/%x_%A_%a.out"
#SBATCH --error="logs/slurm_logs/%x_%A_%a.err"
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1  # Il numero di GPU per nodo (4 nel tuo caso)
#SBATCH --cpus-per-gpu=4  # Numero di CPU per ogni GPU
#SBATCH --partition=all_usr_prod # La partizione in cui vuoi eseguire il job
#SBATCH --account=tesi_fmorandi # Il tuo account, se necessario
#SBATCH --time=10 # Limite di tempo per il job
#SBATCH --qos=all_qos_d+


source activate bytetrack

# Controlla che Conda sia attivato correttamente
echo "Using Python from: $(which python)"
python --version

# Vai nella directory del progetto (se necessario)
cd /homes/fmorandi/stage_bytetrack/ByteTrack || exit 1


# Esporta variabili necessarie per PyTorch e il training distribuito
export PYTHONPATH=.
export WANDB_MODE=offline
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

# Variabili per la distribuzione dei nodi
IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=5000
export OMP_NUM_THREADS=1

# Log delle risorse
echo "CPUs per GPU: $SLURM_CPUS_PER_GPU"
echo "GPUs per nodo: $SLURM_GPUS_PER_NODE"
echo "MASTER ADDR: $MASTER_ADDR"
echo "MASTER PORT: $MASTER_PORT"

# Percorsi relativi o assoluti per il checkpoint e l'exp
exp_file="exps/example/motsynth/yolox_s_motsynth.py"  # Modifica con il percorso corretto del tuo exp file
checkpoint_file="pretrained/yolox_s.pth"  # Modifica con il percorso corretto del tuo checkpoint
output_dir="YOLOX_outputs"  # Modifica con la cartella dove vuoi salvare i risultati

# Avvia il training con torchrun per modalità distribuita
#srun --exclusive -c $SLURM_CPUS_PER_GPU torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_PER_NODE --rdzv-endpoint=$MASTER_ADDR --rdzv-id=$SLURM_JOB_ID --max_restarts=0 tools/train.py -f $exp_file -d 2 -b 18 --fp16 -o -c $checkpoint_file

#exec torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_PER_NODE --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv-id=$SLURM_JOB_ID --max_restarts=0 tools/train.py -f "$exp_file" -d 1 -b 8 --fp16 -o -c "$checkpoint_file"

exec $(which python) -m torch.distributed.run --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_PER_NODE --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv-id=$SLURM_JOB_ID --max_restarts=0 tools/train.py -f "$exp_file" -d 1 -b 8 --fp16 -o -c "$checkpoint_file"
