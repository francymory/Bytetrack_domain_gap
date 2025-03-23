#!/bin/bash
#SBATCH --output="/homes/fmorandi/stage_bytetrack/ByteTrack/YOLOX_outputs/logs/slurm_logs/%x_%A_%a.out"
#SBATCH --error="/homes/fmorandi/stage_bytetrack/ByteTrack/YOLOX_outputs/logs/slurm_logs/%x_%A_%a.err"
#SBATCH --job-name=YOLOX_train
#SBATCH --account=tesi_fmorandi
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=00:10:00
# --constraint=gpu_A40_48G|gpu_L40S_48G
#--nodelist=ailb-login-03


# Attiva l'ambiente Conda
source activate bytetrack  # Assicurati che "bytetrack" sia l'ambiente corretto


# Controlla che Conda sia attivato correttamente
which python  # Dovrebbe stampare il percorso dell'interprete Python di bytetrack
python --version  # Controlla che la versione di Python sia quella prevista

# Controlla la presenza delle GPU
echo ">>> GPU disponibili:"
nvidia-smi  # Deve mostrare le GPU disponibili

# Controlla se PyTorch rileva le GPU
echo ">>> PyTorch CUDA availability:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Controlla la versione di CUDA usata da PyTorch
echo ">>> PyTorch CUDA version:"
python -c "import torch; print('CUDA version:', torch.version.cuda)"

# Vai nella directory del progetto
cd /homes/fmorandi/stage_bytetrack/ByteTrack || exit 1

# Lancia il training
python3 tools/train.py -f exps/example/motsynth/yolox_s_motsynth.py -d 2 -b 8 --fp16 -o -c pretrained/yolox_s.pth