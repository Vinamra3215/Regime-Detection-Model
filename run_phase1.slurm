#!/bin/bash
#SBATCH --job-name=regime_phase1
#SBATCH --partition=btech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/csehome/b24cm1068/Project/logs/phase1_%j.out
#SBATCH --error=/csehome/b24cm1068/Project/logs/phase1_%j.err
#SBATCH --mail-type=END,FAIL

echo "============================================================"
echo "  REGIME DETECTION MODEL — Phase 1 (HMM Regime Labeling)"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $SLURMD_NODENAME"
echo "  Start time : $(date)"
echo "============================================================"

mkdir -p /csehome/b24cm1068/Project/logs

PYTHON=/scratch/b24cm1068/miniconda3/bin/python3
PHASE1_DIR=/csehome/b24cm1068/Project/Phase_1

echo ""
echo "--- Python: $($PYTHON --version) ---"
echo "--- CUDA check ---"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "PyTorch not installed yet (needed from Phase 2)"
echo ""

echo "--- Installing / verifying dependencies ---"
$PYTHON -m pip install --quiet yfinance pandas numpy hmmlearn plotly scikit-learn ta scipy kaleido

echo ""
echo "==============================="
echo "  STEP 1: Running main.py"
echo "==============================="
cd $PHASE1_DIR
$PYTHON main.py

echo ""
echo "==============================="
echo "  STEP 2: Running evaluate.py"
echo "==============================="
$PYTHON evaluate.py

EXITCODE=$?

echo ""
echo "============================================================"
echo "  Phase 1 complete."
echo "  Exit code  : $EXITCODE"
echo "  End time   : $(date)"
echo "  Outputs    : $PHASE1_DIR/outputs/"
echo "============================================================"

exit $EXITCODE
