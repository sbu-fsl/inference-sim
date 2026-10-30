#!/usr/bin/env bash
#SBATCH --job-name=blis-gpubench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/anstyagi/blis-gpubench
#SBATCH --output=/home/anstyagi/blis-gpubench/%x-%j.out
# Kernel timings for BLIS's registry coefficients on one GPU of an ibm host (gpubench.py), submitted to the
# host's shared Slurm stage. Runs in our own venv with vLLM 0.28.0, the version KIRA serves with.
#
#   stage sbatch --job-name=blis-gpubench-ibm1 /home/anstyagi/blis-gpubench/gpubench.sbatch   # on ibm1
#   stage sbatch --job-name=blis-gpubench-ibm2 /home/anstyagi/blis-gpubench/gpubench.sbatch   # on ibm2
#
# No `sbatch --export`: it kills the stage's Slurm daemon. Output: ibm1-<jobid>.json or ibm2-<jobid>.json beside this script.
set -euo pipefail
# the stage's hostname is "stage" on both hosts, so the GPU says which host this is
case $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) in
  *"Quadro RTX 6000"*) HOST=ibm2; VENV=/home/anstyagi/venv-blis-cu129 ;;  # older driver: vLLM's CUDA 12.9 build
  *) HOST=ibm1; VENV=/home/anstyagi/venv-blis ;;
esac
echo "$HOST job ${SLURM_JOB_ID:-local} gpu ${CUDA_VISIBLE_DEVICES:-?} venv $VENV"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv
"$VENV/bin/python" gpubench.py --out "$HOST-${SLURM_JOB_ID:-local}.json"
