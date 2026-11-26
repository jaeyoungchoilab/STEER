#!/bin/bash
# ES matrix of a whole proteome on one GPU. Edit the partition, environment and paths, then: sbatch run_es.sbatch
#SBATCH --job-name=steer-es
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=steer-es.%j.log
set -euo pipefail
module load conda && conda activate steer          # the environment from environment.yml
export HF_HOME=${HF_HOME:-$PWD/hf_cache}            # where the ESM-2 weights are downloaded once
steer es proteome.fa --out es_proteome --float16
