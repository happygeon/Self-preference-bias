#!/bin/bash -l
set -e

# (필요하면) 모듈 로드: module load cuda/12.x
source /home/happygeon02/test/bin/activate

python -c "import sys,torch;print('[PY]',sys.executable);print('[TORCH]',torch.__version__)"
# 첫 줄: GPU 상태 확인 (원하면 제거 가능)
nvidia-smi

# test.py 실행 (Slurm에서 전달받은 모든 인자 그대로 넘김)
python llama_gen.py --jsonl for_train.jsonl
# test.sh