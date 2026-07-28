set -e

device=cuda:7

python train.py --device $device "$@"
python test.py --device $device "$@"

