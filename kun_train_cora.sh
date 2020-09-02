TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
dataset=cora
numperclass=2
t=72
TRAINING_LOG=${dataset}_${numperclass}_${TRAINING_TIMESTAMP}_${t}.log
python train.py --dataset $dataset --numperclass $numperclass 2>&1 | tee ./log/$TRAINING_LOG


TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
numperclass=4
TRAINING_LOG=${dataset}_${numperclass}_${TRAINING_TIMESTAMP}_${t}.log
python train.py --dataset $dataset --numperclass $numperclass 2>&1 | tee ./log/$TRAINING_LOG

TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
numperclass=8
TRAINING_LOG=${dataset}_${numperclass}_${TRAINING_TIMESTAMP}_${t}.log
python train.py --dataset $dataset --numperclass $numperclass 2>&1 | tee ./log/$TRAINING_LOG

TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
numperclass=12
TRAINING_LOG=${dataset}_${numperclass}_${TRAINING_TIMESTAMP}._${t}log
python train.py --dataset $dataset --numperclass $numperclass 2>&1 | tee ./log/$TRAINING_LOG
