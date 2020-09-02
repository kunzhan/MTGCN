TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
dataset=pubmed
t=975
niter=10
numperclass=2
TRAINING_LOG=${dataset}_${numperclass}_${TRAINING_TIMESTAMP}_${t}.log
python train.py --dataset $dataset \
		--numperclass $numperclass \
		--niter $niter 2>&1 | tee ./log/$TRAINING_LOG
numperclass=3
TRAINING_LOG=${dataset}_${numperclass}_${TRAINING_TIMESTAMP}_${t}.log
python train.py --dataset $dataset \
		--numperclass $numperclass \
		--niter $niter 2>&1 | tee ./log/$TRAINING_LOG
numperclass=7
TRAINING_LOG=${dataset}_${numperclass}_${TRAINING_TIMESTAMP}_${t}.log
python train.py --dataset $dataset \
		--numperclass $numperclass \
		--niter $niter 2>&1 | tee ./log/$TRAINING_LOG
numperclass=20
TRAINING_LOG=${dataset}_${numperclass}_${TRAINING_TIMESTAMP}_${t}.log
python train.py --dataset $dataset \
		--numperclass $numperclass \
		--niter $niter 2>&1 | tee ./log/$TRAINING_LOG

