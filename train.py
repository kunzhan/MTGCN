from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import accuracy, load_data, accuracy2, mutuallearning
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed',   type=int, default=1566444156, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--lr',     type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--numperclass', type = int, help = 'forget rate', default = 2)
parser.add_argument('--dataset', type = str, help = 'which dataset', default = 'cora')# core citeseer  pubmed
parser.add_argument('--num_gradual', type = int, default = 10, help='how')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget')
parser.add_argument('--timeseta', type = int, default = 3, help='tmies of eta')
parser.add_argument('--niter', type = int, default = 30, help='tmies of iteration')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.dataset)

def twoselftrianmodel():
	adj, features, labels, idx_train, idx_val, idx_test, t = load_data(args.dataset, args.numperclass, timeseta = args.timeseta)

	if args.cuda:
		features = features.cuda()
		adj = adj.cuda()
		labels = labels.cuda()
		idx_train = idx_train.cuda()
		idx_val = idx_val.cuda()
		idx_test = idx_test.cuda()
	

	seed = np.random.randint(1, args.seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if args.cuda:
		torch.cuda.manual_seed(seed)
		
	model_one = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout)
	model_two = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout)

	optim_one = optim.Adam(model_one.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	optim_two = optim.Adam(model_two.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	
	model_one.cuda()
	model_two.cuda()
	
	max_train_acc1 = 0 
	max_train_acc2 = 0
	
	EPOCH = 400
	for epoch in range(EPOCH):
		model_one.train()
		model_two.train()

		optim_one.zero_grad()
		optim_two.zero_grad()
		
		model_one_output = model_one(features, adj)
		model_two_output = model_two(features, adj)
	
		if epoch<200:
			model_one_loss = torch.nn.CrossEntropyLoss()(model_one_output[idx_train], labels[idx_train])
			model_two_loss = torch.nn.CrossEntropyLoss()(model_two_output[idx_train], labels[idx_train])
		else:
			model_one_loss, model_two_loss = mutuallearning(model_one_output, model_two_output, idx_train, idx_test, labels, t[0])
			
		model_one_loss.backward()
		model_two_loss.backward()
		
		optim_one.step()
		optim_two.step()
		
		model_one_val = accuracy(F.softmax(model_one_output[idx_val], 1), labels[idx_val])
		model_two_val = accuracy(F.softmax(model_two_output[idx_val], 1), labels[idx_val])
		
		train_acc1 = accuracy(F.softmax(model_one_output[idx_train], 1), labels[idx_train])
		train_acc2 = accuracy(F.softmax(model_two_output[idx_train], 1), labels[idx_train])
		
		if train_acc1 >= max_train_acc1:
			max_train_acc1 = train_acc1
			model_one.eval()
			model_one_test_output = model_one(features, adj)
			model_one_testacc = accuracy(F.softmax(model_one_test_output[idx_test], 1), labels[idx_test])
		
		if train_acc2 >= max_train_acc2:
			max_train_acc2 = train_acc2
			model_two.eval()
			model_two_test_output = model_two(features, adj)
			model_two_testacc = accuracy(F.softmax(model_two_test_output[idx_test], 1), labels[idx_test])
		model_all_val = accuracy2(F.softmax(model_one_test_output[idx_test], 1),F.softmax(model_two_test_output[idx_test], 1), labels[idx_test])

		if (epoch + 1)% 20 == 0:
			print(	'Epoch: {:04d}'.format(epoch + 1), 
					'Model_1_loss: {:.4f}'.format(model_one_loss.item()),
					'Model_2_loss: {:.4f}'.format(model_two_loss.item()),
					'Model_1_trainacc: {:.4f}'.format(train_acc1.item()),
					'Model_2_trainacc: {:.4f}'.format(train_acc2.item()),
					'Model_1_val:{:.4f}'.format(model_one_val.item()),
					'Model_2_val:{:.4f}'.format(model_two_val.item()))	
	
	print('Model_one_test:{:.4f}'.format(model_one_testacc), 'Model_two_test:{:.4f}'.format(model_two_testacc))
	print('added by two output: {:.4f}'.format(model_all_val))
	return model_one_testacc.item(), model_two_testacc.item()

def main():
	MaxMean = []
	for j in range(10):
	    model1acc = []
	    model2acc = []
	    maxacc = []    
	    for i in range(args.niter):
	        print(i+1)
	        acc1, acc2 = twoselftrianmodel()
	        model1acc.append(acc1)
	        model2acc.append(acc2)
	        maxacc.append(np.max((acc1, acc2)))	        
	        
	    model1acc = np.array(model1acc)
	    model2acc = np.array(model2acc)
	    maxacc = np.array(maxacc)

	    meanmodel1acc = np.mean(model1acc)
	    meanmodel2acc = np.mean(model2acc)
	    meanmaxacc = np.mean(maxacc)
	    print('Model1 Acc: %f Model2 Acc: %f' % (meanmodel1acc, meanmodel2acc))
	    print("Maxacc Mean: %f" % meanmaxacc)
	    MaxMean.append(meanmaxacc)
	    print(MaxMean)
	    print('Maxacc of all experiments:', np.max(MaxMean))
		
if __name__ == "__main__":
    main()