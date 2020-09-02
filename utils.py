import numpy as np
import scipy.sparse as sp
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import pickle as pkl
import networkx as nx

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_features(features, feature_type='bow'):
    if feature_type == 'bow':
        # """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        # normalize(features, norm='l1', axis=1, copy=False)
    elif feature_type == 'tfidf':
        transformer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False)
        features = transformer.fit_transform(features)
    elif feature_type == 'none':
        features = sp.csr_matrix(sp.eye(features.shape[0]))
    else:
        raise ValueError('Invalid feature type: ' + str(feature_type))
    return features

def normalize_adj(adj, type='sym'):
    """Symmetrically normalize adjacency matrix."""
    if type == 'sym':
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        # d_inv_sqrt = np.power(rowsum, -0.5)
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # return adj*d_inv_sqrt*d_inv_sqrt.flatten()
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    elif type == 'rw':
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj_normalized = d_mat_inv.dot(adj)
        return adj_normalized

def load_data(dataset_str, train_size, validation_size = 500, timeseta = 3, validate = False, shuffle=True):
	"""Load data."""
	if dataset_str in ['USPS-Fea', 'CIFAR-Fea', 'Cifar_10000_fea', 'Cifar_R10000_fea', 'MNIST-Fea', 'MNIST-10000', 'MNIST-5000']:
		data = sio.loadmat('data/{}.mat'.format(dataset_str))
		l = data['labels'].flatten()
		labels = np.zeros([l.shape[0],np.max(data['labels'])+1])
		labels[np.arange(l.shape[0]), l.astype(np.int8)] = 1
		features = data['X']
		sample = features[0].copy()
		adj = data['G']
	else:
		names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
		objects = []
		for i in range(len(names)):
			with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
				if sys.version_info > (3, 0):
					objects.append(pkl.load(f, encoding='latin1'))
				else:
					objects.append(pkl.load(f))

		x, y, tx, ty, allx, ally, graph = tuple(objects)
		adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
		test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
		test_idx_range = np.sort(test_idx_reorder)

		if dataset_str == 'citeseer':
			test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
			tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
			tx_extended[test_idx_range - min(test_idx_range), :] = tx
			tx = tx_extended
			ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
			ty_extended[test_idx_range - min(test_idx_range), :] = ty
			ty = ty_extended

		features = sp.vstack((allx, tx)).tolil()
		# features = sp.eye(features.shape[0]).tolil()
		# features = sp.lil_matrix(allx)

		labels = np.vstack((ally, ty))
		# labels = np.vstack(ally)

		features[test_idx_reorder, :] = features[test_idx_range, :]
		labels[test_idx_reorder, :] = labels[test_idx_range, :]
		features = preprocess_features(features)

	global all_labels
	all_labels = labels.copy()

	# split the data set
	idx = np.arange(len(labels))
	no_class = labels.shape[1]  # number of class
	train_size = [train_size for i in range(labels.shape[1])]
	if shuffle:
		np.random.shuffle(idx)
	idx_train = []
	count = [0 for i in range(no_class)]
	label_each_class = train_size
	next = 0
	for i in idx:
		if count == label_each_class:
			break
		next += 1
		for j in range(no_class):
			if labels[i, j] and count[j] < label_each_class[j]:
				idx_train.append(i)
				count[j] += 1

	test_size = None
	if validate:
		if test_size:
			assert next+validation_size<len(idx)
		idx_val = idx[next:next+validation_size]
		assert next+validation_size+test_size < len(idx)
		idx_test = idx[-test_size:] if test_size else idx[next+validation_size:]

	else:
		if test_size:
			assert next+test_size<len(idx)
		idx_val = idx[-test_size:] if test_size else idx[next:]
		idx_test = idx[-test_size:] if test_size else idx[next:]

	print('labels of each class : ', np.sum(labels[idx_train], axis=0))
	
	eta = np.float(adj.shape[0])/(np.float(adj.sum())/adj.shape[0])**2
	t = (labels[idx_train].sum(axis=0)*timeseta*eta/labels[idx_train].sum()).astype(np.int64)
	
	features = torch.FloatTensor(np.array(features.todense()))
	labels = torch.LongTensor(np.argmax(labels,1))
	adj = adj + sp.eye(adj.shape[0])
	adj = normalize_adj(adj)
	adj = sparse_mx_to_torch_sparse_tensor(adj)
	idx_train = torch.LongTensor(idx_train)
	idx_val = torch.LongTensor(idx_val)
	idx_test = torch.LongTensor(idx_test)

	return adj, features, labels, idx_train, idx_val, idx_test, t

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy2(output1,output2, labels):
	preds1 = output1.max(1)
	preds2 = output2.max(1)
	L = len(labels)
	preds = preds1[0]*0
	for k in range(L):
		if preds1[0][k]>=preds2[0][k]:
			preds[k] = preds1[1][k]
		else:
			preds[k] = preds2[1][k]
	preds = preds.type_as(labels)
	correct = preds.eq(labels).double()
	correct = correct.sum() / L
	return correct


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def selftraining(prediction, t, idx_train, labels):
	prediction = prediction.detach().cpu().numpy()
	idx_train = idx_train.cpu().numpy()
	labels = labels.cpu().numpy()
	
	new_gcn_index = np.argmax(prediction, 1)
	confidence = np.max(prediction, 1)
	sorted_index = np.argsort(-confidence)

	no_class = prediction.shape[1]  # number of class
	t = np.array(np.tile(t, (1,no_class)))
	t = t[0]

	if hasattr(t, '__getitem__'):
		assert len(t) >= no_class
		index = []
		count = [0 for i in range(no_class)]
		for i in sorted_index:
			for j in range(no_class):
				if new_gcn_index[i] == j and count[j] < t[j] and not (i in idx_train):
					index.append(i)
					count[j] += 1
	else:
		index = sorted_index[:t]

	prediction = new_gcn_index
	prediction[idx_train] = labels[idx_train]

	return torch.LongTensor(index).cuda(), torch.LongTensor(prediction).cuda()

def plabelscoefficients(pseudolabels):
	numclass = float(pseudolabels.shape[1])
	max_entropy = torch.log(torch.tensor(numclass))
	log_pseudolabels = torch.log(pseudolabels)
	pseudolabels_entropy = - torch.sum(pseudolabels * log_pseudolabels, 1)
	coef = 1 - pseudolabels_entropy / max_entropy
	return coef

def mutuallearning(model_one, model_two, idx_train, idx_test, labels, t):

	loss_kl = torch.nn.KLDivLoss(reduction='batchmean')
	loss_ce = torch.nn.CrossEntropyLoss()

	model_one_index, model_one_prediction = selftraining(F.softmax(model_one, 1), t, idx_train, labels)
	model_two_index, model_two_prediction = selftraining(F.softmax(model_two, 1), t, idx_train, labels)

	loss_one_sup = loss_ce(model_one[idx_train], labels[idx_train])
	loss_two_sup = loss_ce(model_two[idx_train], labels[idx_train])

	loss_one_kl = loss_kl(F.log_softmax(model_one[model_two_index], dim = 1), F.softmax(Variable(model_two[model_two_index]), dim=1))
	loss_two_kl = loss_kl(F.log_softmax(model_two[model_one_index], dim = 1), F.softmax(Variable(model_one[model_one_index]), dim=1))

	pred1 = F.softmax(model_one[model_one_index], 1)
	pred2 = F.softmax(model_two[model_two_index], 1)
	coef1 = plabelscoefficients(pred1)
	coef2 = plabelscoefficients(pred2)

	loss_one_reg = torch.mean(coef2.detach() * torch.nn.CrossEntropyLoss(reduction = 'none')(model_one[model_two_index], pred2.max(1)[1]))
	loss_two_reg = torch.mean(coef1.detach() * torch.nn.CrossEntropyLoss(reduction = 'none')(model_two[model_one_index], pred1.max(1)[1]))


	model_one_loss = loss_one_sup + loss_one_kl + loss_one_reg
	model_two_loss = loss_two_sup + loss_two_kl + loss_two_reg

	return model_one_loss, model_two_loss