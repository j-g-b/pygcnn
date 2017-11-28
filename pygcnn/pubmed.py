from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from pygcnn.utils import *
from pygcnn.indexing import *
from pygcnn.layers import *

# Load cora graph
X, Y, pubmed_graph = load_pubmed("data/pubmed/Pubmed-Diabetes.NODE.paper.tab", "data/pubmed/Pubmed-Diabetes.DIRECTED.cites.tab")

for i in range(X.shape[0]):
	rowsum = np.sum(X[i, :])
	if rowsum > 0:
		X[i, :] = X[i, :] / rowsum

train_ids = []
for i in range(3):
	samples = np.where(Y[:,i] == 1.)[0].flatten()
	train_ids.append(np.random.choice(samples, 20))

train_ids = np.concatenate(train_ids, axis=0)
test_ids = np.setdiff1d(np.array(range(Y.shape[0])), train_ids)

# Make cora dataset
pubmed = Dataset('pubmed', X, Y, split=(train_ids, test_ids))

print np.sum(pubmed.train_y, axis=0)

dataset_params = { \

	'dataset': pubmed \

}

graph_params = { \

	'G': pubmed_graph, \
	'depth': 1 \

}

mlp_params = { \

	'batch_size': 10, \
	'n_node_features': 500, \
	'n_target_features': 3, \
	'signal_time': 32, \
	'index_hidden': 16, \
	'n_layers': 1, \
	'filter_shape': [[8, 64, 500]], \
	'mlp_dims': [[64, 3]], \
	'act_fun': [tf.nn.elu] \

}

def pubmed_loss(true, pred):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true, logits=pred))

learning_params = { \

	'cost_function': pubmed_loss, \
	'optimizer': tf.train.AdamOptimizer, \
	'learning_rate': 1e-2, \
	'l2_lambda': 5e-4
	
}

gNet = GraphNetwork('pubmed', dataset_params, graph_params, mlp_params, learning_params, orientation='node')

acc_arr = []
for i in range(8000):
	tr_cost, tr_acc = gNet.run('train')
	print "iter"
	acc_arr.append(tr_acc)
	if i % 25 == 0:
		avg_acc = 0
		for j in range(25):
			cost, acc = gNet.run('test')
			avg_acc += acc
		print "\nAverage test accuracy: " + str(avg_acc / 25.0)
		print "Average train accuracy: " + str(np.mean(np.array(acc_arr)))
		acc_arr = []
