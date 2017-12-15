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
X, Y, cora_graph = load_cora("data/cora/cora.content", "data/cora/cora.cites")

for i in range(X.shape[0]):
	rowsum = np.sum(X[i, :])
	if rowsum > 0:
		X[i, :] = X[i, :] / rowsum

train_ids = []
for i in range(7):
	samples = np.where(Y[:,i] == 1.)[0].flatten()
	train_ids.append(np.random.choice(samples, 20))

train_ids = np.concatenate(train_ids, axis=0)
test_ids = np.setdiff1d(np.array(range(Y.shape[0])), train_ids)

# Make cora dataset
cora = Dataset('cora', X, Y, split=(train_ids, test_ids, np.array([0])))

print np.sum(cora.train_y, axis=0)

dataset_params = { \

	'dataset': cora \

}

graph_params = { \

	'G': cora_graph, \
	'depth': 1 \

}

mlp_params = { \

	'batch_size': 10, \
	'n_node_features': 1432, \
	'n_target_features': 7, \
	'signal_time': 32, \
	'index_hidden': 16, \
	'n_layers': 1, \
	'filter_shape': [[8, 64, 1432]], \
	'mlp_dims': [[64, 7]], \
	'act_fun': [tf.nn.elu] \

}

def cora_loss(true, pred):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true, logits=pred))

def cora_accuracy(true, pred):
	return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(true, axis=1), tf.argmax(tf.nn.softmax(pred), axis=1)), tf.float32))


learning_params = { \

	'cost_function': cora_loss, \
	'optimizer': tf.train.AdamOptimizer, \
	'learning_rate': 1e-3, \
	'l2_lambda': 5e-4
	
}

gNet = GraphNetwork('cora', dataset_params, graph_params, mlp_params, learning_params, orientation='node')

plt.ion()
train_cost = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
train_acc = [0,0,0,0,0,0]
test_cost_arr = []
train_cost_arr = []
iter_arr = []
i = 0
while gNet.dataset_params['dataset'].epoch < 50: 
	gNet.run('train')
	train_cost.append(gNet.eval(gNet.cost))
	train_acc.append(gNet.eval(cora_accuracy(gNet.Yph, gNet.prediction)))
	if i % 500 == 0:
		test_acc = []
		test_cost = []
		for j in range(25):
			gNet.run('predict')
			test_acc.append(gNet.eval(cora_accuracy(gNet.Yph, gNet.prediction)))
			test_cost.append(gNet.eval(gNet.cost))
		iter_arr.append(i / 25)
		test_cost_arr.append(np.mean(test_cost))
		train_cost_arr.append(np.mean(train_cost[-5:]))
		plt.clf()
		plt.plot(iter_arr, train_cost_arr)
		plt.plot(iter_arr, test_cost_arr)
		plt.pause(0.001)
		print "\nAverage test accuracy: " + str(np.mean(np.array(test_acc)))
		print "Average train accuracy: " + str(np.mean(np.array(train_acc[-5:])))
		acc_arr = []
	i += 1
print "Final test accuracy: " + str(gNet.eval(cora_accuracy(gNet.dataset_params['dataset'].val_y, gNet.predict(gNet.dataset_params['dataset'].val_ids))))
np.savetxt("/Users/jbryan/Desktop/poster_figures/cora_activations.txt", gNet.activate(cora.val_ids), delimiter="\t")
np.savetxt("/Users/jbryan/Desktop/poster_figures/cora_labels.txt", cora.val_y, delimiter="\t")
