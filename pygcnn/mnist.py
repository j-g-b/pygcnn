from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from pygcnn.utils import *
from pygcnn.indexing import *
from pygcnn.layers import *

# Build mnist graph
pic_size = 28
mnist_graph = nx.grid_2d_graph(pic_size, pic_size)
coord_map = {mnist_graph.nodes()[i]: mnist_graph.nodes()[i][0]*pic_size + mnist_graph.nodes()[i][1] for i in range(len(mnist_graph.nodes()))}
mnist_graph = nx.relabel_nodes(mnist_graph, coord_map)

for node in mnist_graph.nodes():
	tl = node - pic_size - 1
	tr = node - pic_size + 1
	bl = node + pic_size - 1
	br = node + pic_size + 1
	nodes_to_add = [node]
	if tl >= 0 and not node%pic_size == 0:
		nodes_to_add.append(tl)
	if tr >= 0 and not node%pic_size == (pic_size-1):
		nodes_to_add.append(tr)
	if bl < pic_size**2 and not node%pic_size == 0:
		nodes_to_add.append(bl)
	if br < pic_size**2 and not node%pic_size == (pic_size-1):
		nodes_to_add.append(br)
	mnist_graph.add_star(nodes_to_add)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Make MNIST dataset
mnist_data = Dataset('MNIST', np.expand_dims(mnist.train.images, 2), mnist.train.labels, 0.1)

dataset_params = { \

	'dataset': mnist_data \

}

graph_params = { \

	'G': mnist_graph, \
	'depth': 2 \
}

mlp_params = { \

	'batch_size': 50, \
	'n_node_features': 1, \
	'n_target_features': 10, \
	'signal_time': 16, \
	'index_hidden': 32, \
	'n_layers': 1, \
	'filter_shape': [[25, 32, 1]], \
	'mlp_hidden': 512, \
	'act_fun': [tf.nn.elu] \

}

def mnist_loss(true, pred):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true, logits=pred))

learning_params = { \

	'cost_function': mnist_loss, \
	'optimizer': tf.train.AdamOptimizer, \
	'learning_rate': 1e-3 \
	
}

gNet = GraphNetwork('MNIST', dataset_params, graph_params, mlp_params, learning_params, orientation='graph')

for i in range(8000):
	gNet.train()
