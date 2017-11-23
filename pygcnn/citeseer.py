from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from pygcnn.utils import *
from pygcnn.indexing import *
from pygcnn.layers import *

# Load Citeseer graph
X, Y, citeseer_graph = load_citeseer("data/citeseer/citeseer.content", "data/citeseer/citeseer.cites")

# Make Citeseer dataset
citeseer = Dataset('citeseer', X, Y, 0.1)

dataset_params = { \

	'dataset': citeseer \
}

graph_params = { \

	'G': citeseer_graph, \
	'depth': 1 \

}

mlp_params = { \

	'batch_size': 100, \
	'n_node_features': 3702, \
	'n_target_features': 6, \
	'signal_time': 16, \
	'index_hidden': 32, \
	'n_layers': 1, \
	'filter_shape': [[25, 32, 3702]], \
	'mlp_hidden': 512, \
	'act_fun': [tf.nn.elu] \

}

def citeseer_loss(true, pred):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true, logits=pred))

learning_params = { \

	'cost_function': citeseer_loss, \
	'optimizer': tf.train.AdamOptimizer, \
	'learning_rate': 1e-3 \
	
}

gNet = GraphNetwork('citeseer', dataset_params, graph_params, mlp_params, learning_params, orientation='node')

for i in range(8000):
	gNet.run('train')
	if i % 5 == 0:
		gNet.run('test')
		gNet.run('test')
		gNet.run('test')
