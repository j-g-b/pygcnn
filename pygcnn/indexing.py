import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pygcnn.utils import *
import math

def tf_scatter_and_scale(output):
    scatter = tf.nn.softmax(output[:, :-1])
    scale = tf.expand_dims(tf.nn.sigmoid(output[:, -1]), 1)
    return tf.multiply(scale, scatter)

def tf_init_weights(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    weights = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(weights)

def tf_init_bias(shape):
    """Bias init."""
    bias = tf.abs(tf.random_normal(shape, stddev=0.01))
    return tf.Variable(bias)

def tf_index_node(graph, node, edge_weights, n_timepoints, mlp):
	d_neighborhood = make_neighborhood(graph, node, edge_weights)
	A = normalized_adj(d_neighborhood)
	x = np.zeros((A.shape[0], 1))
	for j in range(x.size):
		if d_neighborhood.nodes()[j] == node:
			x[j] = 1.0
	subgraph_features = propagate(x, A, n_timepoints - 1)
	indexer_output = mlp(tf.constant(subgraph_features, dtype=tf.float32))
	return indexer_output, d_neighborhood

def mnist_rel_coords(neighborhoods):
    rel_coords = []
    for n in range(len(neighborhoods)):
        rel_coords.append([[((n%28) - (neighborhoods[n][i]%28)), ((n/28) - (neighborhoods[n][i]/28)), n] for i in range(len(neighborhoods[n]))])
    return rel_coords

def index_graph(graph, depth, n_timepoints, edge_weight_fun):
    subgraph_features = []
    neighborhoods = []
    for node in graph.nodes():
        d_neighborhood = make_neighborhood(graph, node, depth, edge_weight_fun)
        A = normalized_adj(d_neighborhood)
        x = np.zeros((A.shape[0], 1))
        for j in range(x.size):
            if d_neighborhood.nodes()[j] == node:
                x[j] = 1.0
        subgraph_features.append(propagate(x, A, n_timepoints - 1))
        neighborhoods.append(d_neighborhood.nodes())
    return subgraph_features, neighborhoods

def plotNNFilter(units, weights):
    filters = units.shape[2]
    filter_weights = weights.shape[1]
    filter_size = weights.shape[0]
    plt.figure(1, figsize=(20,20))
    n_columns = math.ceil(np.sqrt(filters))
    n_rows = n_columns
    for j in [0,1]:
    	plt.clf()
    	for i in range(filters):
        	plt.subplot(n_rows, n_columns, i+1)
        	plt.title('Filter ' + str(i))
        	plt.imshow(np.reshape(units[j,:,i], (28,28)), interpolation="nearest", cmap="gray")
        plt.show()
    	plt.pause(0.001)
    	# plt.clf()
    	# for i in range(filter_weights):
     #    	plt.subplot(n_rows, n_columns, i+1)
     #    	plt.title('Filter ' + str(i))
     #    	plt.imshow(np.reshape(weights[:,i,0], (5,5)), interpolation="nearest", cmap="gray")
     #    plt.show()
    	# plt.pause(0.001)
