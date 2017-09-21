from pygcnn.utils import *
from pygcnn.indexing import *
import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
import tensorflow as tf

weights = np.array([0.5, 0.5]) #np.random.uniform(0.0, 1.0, 2)
random_graph = nx.karate_club_graph()
root_node = random_graph.nodes()[0]
d_neighborhood = make_neighborhood(random_graph, root_node, weights)

A = normalized_adj(d_neighborhood)
x = np.zeros((A.shape[0], 1))
x[np.where(np.array(random_graph.nodes()) == root_node)] = 1.0

n_timepoints = 25
hidden_layer_size = 10
filter_size = 5
input_features = propagate(x, A, n_timepoints - 1)

print input_features

w1 = np_init_weights(shape=(n_timepoints, hidden_layer_size))
w2 = np_init_weights(shape=(hidden_layer_size, filter_size))

indexer_output = np_forwardprop(input_features, w1, w2)

print indexer_output