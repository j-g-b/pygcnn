import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from pygcnn.utils import *
from pygcnn.indexing import *
from pygcnn.layers import *

# Load pubmed graph
pubmed_graph = nx.to_networkx_graph(pkl.load(open('GCN_data/ind.pubmed.graph')))

pubmed = Dataset('pubmed', pkl.load(open('GCN_data/ind.pubmed.allx')).todense().transpose(), pkl.load(open('GCN_data/ind.pubmed.ally')), 0.1)

n_node_features = 1
batch_size = 50
n_timepoints = 20
graph_hidden_layer_size = 5
filter_size = 20
n_filter_features = 25
hidden_layer_size = 50
keep_prob = 1.0
learning_rate = 1e-4

edge_weights = np.array([0.9, 0.9])
keep_prob_ph = tf.placeholder(tf.float32)

# Placeholders
pubmed_x = tf.placeholder("float32", shape=(batch_size, pkl.load(open('GCN_data/ind.pubmed.allx')).todense().shape[0]))
pubmed_y = tf.placeholder("float32", shape=(batch_size, pkl.load(open('GCN_data/ind.pubmed.ally')).shape[1]))

subgraph_features, neighborhoods = index_graph(pubmed_graph, edge_weights, n_timepoints)

Convolve1 = GraphConvolution(name='first_conv', subgraph_features=subgraph_features, neighborhoods=neighborhoods, filter_shape=[filter_size, n_filter_features, n_node_features], n_timepoints=n_timepoints, edge_weights=edge_weights, act_fun=tf.nn.elu)
conv_output = Convolve1(tf.expand_dims(pubmed_x, 2))
convolved_features = tf.reshape(conv_output, [-1, conv_output.get_shape().as_list()[1]*conv_output.get_shape().as_list()[2]])
OutputMLP = MLP(name='dense_output', dims=[[convolved_features.get_shape().as_list()[1], hidden_layer_size], [hidden_layer_size, pkl.load(open('GCN_data/ind.pubmed.ally')).shape[1]]], output_fun=tf.identity, dropout=[0, 1 - keep_prob_ph])
yhat = OutputMLP(convolved_features)

prediction = tf.argmax(yhat, axis=1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=pubmed_y, logits=yhat))
updates = tf.train.AdamOptimizer().minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(pubmed_y, axis=1)), tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss = []
test_loss = []
plt.ion()
for epoch in range(2000):
	batch = pubmed.next_batch(batch_size)
	val_batch = pubmed.next_test_batch(batch_size)
	sess.run(updates, feed_dict={pubmed_x: batch[0], pubmed_y: batch[1], keep_prob_ph: keep_prob})
	predict = sess.run(prediction, feed_dict={pubmed_x: batch[0], pubmed_y: batch[1], keep_prob_ph: keep_prob})
	if epoch % 25 == 0:
		acc = sess.run(accuracy, feed_dict={pubmed_x: batch[0], pubmed_y: batch[1], keep_prob_ph: 1.0})
		loss.append(sess.run(cost, feed_dict={pubmed_x: batch[0], pubmed_y: batch[1], keep_prob_ph: 1.0}))
		test_loss.append(sess.run(cost, feed_dict={pubmed_x: val_batch[0], pubmed_y: val_batch[1], keep_prob_ph: 1.0}))
		test_acc = sess.run(accuracy, feed_dict={pubmed_x: val_batch[0], pubmed_y: val_batch[1], keep_prob_ph: 1.0})
		print "Accuracy: " + str(acc) + " Test accuracy: " + str(test_acc)
		first_layer_activation = sess.run(conv_output, feed_dict={pubmed_x: batch[0], pubmed_y: batch[1], keep_prob_ph: 1.0})
		plt.clf()
		plotNNFilter(first_layer_activation)
		plt.pause(0.01)
