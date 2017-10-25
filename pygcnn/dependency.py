from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from competition import Competition
from sklearn.metrics import roc_auc_score
from taigapy import TaigaClient
import seaborn as sns
import random

from pygcnn.utils import *
from pygcnn.indexing import *
from pygcnn.layers import *
from pygcnn.backbone import *

# Load graph
dep_graph = nx.read_gpickle("data/pickles/gene_embedding.gpickle")

# Load training + test data
client = Competition()
taiga_client = TaigaClient()

features = taiga_client.get(name='rnaseq-gene-expression-5362', file='RNAseq_CCLE_RSEM_TPM', version='6').transpose().rename(lambda x: x.split(" (", 1)[0]).transpose()
targets = taiga_client.get(name='avana-1-2-8b72', version=2, file='gene_dependency').transpose().rename(lambda x: x.split(" (", 1)[0]).transpose()

cls = list(set(list(features.transpose())) & set(list(targets.transpose())))
random.shuffle(cls)
train_cls = cls[50:]
test_cls = cls[:50]

feature_genes = list(set(dep_graph.nodes()) & set(list(features)))
target_genes = list(set(client.get_genes()) & set(list(targets)))

features = features[feature_genes]
targets = targets[target_genes]
dep_graph = nx.relabel_nodes(dep_graph.subgraph(list(features)), {list(features)[i]: i for i in range(features.shape[1])})

features = {'train': features.loc[train_cls].dropna(axis=0, how='all'), 'test': features.loc[test_cls]}
targets = {'train': targets.loc[train_cls].dropna(axis=0, how='all'), 'test': targets.loc[test_cls]}

dependency = Dataset('dependency', np.expand_dims(features['train'].as_matrix(), axis=2), targets['train'].as_matrix(), test_size=0)

n_node_features = dependency.train_x.shape[2]
batch_size = 10
n_timepoints = 20
graph_hidden_layer_size = 10
filter_size = 32
n_filter_features = 128
hidden_layer_size = 4096
keep_prob = 1.0
learning_rate = 2.5e-5
soft_sharing = 1.0
filter_shape = [[filter_size, n_filter_features, n_node_features]]

edge_weights = [np.array([0.9, 0.9])]
keep_prob_ph = tf.placeholder(tf.float32)
hidden_mask = tf_task_mask([hidden_layer_size, targets['train'].shape[1]], soft_sharing=soft_sharing)

# Placeholders
dep_x = tf.placeholder("float32", shape=(batch_size, dependency.train_x.shape[1], dependency.train_x.shape[2]))
dep_y = tf.placeholder("float32", shape=(batch_size, dependency.train_y.shape[1]))

Convolve1 = GraphConvolution(name='first_conv', G=dep_graph, filter_shape=filter_shape, n_layers=1, n_timepoints=n_timepoints, partition_resolution=[0.1], index_hidden_layer_size=graph_hidden_layer_size, edge_weights=edge_weights, act_fun=tf.nn.elu)
conv_output = Convolve1(dep_x)
convolved_features = tf.reshape(conv_output, [-1, conv_output.get_shape().as_list()[1]*conv_output.get_shape().as_list()[2]])
OutputMLP = MLP(name='dense_output', dims=[[convolved_features.get_shape().as_list()[1], hidden_layer_size], [hidden_layer_size, dependency.train_y.shape[1]]], output_fun=tf.identity, dropout=[1 - keep_prob_ph, 0], mask=[None, hidden_mask])
yhat = OutputMLP(tf.concat([convolved_features], axis=1))

prediction = yhat
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=dep_y, logits=yhat))
updates = tf.train.AdamOptimizer(learning_rate).minimize(cost)

print "Initializing variables..."
sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss = []
focal_node_indices = []
mlp_1_weights = []
mlp_2_weights = []
plt.ion()
print "Starting training..."
for epoch in range(2000):
	dep_batch = dependency.next_batch(batch_size)
	dep_val_batch = dependency.next_test_batch(batch_size)
	deps = dep_batch[0]
	sess.run(updates, feed_dict={dep_x: deps, dep_y: dep_batch[1], keep_prob_ph: keep_prob})
	#focal_node_indices.append(np.sum(focal_node_index(Convolve1, sess), axis=0))
	if epoch % 25 == 0:
		predict = sess.run(tf.nn.sigmoid(prediction), feed_dict={dep_x: deps, dep_y: dep_batch[1], keep_prob_ph: 1.0})
		val_predict = sess.run(tf.nn.sigmoid(prediction), feed_dict={dep_x: dep_val_batch[0], dep_y: dep_val_batch[1], keep_prob_ph: 1.0})
		acc = sess.run(cost, feed_dict={dep_x: deps, dep_y: dep_batch[1], keep_prob_ph: 1.0})
		val_acc = sess.run(cost, feed_dict={dep_x: dep_val_batch[0], dep_y: dep_val_batch[1], keep_prob_ph: 1.0})
		train_auc = []
		val_auc = []
		for i in range(dep_val_batch[1].shape[1]):
			try:
				train_auc.append(roc_auc_score(np.where(dep_batch[1][:,i]>0.5, 1, 0), predict[:,i]))
				val_auc.append(roc_auc_score(np.where(dep_val_batch[1][:,i]>0.5, 1, 0), val_predict[:,i]))
			except ValueError:
				pass
		val_auc = np.array(val_auc)
		train_auc = np.array(train_auc)
		print "Cost: " + str(acc) + ", AUC: " + str(np.mean(train_auc)) + ", Test cost: " + str(val_acc) + ", Test AUC: " + str(np.mean(val_auc))
		test_set_preds = np.zeros((features['test'].as_matrix().shape[0], len(target_genes)))
		for test_epoch in range(int(np.ceil(float(features['test'].as_matrix().shape[0]) / batch_size))):
			indices = np.array(range((batch_size*test_epoch), batch_size*(test_epoch+1))) % features['test'].as_matrix().shape[0]
			print indices
			test_set_preds[indices, :] = sess.run(tf.nn.sigmoid(prediction), feed_dict={dep_x: np.expand_dims(features['test'].as_matrix()[indices,:], axis=2), dep_y: targets['test'].as_matrix()[indices,:], keep_prob_ph: 1.0})
		test_auc = []
		for i in range(targets['test'].as_matrix().shape[1]):
			try:
				test_auc.append(roc_auc_score(np.where(targets['test'].as_matrix()[:,i]>0.5, 1, 0), test_set_preds[:,i]))
			except ValueError:
				pass
		print "AUC on held-out test set is: " + str(np.mean(test_auc))

