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

# Load training + test data
client = Competition()
taiga_client = TaigaClient()

features = taiga_client.get(name='rnaseq-gene-expression-5362', file='RNAseq_CCLE_RSEM_TPM', version='6').transpose().rename(lambda x: x.split(" (", 1)[0]).transpose()
targets = taiga_client.get(name='ccle-copy-number-variants-hgnc-mapped', version='4').rename(lambda x: x.split(" (", 1)[0]).transpose()

cls = list(set(list(features.transpose())) & set(list(targets.transpose())))
random.shuffle(cls)
train_cls = cls[50:]
test_cls = cls[:50]

feature_genes = list(set(list(features)) & set(list(targets)))
target_genes = list(set(list(features)) & set(list(targets)))

features = features[feature_genes]
targets = targets[target_genes]

features = {'train': features.loc[train_cls].dropna(axis=0, how='all'), 'test': features.loc[test_cls]}
targets = {'train': targets.loc[train_cls].dropna(axis=0, how='all'), 'test': targets.loc[test_cls]}

dependency = Dataset('dependency', np.expand_dims(features['train'].as_matrix(), axis=2), targets['train'].as_matrix(), test_size=0)

# Create linear graph from genomic coordinates
ccds = pd.read_csv('data/ccds/ccds.csv')
cds_dict = {}
for chrom in np.unique(ccds['#chromosome']):
	chrom_df = ccds[(ccds['#chromosome'] == chrom) & (ccds['gene'].isin(feature_genes))]
	genes = np.array(chrom_df['gene'])
	_, idx = np.unique(genes, return_index=True)
	cds_dict[chrom] = genes[np.sort(idx)].tolist()

gene2int = {}
for i in range(len(feature_genes)):
	gene2int[feature_genes[i]] = i

dep_graph = nx.Graph()
for gene in feature_genes:
	dep_graph.add_node(gene2int[gene])

for chrom in cds_dict.keys():
	for i in range(len(cds_dict[chrom])-1):
		try:
			dep_graph.add_edge(gene2int[cds_dict[chrom][i]], gene2int[cds_dict[chrom][i+1]])
		except ValueError:
			pass

n_node_features = dependency.train_x.shape[2]
batch_size = 10
n_timepoints = 20
graph_hidden_layer_size = 10
filter_size = 11
n_filter_features = 8
hidden_layer_size = 1024
keep_prob = 1.0
learning_rate = 2.5e-5
soft_sharing = 1.0
filter_shape = [[filter_size, n_filter_features, n_node_features]]

edge_weights = [np.array([0.9, 0.9, 0.9, 0.9])]
keep_prob_ph = tf.placeholder(tf.float32)

# Placeholders
dep_x = tf.placeholder("float32", shape=(batch_size, dependency.train_x.shape[1], dependency.train_x.shape[2]))
dep_y = tf.placeholder("float32", shape=(batch_size, dependency.train_y.shape[1]))

Convolve1 = GraphConvolution(name='first_conv', G=dep_graph, filter_shape=filter_shape, n_layers=1, n_timepoints=n_timepoints, partition_resolution=[0.1], index_hidden_layer_size=graph_hidden_layer_size, edge_weights=edge_weights, act_fun=tf.nn.elu)
conv_output = Convolve1(dep_x)
convolved_features = tf.reshape(conv_output, [-1, conv_output.get_shape().as_list()[1]*conv_output.get_shape().as_list()[2]])
OutputMLP = MLP(name='dense_output', dims=[[convolved_features.get_shape().as_list()[1], hidden_layer_size], [hidden_layer_size, dependency.train_y.shape[1]]], output_fun=tf.identity, dropout=[1 - keep_prob_ph, 0], mask=[None, None])
yhat = OutputMLP(tf.concat([convolved_features], axis=1))

prediction = yhat
cost = tf.nn.l2_loss(dep_y - yhat)
tss = tf.nn.l2_loss(dep_y - tf.reduce_mean(dep_y))
updates = tf.train.AdamOptimizer(learning_rate).minimize(cost)

print "Initializing variables..."
sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss = []
plt.ion()
print "Starting training..."
for epoch in range(1000):
	dep_batch = dependency.next_batch(batch_size)
	dep_val_batch = dependency.next_test_batch(batch_size)
	deps = dep_batch[0]
	sess.run(updates, feed_dict={dep_x: deps, dep_y: dep_batch[1], keep_prob_ph: keep_prob})
	if epoch % 10 == 0:
		predict = sess.run(tf.nn.sigmoid(prediction), feed_dict={dep_x: deps, dep_y: dep_batch[1], keep_prob_ph: 1.0})
		val_predict = sess.run(tf.nn.sigmoid(prediction), feed_dict={dep_x: dep_val_batch[0], dep_y: dep_val_batch[1], keep_prob_ph: 1.0})
		acc = sess.run(cost, feed_dict={dep_x: deps, dep_y: dep_batch[1], keep_prob_ph: 1.0})
		denom = sess.run(tss, feed_dict={dep_x: deps, dep_y: dep_batch[1], keep_prob_ph: 1.0})
		print "R2: " + str(1.0 - ((acc) / denom))
		test_set_preds = np.zeros((features['test'].as_matrix().shape[0], len(target_genes)))
		for test_epoch in range(int(np.ceil(float(features['test'].as_matrix().shape[0]) / batch_size))):
			indices = np.array(range((batch_size*test_epoch), batch_size*(test_epoch+1))) % features['test'].as_matrix().shape[0]
			test_set_preds[indices, :] = sess.run(prediction, feed_dict={dep_x: np.expand_dims(features['test'].as_matrix()[indices,:], axis=2), dep_y: targets['test'].as_matrix()[indices,:], keep_prob_ph: 1.0})
		print "Test R2: " + str(1.0 - (np.mean(np.square(targets['test'].as_matrix() - test_set_preds)) / np.mean(np.square(targets['test'].as_matrix() - np.mean(targets['test'].as_matrix())))))

