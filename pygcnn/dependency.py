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

from pygcnn.utils import *
from pygcnn.indexing import *
from pygcnn.layers import *

# Build graph
client = Competition()
taiga_client = TaigaClient()

train_cls = client.get_training_set().index.format()
test_cls = client.get_test_set().index.format()

exp_features = taiga_client.get(name='ccle-rnaseq-expression-genes', version='3').transpose().rename(lambda x: x.split(" (", 1)[0]).transpose()
mut_features = taiga_client.get(name='ccle-mis-mut-binary-matrix', version='1').transpose().rename(lambda x: x.split(" (", 1)[0]).transpose()
cn_features = taiga_client.get(name='ccle-copy-number-variants-hgnc-mapped', version='4').rename(lambda x: x.split(" (", 1)[0]).transpose()
lin_features = taiga_client.get(name='ccle-lines-lineages', version='4')
cls_to_train = list(set(train_cls) & set(list(exp_features.index)) & set(list(mut_features.index)) & set(list(cn_features.index)) & set(list(exp_features.index)))
exp_features = {'train': exp_features.loc[cls_to_train].dropna(axis=0, how='all'), 'test': exp_features.loc[test_cls]}
mut_features = {'train': mut_features.loc[cls_to_train].dropna(axis=0, how='all'), 'test': mut_features.loc[test_cls]}
cn_features = {'train': cn_features.loc[cls_to_train].dropna(axis=0, how='all'), 'test': cn_features.loc[test_cls]}
lin_features = {'train': lin_features.loc[cls_to_train].dropna(axis=0, how='all'), 'test': lin_features.loc[test_cls]}

target_genes_to_use = client.genes
targets = client.get_training_set().loc[list(exp_features['train'].index)].dropna(axis=0, how='all')[target_genes_to_use]

msigdb_dict = dict_from_msigdb("data/msigdb.v6.0.symbols.gmt.txt", max_size=25)
gene_set_indx, gene_indx, msigdb_matrix = sp_matrix_from_msigdb_dict(msigdb_dict)

top_k = select_k_best(exp_features['train'], targets, 5)
query_genes = list(set(list(pd.read_csv('data/gene_list.csv')['gene']) + client.genes + np.concatenate(top_k.values()).tolist()) & set(gene_indx.keys()))
query_gene_indices = [gene_indx[gene] for gene in query_genes]

msigdb_adj = msigdb_matrix.transpose().dot(msigdb_matrix)
msigdb_adj = msigdb_adj[query_gene_indices, :]
msigdb_adj.eliminate_zeros()
msigdb_adj = msigdb_adj.tocoo()
inv_map = {v: k for k, v in gene_indx.iteritems()}
queried_genes = query_genes
#queried_genes = [inv_map[i] for i in np.unique(msigdb_adj.col)]
genes_to_use = list(set(query_genes + queried_genes) & set(list(exp_features['train'])) & set(list(mut_features['train'])) & set(gene_indx.keys()) & set(list(cn_features['train'])))

features_to_use = np.stack([exp_features['train'][genes_to_use].as_matrix(), mut_features['train'][genes_to_use].as_matrix(), cn_features['train'][genes_to_use].as_matrix()], axis=2)

test_features_to_use = np.stack([exp_features['test'][genes_to_use].as_matrix(), mut_features['test'][genes_to_use].as_matrix(), cn_features['test'][genes_to_use].as_matrix()], axis=2)
test_set = client.get_test_set()

dependency = Dataset('dependency', features_to_use, targets.as_matrix(), 0)

msigdb_adj = msigdb_matrix.transpose().dot(msigdb_matrix)
genes_to_use_indices = [gene_indx[gene] for gene in genes_to_use]
msigdb_adj = msigdb_adj[genes_to_use_indices, :]
msigdb_adj = msigdb_adj[:, genes_to_use_indices]
msigdb_adj = msigdb_adj.tocoo()
for i in range(len(msigdb_adj.data)):
	if msigdb_adj.row[i] == msigdb_adj.col[i]:
		msigdb_adj.data[i] = 0

msigdb_adj.eliminate_zeros()
msigdb_adj.data = msigdb_adj.data / np.max(msigdb_adj.data)
dist_mat = np.corrcoef(np.transpose(dependency.train_x[:,:,0] + 1e-6*np.random.randn(dependency.train_x.shape[0], dependency.train_x.shape[1])))
np.fill_diagonal(dist_mat, 0)
msigdb_adj = msigdb_adj.tocoo()
for i in range(len(msigdb_adj.data)):
	msigdb_adj.data[i] = np.sign(dist_mat[msigdb_adj.row[i], msigdb_adj.col[i]])*msigdb_adj.data[i]

msigdb_adj.eliminate_zeros()
dep_graph = nx.minimum_spanning_tree(nx.from_scipy_sparse_matrix(msigdb_adj))

n_node_features = dependency.train_x.shape[2]
batch_size = 10
n_timepoints = 20
graph_hidden_layer_size = 5
filter_size = 25
n_filter_features = 25
hidden_layer_size = 5000
keep_prob = 1.0
learning_rate = 2.5e-5
soft_sharing = 0.95

edge_weights = np.array([0.9, 0.9])
keep_prob_ph = tf.placeholder(tf.float32)
hidden_mask = tf_task_mask([hidden_layer_size, targets.shape[1]], soft_sharing=soft_sharing)

# Placeholders
dep_x = tf.placeholder("float32", shape=(batch_size, features_to_use.shape[1], features_to_use.shape[2]))
dep_y = tf.placeholder("float32", shape=(batch_size, targets.shape[1]))
lin_x = tf.placeholder("float32", shape=(batch_size, lin_features['train'].shape[1]))

subgraph_features, neighborhoods = index_graph(dep_graph, edge_weights, n_timepoints)

Convolve1 = GraphConvolution(name='first_conv', subgraph_features=subgraph_features, neighborhoods=neighborhoods, filter_shape=[filter_size, n_filter_features, n_node_features], n_timepoints=n_timepoints, edge_weights=edge_weights, act_fun=tf.nn.elu)
conv_output = Convolve1(dep_x)
convolved_features = tf.reshape(conv_output, [-1, conv_output.get_shape().as_list()[1]*conv_output.get_shape().as_list()[2]])
OutputMLP = MLP(name='dense_output', dims=[[convolved_features.get_shape().as_list()[1] + lin_features['train'].shape[1], hidden_layer_size], [hidden_layer_size, targets.shape[1]]], output_fun=tf.identity, dropout=[1 - keep_prob_ph, 0], mask=[None, hidden_mask])
yhat = OutputMLP(tf.concat([convolved_features, lin_x], axis=1))

prediction = yhat
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=dep_y, logits=yhat))
updates = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss = []
test_loss = []
plt.ion()
for epoch in range(1000):
	dep_batch = dependency.next_batch(batch_size)
	dep_test_batch = dependency.next_test_batch(batch_size)
	deps = dep_batch[0]
	sess.run(updates, feed_dict={dep_x: deps, dep_y: dep_batch[1], lin_x: lin_features['train'].as_matrix()[dep_batch[2].flatten(), :], keep_prob_ph: keep_prob})
	if epoch % 25 == 0:
		predict = sess.run(tf.nn.sigmoid(prediction), feed_dict={dep_x: deps, dep_y: dep_batch[1], lin_x: lin_features['train'].as_matrix()[dep_batch[2].flatten(), :], keep_prob_ph: 1.0})
		test_predict = sess.run(tf.nn.sigmoid(prediction), feed_dict={dep_x: dep_test_batch[0], dep_y: dep_test_batch[1], lin_x: lin_features['train'].as_matrix()[dep_test_batch[2].flatten(), :], keep_prob_ph: 1.0})
		acc = sess.run(cost, feed_dict={dep_x: deps, dep_y: dep_batch[1], lin_x: lin_features['train'].as_matrix()[dep_batch[2].flatten(), :], keep_prob_ph: 1.0})
		test_acc = sess.run(cost, feed_dict={dep_x: dep_test_batch[0], dep_y: dep_test_batch[1], lin_x: lin_features['train'].as_matrix()[dep_test_batch[2].flatten(), :], keep_prob_ph: 1.0})
		train_auc = []
		test_auc = []
		for i in range(dep_test_batch[1].shape[1]):
			try:
				train_auc.append(roc_auc_score(np.where(dep_batch[1][:,i]>0.5, 1, 0), predict[:,i]))
				test_auc.append(roc_auc_score(np.where(dep_test_batch[1][:,i]>0.5, 1, 0), test_predict[:,i]))
			except ValueError:
				pass
		test_auc = np.array(test_auc)
		train_auc = np.array(train_auc)
		print "Cost: " + str(acc) + ", AUC: " + str(np.mean(train_auc)) + ", Test cost: " + str(test_acc) + ", Test AUC: " + str(np.mean(test_auc))

test_set_matrix = np.zeros((test_set.shape[0], len(target_genes_to_use)))
for test_epoch in range(int(np.ceil(float(test_set.shape[0]) / batch_size))):
	indices = np.array(range((batch_size*test_epoch), batch_size*(test_epoch+1))) % test_set.shape[0]
	print indices
	test_set_matrix[indices, :] = sess.run(tf.nn.sigmoid(prediction), feed_dict={dep_x: test_features_to_use[indices,:,:], dep_y: test_set_matrix[indices,:], lin_x: lin_features['test'].as_matrix()[indices, :], keep_prob_ph: 1.0})

test_set[target_genes_to_use] = test_set_matrix
client.score(test_set)

