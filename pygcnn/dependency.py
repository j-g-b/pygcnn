import tensorflow as tf
import networkx as nx
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.metrics import roc_auc_score
import random
import sys
import datetime

from pygcnn.utils import *
from pygcnn.indexing import *
from pygcnn.layers import *

# Load graph
dep_graph = parse_gene_graph("data/reactome_functional_interactions/FIsInGene_022717_with_annotations.txt")
hallmark_genes = pd.read_csv("data/cancer_gene_census/Census_allThu_Dec_7_14_32_15_2017.csv")
hallmark_genes = np.array(hallmark_genes['Gene Symbol']).tolist()

features = pd.read_csv("data/b7998b47ac65462a9e6ec7c4e973df66_RNAseq_CCLE_RSEM_TPM.csv", index_col=0).transpose().rename(lambda x: x.split(" (", 1)[0]).transpose()
targets = pd.read_csv("data/bffb8b6c5620436eb4234a8b449a146e_gene_dependency.csv", index_col=0).transpose().rename(lambda x: x.split(" (", 1)[0]).transpose()
feature_genes = list(set(dep_graph.nodes()) & set(list(features)) & set(hallmark_genes + [sys.argv[1]]))
target_genes = [sys.argv[1]]

np.savetxt("/Users/jbryan/Desktop/poster_figures/gene_names.txt", np.array(feature_genes), delimiter="\t", fmt="%s")

cls = list(set(list(features.transpose())) & set(list(targets.transpose())))

features = features[feature_genes]
targets = targets[target_genes]
gene2int = {list(features)[i]: i for i in range(features.shape[1])}
int2gene = {i: list(features)[i] for i in range(features.shape[1])}
dep_graph = nx.relabel_nodes(dep_graph.subgraph(list(features)), gene2int)

features = features.loc[cls].dropna(axis=0, how='all')
targets = targets.loc[cls].dropna(axis=0, how='all')

dependency_data = Dataset('dependency', np.expand_dims(features.as_matrix(), axis=2), targets.as_matrix(), val_size=0.1, test_size=0.1, stratify=True)

dataset_params = { \

	'dataset': dependency_data \

}

graph_params = { \

	'G': dep_graph, \
	'depth': 1, \
	'edge_weight_fun': gene_edge_fun \

}

mlp_params = { \

	'batch_size': 25, \
	'n_node_features': 1, \
	'n_target_features': dependency_data.train_y.shape[1], \
	'signal_time': 64, \
	'index_hidden': 32, \
	'n_layers': 1, \
	'filter_shape': [[16, 8, 1]], \
	'mlp_dims': [[8*dependency_data.train_x.shape[1], 64], [64, dependency_data.train_y.shape[1]]], \
	'act_fun': [tf.nn.elu] \

}

def dep_loss(true, pred):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=true, logits=pred))

def optimizer(lr, m=0):
	#return tf.train.MomentumOptimizer(lr, m, use_nesterov=False)
	return tf.train.AdamOptimizer(lr)

learning_params = { \

	'cost_function': dep_loss, \
	'optimizer': optimizer, \
	'learning_rate': 2.5e-5
	
}

gNet = GraphNetwork('dependency', dataset_params, graph_params, mlp_params, learning_params, orientation='graph')

#plt.ion()
cost = []
val_cost = []
iter_arr = []
i = 0
test_auc = 0
val_auc = 0
while gNet.dataset_params['dataset'].epoch < 100:
	gNet.run('train')
	tr_cost = gNet.eval(gNet.cost)
	if np.sum(np.round(gNet.eval(gNet.Yph))) > 0 and np.sum(np.round(gNet.eval(gNet.Yph))) < gNet.eval(gNet.Yph).shape[0]:
		tr_auc = roc_auc_score(np.round(gNet.eval(gNet.Yph)), gNet.eval(tf.nn.sigmoid(gNet.prediction)))
	else:
		tr_auc = np.nan
	val_auc = roc_auc_score(np.round(dependency_data.val_y), sigmoid(gNet.predict(dependency_data.val_ids)))
	test_auc = roc_auc_score(np.round(dependency_data.test_y), sigmoid(gNet.predict(dependency_data.test_ids)))	
	iter_arr.append(i)
	cost.append(tr_cost)
	val_cost.append(dep_loss(tf.constant(dependency_data.val_y, dtype=tf.float32), tf.constant(gNet.predict(dependency_data.val_ids), dtype=tf.float32)).eval(session=gNet.session))
	if i%25 == 0:
		print("AUC is: " + str(np.mean(np.array(tr_auc))))
		print("Validation AUC is: " + str(roc_auc_score(np.round(dependency_data.val_y), sigmoid(gNet.predict(dependency_data.val_ids)))))
		print("Test AUC is: " + str(roc_auc_score(np.round(dependency_data.test_y), sigmoid(gNet.predict(dependency_data.test_ids)))))
		print("Epoch is: " + str(gNet.dataset_params['dataset'].epoch))
		#plt.clf()
		#plt.scatter(sigmoid(gNet.predict(dependency_data.val_ids)), dependency_data.val_y)
		#plt.pause(1.0)
		#plt.clf()
		#plt.plot(iter_arr, cost)
		#plt.plot(iter_arr, val_cost)
		#plt.pause(0.001)
	i += 1

tr_auc = roc_auc_score(np.round(dependency_data.train_y), sigmoid(gNet.predict(dependency_data.train_ids)))
predictive_auc = roc_auc_score(np.concatenate([np.round(dependency_data.val_y), np.round(dependency_data.test_y)], axis=0), sigmoid(gNet.predict(np.concatenate([dependency_data.val_ids, dependency_data.test_ids], axis=0))))
outfile = open('results.txt', 'w')
outlist = [sys.argv[1], tr_auc, val_auc, test_auc, predictive_auc, gNet.dataset_params['dataset'].epoch]
for item in outlist:
  outfile.write("%s\n" % item)
outfile.close()

for i in range(8):
	np.savetxt("/Users/jbryan/Desktop/poster_figures/dep_activations_" + str(i) + ".txt", gNet.activate(dependency_data.sample_ids)[:,:,i], delimiter="\t")
np.savetxt("/Users/jbryan/Desktop/poster_figures/dep_labels.txt", dependency_data.targets, delimiter="\t")