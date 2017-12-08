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
import sys
import datetime

from pygcnn.utils import *
from pygcnn.indexing import *
from pygcnn.layers import *
from pygcnn.backbone import *

# Load graph
dep_graph = parse_gene_graph("data/reactome_functional_interactions/FIsInGene_022717_with_annotations.txt")
hallmark_genes = pd.read_csv("data/cancer_gene_census/Census_allThu_Dec_7_14_32_15_2017.csv")
hallmark_genes = np.array(hallmark_genes['Gene Symbol']).tolist()

# Load training + test data
client = Competition()
taiga_client = TaigaClient()

features = taiga_client.get(name='rnaseq-gene-expression-5362', file='RNAseq_CCLE_RSEM_TPM', version='6').transpose().rename(lambda x: x.split(" (", 1)[0]).transpose()
targets = taiga_client.get(name='avana-1-2-8b72', version=2, file='gene_dependency').transpose().rename(lambda x: x.split(" (", 1)[0]).transpose()

cls = list(set(list(features.transpose())) & set(list(targets.transpose())))

feature_genes = list(set(dep_graph.nodes()) & set(list(features)) & set(hallmark_genes))
#target_genes = list(set(client.get_genes()) & set(list(targets)))
target_genes = [sys.argv[1]]

features = features[feature_genes]
targets = targets[target_genes]
gene2int = {list(features)[i]: i for i in range(features.shape[1])}
int2gene = {i: list(features)[i] for i in range(features.shape[1])}
dep_graph = nx.relabel_nodes(dep_graph.subgraph(list(features)), gene2int)

features = features.loc[cls].dropna(axis=0, how='all')
targets = targets.loc[cls].dropna(axis=0, how='all')

dependency_data = Dataset('dependency', np.expand_dims(features.as_matrix(), axis=2), np.round(targets.as_matrix()), test_size=0.1, stratify=True)

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
	'signal_time': 16, \
	'index_hidden': 32, \
	'n_layers': 1, \
	'filter_shape': [[25, 32, 1]], \
	'mlp_dims': [[32*dependency_data.train_x.shape[1], 256], [256, dependency_data.train_y.shape[1]]], \
	'act_fun': [tf.nn.elu] \

}

def dep_loss(true, pred):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=true, logits=pred))

learning_params = { \

	'cost_function': dep_loss, \
	'optimizer': tf.train.AdamOptimizer, \
	'learning_rate': 1e-6
	
}

gNet = GraphNetwork('dependency', dataset_params, graph_params, mlp_params, learning_params, orientation='graph')

#plt.ion()
cost = []
test_cost = []
iter_arr = []
i = 0
test_auc = 0
improvement_window = 10
curr_epoch = 0
while True:
	gNet.run('train')
	if gNet.dataset_params['dataset'].epoch % improvement_window == 0 and gNet.dataset_params['dataset'].epoch >= curr_epoch:
		if roc_auc_score(dependency_data.test_y, sigmoid(gNet.predict(dependency_data.test_ids))) <= test_auc:
			break
		else:
			test_auc = roc_auc_score(dependency_data.test_y, sigmoid(gNet.predict(dependency_data.test_ids)))
		curr_epoch = curr_epoch + improvement_window
	if i%50 == 0:
		tr_cost = gNet.eval(gNet.cost)
		tr_auc = []
		labels = gNet.eval(gNet.Yph)
		preds = gNet.eval(tf.nn.sigmoid(gNet.prediction))
		for j in range(dependency_data.train_y.shape[1]):
			if np.sum(labels[:,j]) == 0 or np.sum(labels[:,j]) == labels.shape[0]:
				pass
			else:
				auc = roc_auc_score(labels[:,j], preds[:,j])
				tr_auc.append(auc)
		cost.append(np.mean(np.array(tr_auc)))
		print "AUC is: " + str(np.mean(np.array(tr_auc)))
		print "Test AUC is: " + str(roc_auc_score(dependency_data.test_y, sigmoid(gNet.predict(dependency_data.test_ids))))
		print "Epoch is: " + str(gNet.dataset_params['dataset'].epoch)
		#plt.clf()
		#plt.scatter(preds, labels)
		#plt.pause(0.001)
	i += 1

outfile = open('results/dep/' + sys.argv[1] + '_' + '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + '.txt', 'w')
outlist = [sys.argv[1], test_auc]
for item in outlist:
  outfile.write("%s\n" % item)
outfile.close()
