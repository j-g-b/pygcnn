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

cn_graph = nx.Graph()
for gene in feature_genes:
	cn_graph.add_node(gene2int[gene])

for chrom in cds_dict.keys():
	for i in range(len(cds_dict[chrom])-1):
		try:
			cn_graph.add_edge(gene2int[cds_dict[chrom][i]], gene2int[cds_dict[chrom][i+1]])
		except ValueError:
			pass

graph_params = { \

	'G': cn_graph, \
	'depth': 4 \

}

mlp_params = { \

	'batch_size': 100, \
	'n_node_features': 1, \
	'n_target_features': 1, \
	'signal_time': 16, \
	'index_hidden': 8, \
	'n_layers': 1, \
	'filter_shape': [[9, 32, 1]], \
	'mlp_hidden': 128, \
	'act_fun': [tf.nn.elu] \

}

gNet = GraphNetwork('CN', graph_params, mlp_params, orientation='node')

yhat = gNet.prediction
cost = tf.nn.l2_loss(gNet.Yph - yhat)
tss = tf.nn.l2_loss(gNet.Yph - tf.reduce_mean(gNet.Yph))
updates = tf.train.AdamOptimizer().minimize(cost)

print "Initializing variables..."
sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss = []
plt.ion()
print "Starting training..."
for epoch in range(50000):
	dep_batch = dependency.next_batch(1)
	dep_val_batch = dependency.next_test_batch(1)
	deps = dep_batch[0]
	node_batch = np.array(np.random.choice(range(1000), mlp_params['batch_size']), dtype="int64")
	sess.run(updates, feed_dict={gNet.Xph: deps, gNet.Yph: np.expand_dims(np.take(dep_batch[1], node_batch), 1), gNet.Nph: node_batch})
	if epoch % 100 == 0:
		acc = sess.run(cost, feed_dict={gNet.Xph: deps, gNet.Yph: np.expand_dims(np.take(dep_batch[1], node_batch), 1), gNet.Nph: node_batch})
		denom = sess.run(tss, feed_dict={gNet.Xph: deps, gNet.Yph: np.expand_dims(np.take(dep_batch[1], node_batch), 1), gNet.Nph: node_batch})
		print "R2: " + str(1.0 - ((acc) / denom))
