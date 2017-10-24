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
from pygcnn.backbone import *

gene_sets = list(dict_from_msigdb("data/msig/all.v6.1.symbols.gmt.txt", max_size=25).values())
genes = set(np.concatenate(gene_sets, axis=0).tolist())

gene2int = {}
int2gene = {}
n_genes = len(genes)
batch_size = 256
num_sampled = 128

for i, gene in enumerate(genes):
	gene2int[gene] = i
	int2gene[i] = gene

train_data = []
for gene_set in gene_sets:
	for gene_index in range(len(gene_set)):
		for partner_index in range(len(gene_set)):
			if gene_index != partner_index:
				train_data.append([gene2int[gene_set[gene_index]], gene2int[gene_set[partner_index]]])

train_data = np.stack(train_data, axis=0)

train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
query_genes_ph = tf.placeholder(tf.int32)

embedding_size = 512
embeddings = tf.Variable(tf.random_uniform([n_genes, embedding_size], -1.0, 1.0))
embedder = MLP(name='embed', dims=[[n_genes, embedding_size], [embedding_size, n_genes]], output_fun=tf.identity, dropout=[0, 0])
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
query_embeddings = tf.nn.embedding_lookup(normalized_embeddings, query_genes_ph)
similarity = tf.matmul(query_embeddings, normalized_embeddings, transpose_b=True)

loss = tf.reduce_mean(tf.nn.nce_loss(weights=embedder.layers[0].weights, biases=embedder.layers[1].bias, labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=n_genes))

update = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
indices = np.arange(len(train_data))
np.random.shuffle(indices)
epoch = 1
for i in range(20000):
	batch_indices = np.arange(i*batch_size, (i+1)*batch_size)
	if (i+1)*batch_size >= epoch*len(train_data):
		print "Concluding epoch " + str(epoch) + "..."
		np.random.shuffle(indices)
		epoch = epoch + 1
	batch_indices = np.array([(k%len(train_data)) for k in batch_indices])
	batch_indices = indices[batch_indices]
	batch = train_data[batch_indices, :]
	sess.run(update, feed_dict={train_inputs: batch[:,0], train_labels: np.reshape(batch[:,1], [batch_size, 1])})
	if i % 200 == 0:
		print(sess.run(loss, feed_dict={train_inputs: batch[:,0], train_labels: np.reshape(batch[:,1], [batch_size, 1])}))
		print "Most similar genes to BRAF, PDE3A, MDM2 are: "
		find_similar_genes(similarity, ['BRAF', 'PDE3A', 'MDM2'], query_genes_ph, sess, gene2int, int2gene)

dep_graph = nearest_neighbors(normalized_embeddings, sess, 4, int2gene)
dep_graph = disparity_filter(dep_graph)
dep_graph = disparity_filter_alpha_cut(dep_graph)
nx.write_gpickle(dep_graph, "data/pickles/gene_embedding.gpickle")

