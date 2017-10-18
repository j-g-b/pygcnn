import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
import community as cm

from pygcnn.utils import *
from pygcnn.indexing import *

class Dataset(object):
	def __init__(self, name, features, targets, test_size):
		assert features.shape[0] == targets.shape[0]
		self.name = name
		self.features = features
		self.targets = targets
		self.test_size = test_size
		self.sample_ids = np.reshape(np.arange(features.shape[0]), (-1,1))
		if self.test_size != 0:
			self.train_x, self.test_x, self.train_y, self.test_y, self.train_ids, self.test_ids = train_test_split(features, targets, self.sample_ids, test_size=test_size)
		else:
			self.train_x, self.test_x, self.train_y, self.test_y, self.train_ids, self.test_ids = features, np.copy(features), targets, np.copy(targets), np.copy(self.sample_ids), np.copy(self.sample_ids)
		self.batch_indx = 0
		self.test_batch_indx = 0
	def next_batch(self, batch_size):
		next_x_batch = []
		next_y_batch = []
		next_id_batch = []
		for indx in range(self.batch_indx, self.batch_indx + batch_size):
			if indx == self.train_x.shape[0]:
				new_indices = np.random.permutation(self.train_x.shape[0])
				self.train_x = self.train_x[new_indices, :]
				self.train_y = self.train_y[new_indices, :]
				self.train_ids = self.train_ids[new_indices, :]
			next_x_batch.append(self.train_x[indx % self.train_x.shape[0], :])
			next_y_batch.append(self.train_y[indx % self.train_x.shape[0], :])
			next_id_batch.append(self.train_ids[indx % self.train_x.shape[0], :])
		self.batch_indx += batch_size
		self.batch_indx = self.batch_indx % self.train_x.shape[0]
		next_batch = (np.stack(next_x_batch, axis=0), np.stack(next_y_batch, axis=0), np.stack(next_id_batch, axis=0))
		return next_batch
	def next_test_batch(self, batch_size):
		next_x_batch = []
		next_y_batch = []
		next_id_batch = []
		for indx in range(self.test_batch_indx, self.test_batch_indx + batch_size):
			if indx == self.test_x.shape[0]:
				new_indices = np.random.permutation(self.test_x.shape[0])
				self.test_x = self.test_x[new_indices, :]
				self.test_y = self.test_y[new_indices, :]
				self.test_ids = self.test_ids[new_indices, :]
			next_x_batch.append(self.test_x[indx % self.test_x.shape[0], :])
			next_y_batch.append(self.test_y[indx % self.test_x.shape[0], :])
			next_id_batch.append(self.test_ids[indx % self.test_x.shape[0], :])
		self.test_batch_indx += batch_size
		self.test_batch_indx = self.test_batch_indx % self.test_x.shape[0]
		next_batch = (np.stack(next_x_batch, axis=0), np.stack(next_y_batch, axis=0), np.stack(next_id_batch, axis=0))
		return next_batch

class DataDict(object):
	def __init__(self, name, sample_ids):
		self.name = name
		self.sample_ids = sample_ids
		self.sample_dict = dict.fromkeys(sample_ids)
		self.feature_type_dict = {}
		for sample_id in self.sample_dict.keys():
			self.sample_dict[sample_id] = {}
	def add_df(self, feature_type, df):
		self.feature_type_dict[feature_type] = {'names': list(df)}
		common_samples = list(set(list(df.index)) & set(self.sample_ids))
		for sample_id in common_samples:
			self.sample_dict[sample_id][feature_type] = np.array(df.loc[sample_id])
		self.sample_ids = common_samples
		self.sample_dict = {sample_id: self.sample_dict[sample_id] for sample_id in common_samples}
	def gather_samples(self, sample_ids, feature_type):
		return {sample_id: self.sample_dict[sample_id][feature_type] for sample_id in sample_ids}
	def train_test_split(self, test_size):
		indices = np.arange(len(self.sample_ids))
		np.random.shuffle(indices)
		sample_scramble = [self.sample_ids[i] for i in indices.tolist()]
		self.train_ids = sample_scramble[0:int(np.floor((1.0 - test_size)*len(sample_scramble)))]
		self.test_ids = sample_scramble[(int(np.floor((1.0 - test_size)*len(sample_scramble)))):]
	def gather_features(self, feature_types, sample_ids=None):
		features = []
		if not sample_ids:
			sample_ids = self.sample_ids
		for feature_type in feature_types:
			sample_features = self.gather_samples(sample_ids, feature_type)
			features.append(np.stack([sample_features[sample_id] for sample_id in sample_ids]))
		return features
	def combine_feature_types(self, feature_types, sample_ids=None):
		if not sample_ids:
			sample_ids = self.sample_ids
		if len(feature_types) == 1:
			return self.gather_features(feature_types)[0]
		else:
			features = []
			common_features = self.feature_type_dict[feature_types[0]]['names']
			for feature_type in feature_types:
				common_features = list(set(common_features) & set(self.feature_type_dict[feature_type]['names']))
			for feature_type in feature_types:
				name_index_dict = {self.feature_type_dict[feature_type]['names'][i]: i for i in range(len(self.feature_type_dict[feature_type]['names']))}
				f = self.gather_features([feature_type], sample_ids)[0]
				features.append(np.stack([f[:, name_index_dict[name]] for name in common_features], axis=-1))
			return np.stack(features, axis=-1)

class Layer(object):
	def __init__(self, name, input_dim, output_dim, act_fun=tf.nn.elu, dropout=0, mask=None):
		self.name = name
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.act_fun = act_fun
		self.dropout = dropout
		self.weights = tf_init_weights(shape=[input_dim, output_dim])
		self.bias = tf_init_bias(shape=[output_dim])
		self.mask = None

class MLP(object):
	def __init__(self, name, dims=[[10,10],[10,10]], act_fun=[tf.nn.elu], dropout=[0,0], output_fun=tf.nn.softmax, mask=[None,None]):
		self.name = name
		self.dims = dims
		self.act_fun = act_fun
		self.dropout = dropout
		self.output_fun = output_fun
		self.act_fun.append(output_fun)
		self.mask = mask
		self.layers = [Layer(name=self.name + '_layer_' + str(i), input_dim=dims[i][0], output_dim=dims[i][1], act_fun=self.act_fun[i], dropout=self.dropout[i], mask=self.mask[i]) for i in range(len(dims))]
	def __call__(self, inputs):
		output = inputs
		for layer in self.layers:
			if layer.mask is None:
				output = layer.act_fun(tf.matmul(tf.nn.dropout(output, 1 - layer.dropout), layer.weights) + layer.bias)
			else:
				output = layer.act_fun(tf.matmul(tf.nn.dropout(output, 1 - layer.dropout), tf.multiply(layer.mask, layer.weights), b_is_sparse=tf.reduce_mean(layer.mask) < 0.1) + layer.bias)
		return output

class GraphConvolution(object):
	def __init__(self, name, G, filter_shape, n_layers=2, n_timepoints=20, edge_weights=[0.9, 0.9], act_fun=tf.nn.elu, index_hidden_layer_size=10, partition_resolution=0.5):
		# filter shape should be (filter_size, n_filter_output_features, n_node_features)
		self.name = name
		self.n_layers = n_layers
		self.edge_weights = edge_weights
		self.act_fun = act_fun
		self.n_timepoints = n_timepoints
		self.G = [G]
		self.partition = []
		self.subgraph_features = []
		self.n_nodes = []
		self.neighborhood_sizes = []
		self.neighborhoods = []
		self.partition_sizes = []
		self.graph_feature_padder = []
		self.filter_shape = []
		self.filter_weights = []
		self.filter_bias = []
		self.indexing_mlp = []
		# Set up convolutional layers with induced graphs
		for i in range(n_layers):
			best_partition = cm.best_partition(self.G[i], resolution=partition_resolution)
			partition_dict = {val: [] for val in list(set(best_partition.values()))}
			for j in range(len(best_partition.keys())):
				partition_dict[best_partition.values()[j]].append(best_partition.keys()[j])
			partition_arr = list(partition_dict.values())
			self.partition_sizes.append([len(p) for p in partition_arr])
			max_comm_size = max([len(p) for p in partition_arr])
			for j in range(len(partition_arr)):
				pad = (max_comm_size - len(partition_arr[j]))*[len(self.G[i].nodes())]
				partition_arr[j] = partition_arr[j] + pad
			self.partition.append(np.concatenate(partition_arr, axis=0))
			induced_graph = cm.induced_graph(best_partition, self.G[i])
			self.G.append(induced_graph)
			self.indexing_mlp.append(MLP(name=name + '_indexing_mlp_' + str(i), dims=[[n_timepoints, index_hidden_layer_size], [index_hidden_layer_size, filter_shape[i][0]]], act_fun=[tf.nn.elu], dropout=[0,0], output_fun=tf.nn.softmax))
		for i in range(n_layers):
			subgraph_features, neighborhoods = index_graph(self.G[i], edge_weights, n_timepoints)
			n_nodes = len(neighborhoods)
			neighborhood_sizes = [len(neighborhoods[k]) for k in range(len(neighborhoods))]
			self.subgraph_features.append(np.concatenate(subgraph_features, axis=0))
			self.n_nodes.append(n_nodes)
			self.neighborhood_sizes.append(neighborhood_sizes)
			self.neighborhoods.append(np.concatenate([neighborhoods[k] + [n_nodes]*(max(neighborhood_sizes)-neighborhood_sizes[k]) for k in range(len(neighborhoods))], axis=0))
			self.graph_feature_padder.append(np.concatenate([range(sum(neighborhood_sizes[0:k]), sum(neighborhood_sizes[0:(k+1)])) + [np.concatenate(subgraph_features, axis=0).shape[0]]*(max(neighborhood_sizes)-neighborhood_sizes[k]) for k in range(len(neighborhoods))], axis=0))
			self.filter_shape.append(filter_shape[i])
			self.filter_weights.append(tf_init_weights(shape=filter_shape[i]))
			self.filter_bias.append(tf_init_bias(shape=[filter_shape[i][1]]))
	def __call__(self, inputs):
		X = tf.transpose(inputs, perm=[1,2,0])
		for i in range(self.n_layers):
			indexer_output = self.indexing_mlp[i](tf.constant(self.subgraph_features[i], dtype=tf.float32))
			indexer_output = tf.gather(tf.concat([indexer_output, tf.zeros([1, indexer_output.get_shape().as_list()[1]])], axis=0), tf.constant(self.graph_feature_padder[i], dtype=tf.int64))
			indexer_output = tf.reshape(indexer_output, shape=[self.n_nodes[i], max(self.neighborhood_sizes[i]), -1])
			X_neighborhood = tf.gather(tf.concat([X, tf.zeros([1, X.get_shape().as_list()[1], X.get_shape().as_list()[2]])], axis=0), tf.constant(self.neighborhoods[i], dtype=tf.int64))
			X_neighborhood = tf.reshape(X_neighborhood, shape=[self.n_nodes[i], max(self.neighborhood_sizes[i]), self.filter_shape[i][2], -1])
			indexed_features = tf.matmul(tf.tile(tf.expand_dims(tf.transpose(indexer_output, perm=[0,2,1]), 0), [self.filter_shape[i][2], 1, 1, 1]), tf.transpose(X_neighborhood, perm=[2,0,1,3]))
			convolved_signal = tf.matmul(tf.reshape(tf.transpose(indexed_features, perm=[3,1,2,0]), shape=[X.get_shape().as_list()[2], -1, self.filter_shape[i][0]*self.filter_shape[i][2]]), tf.tile(tf.expand_dims(tf.transpose(tf.reshape(self.filter_weights[i], shape=[self.filter_shape[i][1], -1])), 0), [X.get_shape().as_list()[2], 1, 1]))
			convolved_signal = tf.transpose(convolved_signal, perm=[1,2,0])
			convolved_signal = tf.gather(tf.concat([convolved_signal, -np.inf*tf.ones([1, convolved_signal.get_shape().as_list()[1], convolved_signal.get_shape().as_list()[2]])], axis=0), tf.constant(self.partition[i], dtype=tf.int64))
			X = tf.reduce_max(tf.reshape(convolved_signal, shape=[len(self.partition_sizes[i]), max(self.partition_sizes[i]), convolved_signal.get_shape().as_list()[1], convolved_signal.get_shape().as_list()[2]]), axis=1)
		return tf.transpose(X, perm=[2,0,1])

