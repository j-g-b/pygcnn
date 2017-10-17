import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import datetime

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
	def __init__(self, name, subgraph_features, neighborhoods, filter_shape, n_timepoints=20, edge_weights=[0.9, 0.9], act_fun=tf.nn.elu, index_hidden_layer_size=10):
		# filter shape should be (filter_size, n_filter_output_features, n_node_features)
		self.name = name
		self.subgraph_features = np.concatenate(subgraph_features, axis=0)
		self.n_nodes = len(neighborhoods)
		self.neighborhood_sizes = [len(neighborhoods[i]) for i in range(len(neighborhoods))]
		self.neighborhoods = np.concatenate([neighborhoods[i] + [self.n_nodes]*(max(self.neighborhood_sizes)-self.neighborhood_sizes[i]) for i in range(len(neighborhoods))], axis=0)
		self.graph_feature_padder = np.concatenate([range(sum(self.neighborhood_sizes[0:i]), sum(self.neighborhood_sizes[0:(i+1)])) + [self.subgraph_features.shape[0]]*(max(self.neighborhood_sizes)-self.neighborhood_sizes[i]) for i in range(len(neighborhoods))], axis=0)
		self.filter_shape = filter_shape
		self.filter_weights = tf_init_weights(shape=filter_shape)
		self.filter_bias = tf_init_bias(shape=[filter_shape[1]])
		self.edge_weights = edge_weights
		self.act_fun = act_fun
		self.n_timepoints = n_timepoints
		self.indexing_mlp = MLP(name=name + '_indexing_mlp', dims=[[n_timepoints, index_hidden_layer_size], [index_hidden_layer_size, filter_shape[0]]], act_fun=[tf.nn.elu], dropout=[0,0], output_fun=tf.nn.softmax)
	def __call__(self, inputs):
		X = tf.transpose(inputs, perm=[1,2,0])
		indexer_output = self.indexing_mlp(tf.constant(self.subgraph_features, dtype=tf.float32))
		indexer_output = tf.gather(tf.concat([indexer_output, tf.zeros([max(self.neighborhood_sizes), indexer_output.get_shape().as_list()[1]])], axis=0), tf.constant(self.graph_feature_padder, dtype=tf.int64))
		indexer_output = tf.reshape(indexer_output, shape=[self.n_nodes, max(self.neighborhood_sizes), -1])
		X_neighborhood = tf.gather(tf.concat([X, tf.zeros([max(self.neighborhood_sizes), X.get_shape().as_list()[1], X.get_shape().as_list()[2]])], axis=0), tf.constant(self.neighborhoods, dtype=tf.int64))
		X_neighborhood = tf.reshape(X_neighborhood, shape=[self.n_nodes, max(self.neighborhood_sizes), self.filter_shape[2], -1])
		indexed_features = tf.matmul(tf.tile(tf.expand_dims(tf.transpose(indexer_output, perm=[0,2,1]), 0), [self.filter_shape[2], 1, 1, 1]), tf.transpose(X_neighborhood, perm=[2,0,1,3]))
		convolved_signal = tf.matmul(tf.reshape(tf.transpose(indexed_features, perm=[3,1,2,0]), shape=[X.get_shape().as_list()[2], -1, self.filter_shape[0]*self.filter_shape[2]]), tf.tile(tf.expand_dims(tf.transpose(tf.reshape(self.filter_weights, shape=[self.filter_shape[1], -1])), 0), [X.get_shape().as_list()[2], 1, 1]))
		return convolved_signal

