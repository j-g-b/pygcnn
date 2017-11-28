import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
import community as cm

from pygcnn.utils import *
from pygcnn.indexing import *

class Dataset(object):
	def __init__(self, name, features, targets, test_size=None, split=None):
		assert features.shape[0] == targets.shape[0]
		self.name = name
		self.features = features
		self.targets = targets
		self.test_size = test_size
		self.sample_ids = np.reshape(np.arange(features.shape[0]), (-1,1))
		if split is None:
			if self.test_size is not None:
				self.train_x, self.test_x, self.train_y, self.test_y, self.train_ids, self.test_ids = train_test_split(features, targets, self.sample_ids, test_size=test_size)
			else:
				self.train_x, self.test_x, self.train_y, self.test_y, self.train_ids, self.test_ids = features, np.copy(features), targets, np.copy(targets), np.copy(self.sample_ids), np.copy(self.sample_ids)
		else:
			self.train_x = np.take(features, split[0], axis=0)
			self.train_y = np.take(targets, split[0], axis=0)
			self.test_x = np.take(features, split[1], axis=0)
			self.test_y = np.take(targets, split[1], axis=0)
			self.train_ids = split[0]
			self.test_ids = split[1]
		self.train_batch_indx = 0
		self.train_indices = np.random.permutation(self.train_x.shape[0])
		self.test_batch_indx = 0
		self.test_indices = np.random.permutation(self.test_x.shape[0])
	def next_batch(self, batch_size):
		# returns tuple containing:
		# 	0: batch of X data with `batch_size` rows
		#	1: batch of Y data with `batch_size` rows
		#	2: indices of self.features and self.targets contained within batch
		next_x_batch = []
		next_y_batch = []
		next_id_batch = []
		for indx in range(self.train_batch_indx, self.train_batch_indx + batch_size):
			if indx == self.train_x.shape[0]:
				self.train_indices = np.random.permutation(self.train_x.shape[0])
			next_x_batch.append(self.train_x[self.train_indices[indx % self.train_x.shape[0]], :])
			next_y_batch.append(self.train_y[self.train_indices[indx % self.train_x.shape[0]], :])
			next_id_batch.append(self.train_ids[self.train_indices[indx % self.train_x.shape[0]]])
		self.train_batch_indx = (self.train_batch_indx + batch_size) % self.train_x.shape[0]
		next_batch = (np.stack(next_x_batch, axis=0), np.stack(next_y_batch, axis=0), np.stack(next_id_batch, axis=0))
		return next_batch
	def next_test_batch(self, batch_size):
		# returns tuple containing:
		# 	0: batch of X data with `batch_size` rows
		#	1: batch of Y data with `batch_size` rows
		#	2: indices of self.features and self.targets contained within batch
		next_x_batch = []
		next_y_batch = []
		next_id_batch = []
		for indx in range(self.test_batch_indx, self.test_batch_indx + batch_size):
			if indx == self.test_x.shape[0]:
				self.test_indices = np.random.permutation(self.test_x.shape[0])
			next_x_batch.append(self.test_x[self.test_indices[indx % self.test_x.shape[0]], :])
			next_y_batch.append(self.test_y[self.test_indices[indx % self.test_x.shape[0]], :])
			next_id_batch.append(self.test_ids[self.test_indices[indx % self.test_x.shape[0]]])
		self.test_batch_indx = (self.test_batch_indx + batch_size) % self.test_x.shape[0]
		next_batch = (np.stack(next_x_batch, axis=0), np.stack(next_y_batch, axis=0), np.stack(next_id_batch, axis=0))
		return next_batch

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
	def __init__(self, name, filter_shape, n_layers=1, act_fun=[tf.nn.elu], index_dims=[20, 10], dropout=[0.5]):	
		# filter shape should be (filter_size, n_filter_output_features, n_node_features)
		self.name = name
		self.n_layers = n_layers
		self.act_fun = act_fun
		self.index_dims = index_dims
		self.filter_shape = []
		self.filter_weights = []
		self.filter_bias = []
		self.indexing_mlp = []
		self.dropout = dropout
		self.reduce_weights = []
		# Set up convolutional layers with induced graphs
		for i in range(n_layers):
			self.indexing_mlp.append(MLP(name=name + '_indexing_mlp_' + str(i), dims=[[index_dims[0], index_dims[1]], [index_dims[1], filter_shape[i][0]]], act_fun=[tf.nn.elu], dropout=[0,0], output_fun=tf.nn.softmax))
			self.filter_shape.append(filter_shape[i])
			self.filter_weights.append(tf_init_weights(shape=[(filter_shape[i][0]*filter_shape[i][2]), filter_shape[i][1]]))
			self.filter_bias.append(tf_init_bias(shape=[filter_shape[i][1]]))
	def __call__(self, inputs, n_nodes, subgraph_features, neighborhoods, neighborhood_sizes, graph_feature_padder):
		# assumes that all input matrices are Tensorflow Tensors
		self.activation = []
		X = tf.transpose(inputs, perm=[1,2,0])
		for i in range(self.n_layers):
			indexer_output = self.indexing_mlp[i](subgraph_features[i])
			indexer_output = tf.gather(tf.concat([indexer_output, tf.zeros([1, indexer_output.get_shape().as_list()[1]])], axis=0), graph_feature_padder[i])
			indexer_output = tf.reshape(indexer_output, shape=[n_nodes[i], max(neighborhood_sizes[i]), -1])
			X_neighborhood = tf.gather(tf.concat([X, tf.zeros([1, X.get_shape().as_list()[1], X.get_shape().as_list()[2]])], axis=0), neighborhoods[i])
			X_neighborhood = tf.reshape(X_neighborhood, shape=[n_nodes[i], max(neighborhood_sizes[i]), self.filter_shape[i][2], -1])
			indexed_features = tf.matmul(tf.tile(tf.expand_dims(tf.transpose(indexer_output, perm=[0,2,1]), 0), [self.filter_shape[i][2], 1, 1, 1]), tf.transpose(X_neighborhood, perm=[2,0,1,3]))
			convolved_signal = self.act_fun[i](tf.matmul(tf.nn.dropout(tf.reshape(tf.transpose(indexed_features, perm=[3,1,2,0]), shape=[X.get_shape().as_list()[2], -1, self.filter_shape[i][0]*self.filter_shape[i][2]]), 1 - self.dropout[i]), tf.tile(tf.expand_dims(tf.transpose(tf.reshape(self.filter_weights[i], shape=[self.filter_shape[i][1], -1])), 0), [X.get_shape().as_list()[2], 1, 1])) + self.filter_bias[i])
			self.activation.append(convolved_signal)
			convolved_signal = tf.transpose(convolved_signal, perm=[1,2,0])
			X = convolved_signal
		return tf.transpose(X, perm=[2,0,1])

class taN(object):
	def __init__(self, G, GC, depth=2):
		self.G = [G]
		self.GC = GC
		self.n_layers = GC.n_layers
		self.subgraph_features = []
		self.n_nodes = []
		self.neighborhood_sizes = []
		self.neighborhoods = []
		self.graph_feature_padder = []
		self.edge_weights = [np.array(depth*[0.9]) for i in range(self.n_layers)]
		self.n_timepoints = GC.index_dims[0]
		self.output = None
		for i in range(self.n_layers):
			subgraph_features, neighborhoods = index_graph(self.G[i], self.edge_weights[i], self.n_timepoints)
			n_nodes = len(neighborhoods)
			neighborhood_sizes = [len(neighborhoods[k]) for k in range(len(neighborhoods))]
			self.subgraph_features.append(subgraph_features)
			self.n_nodes.append(n_nodes)
			self.neighborhood_sizes.append(neighborhood_sizes)
			self.neighborhoods.append(np.stack([neighborhoods[k] + [n_nodes]*(max(neighborhood_sizes)-neighborhood_sizes[k]) for k in range(len(neighborhoods))], axis=0))
			self.graph_feature_padder.append(np.stack([range(sum(neighborhood_sizes[0:k]), sum(neighborhood_sizes[0:(k+1)])) + [np.concatenate(subgraph_features, axis=0).shape[0]]*(max(neighborhood_sizes)-neighborhood_sizes[k]) for k in range(len(neighborhoods))], axis=0))
			self.edge_weights.append(self.edge_weights[i])
	def __call__(self, inputs, node_batch_size):
		if node_batch_size is None:
			# First need to flatten everything
			self.output = self.GC(inputs, \
									self.n_nodes, \
									[tf.constant(np.concatenate(self.subgraph_features[l], axis=0), dtype=tf.float32) for l in range(len(self.subgraph_features))], \
									[tf.constant(self.neighborhoods[l].flatten(), dtype=tf.int64) for l in range(len(self.neighborhoods))], \
									self.neighborhood_sizes, \
									[tf.constant(self.graph_feature_padder[l].flatten(), dtype=tf.int64) for l in range(len(self.graph_feature_padder))], \
									)
		else:
			self.NBph = [tf.placeholder("int64", shape=(node_batch_size*max(self.neighborhood_sizes[0])))]
			self.GFph = [tf.placeholder("int64", shape=(node_batch_size*max(self.neighborhood_sizes[0])))]
			self.SGph = [tf.placeholder("float32", shape=(node_batch_size*max(self.neighborhood_sizes[0]), np.concatenate(self.subgraph_features[0], axis=0).shape[1]))]
			self.output = self.GC(inputs, \
									[node_batch_size], \
									self.SGph, \
									self.NBph, \
									self.neighborhood_sizes, \
									self.GFph)

class GraphNetwork(object):
	def __init__(self, name, dataset_params, graph_params, mlp_params, learning_params, orientation='graph'):
		assert orientation in ['graph', 'node']
		self.dataset_params = dataset_params
		self.graph_params = graph_params
		self.mlp_params = mlp_params
		self.learning_params = learning_params
		self.orientation = orientation
		if orientation == 'graph':
			self.Xph = tf.placeholder("float32", shape=(mlp_params['batch_size'], len(graph_params['G'].nodes()), mlp_params['n_node_features']))
			self.Yph = tf.placeholder("float32", shape=(mlp_params['batch_size'], mlp_params['n_target_features']))
			self.node_batch_size = None
			self.dropout = tf.placeholder(tf.float32)
			self.filter_dropout = tf.placeholder(tf.float32)
		if orientation == 'node':
			self.Xph = tf.placeholder("float32", shape=(1, None, mlp_params['n_node_features']))
			self.Yph = tf.placeholder("float32", shape=(mlp_params['batch_size'], mlp_params['n_target_features']))
			self.dropout = tf.placeholder(tf.float32)
			self.filter_dropout = tf.placeholder(tf.float32)
			self.node_batch_size = mlp_params['batch_size']
		print "Initializing graph convolution..."
		self.GC = GraphConvolution(name=name + '_gc', filter_shape=mlp_params['filter_shape'], n_layers=mlp_params['n_layers'], act_fun=mlp_params['act_fun'], index_dims=[mlp_params['signal_time'], mlp_params['index_hidden']], dropout=[self.filter_dropout])
		print "Initializing taN..."
		self.TAN = taN(G=graph_params['G'], GC=self.GC, depth=graph_params['depth'])
		print "Running taN..."
		self.TAN(self.Xph, self.node_batch_size)
		print "Initializing readout layer..."
		if orientation == 'graph':
			mlp_input_size = self.TAN.output.get_shape().as_list()[1]*self.TAN.output.get_shape().as_list()[2]
		else:
			mlp_input_size = self.TAN.output.get_shape().as_list()[0]*self.TAN.output.get_shape().as_list()[2]
		self.MLP = MLP(name=name + '_mlp', dims=mlp_params['mlp_dims'], output_fun=tf.identity, dropout=len(mlp_params['mlp_dims'])*[self.dropout])
		print "Setting output..."
		if orientation == 'graph':
			output = tf.reshape(self.TAN.output, shape=[self.TAN.output.get_shape().as_list()[0], -1])
		else:
			output = tf.reshape(self.TAN.output, shape=[mlp_params['batch_size'], -1])
		self.prediction = self.MLP(output)
		print "Initializing optimizer..."
		self.cost = learning_params['cost_function'](self.Yph, self.prediction)
		if 'l2_lambda' in learning_params.keys():
			for i in range(self.TAN.n_layers):
				self.cost += learning_params['l2_lambda']*tf.nn.l2_loss(self.TAN.GC.filter_weights[i])
				for j in range(len(self.TAN.GC.indexing_mlp[i].dims)):
					self.cost += learning_params['l2_lambda']*tf.nn.l2_loss(self.TAN.GC.indexing_mlp[i].layers[j].weights)
			for i in range(len(self.MLP.dims)):
				self.cost += learning_params['l2_lambda']*tf.nn.l2_loss(self.MLP.layers[i].weights)
		self.update = learning_params['optimizer'](learning_params['learning_rate']).minimize(self.cost)
		print "Initializing Tensorflow session and variables..."
		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())
	def run(self, mode='train'):
		assert mode in ['train', 'test']
		if mode == 'train':
			batch = self.dataset_params['dataset'].next_batch(self.mlp_params['batch_size'])
		else:
			batch = self.dataset_params['dataset'].next_test_batch(self.mlp_params['batch_size'])
		if self.orientation == 'node':
			focal_node_batch = batch[2].flatten()
			node_batch = np.unique(np.take(self.TAN.neighborhoods[0], focal_node_batch, axis=0).flatten())
			node_batch = np.delete(node_batch, np.where(node_batch == self.TAN.n_nodes[0]))
			X = np.expand_dims(np.take(self.dataset_params['dataset'].features, node_batch, axis=0), 0)
			translate_nb_dict = {node_batch[i]: i for i in range(X.shape[1])}
			translate_nb_dict[self.TAN.n_nodes[0]] = self.mlp_params['batch_size']
			max_neighborhood_size = max(self.TAN.neighborhood_sizes[0])
			neighborhood_sizes = np.take(self.TAN.neighborhood_sizes[0], focal_node_batch)
			subgraph_features = np.take(self.TAN.subgraph_features[0], focal_node_batch)
			subgraph_feature_zero_pad = [np.zeros((max_neighborhood_size - neighborhood_sizes[k], subgraph_features[0].shape[1])) for k in range(self.mlp_params['batch_size'])]
			subgraph_features = np.concatenate([np.concatenate([subgraph_features[k], subgraph_feature_zero_pad[k]], axis=0) for k in range(self.mlp_params['batch_size'])], axis=0)
			graph_feature_padder = np.stack([range(k*max_neighborhood_size, k*max_neighborhood_size + neighborhood_sizes[k]) + [subgraph_features.shape[0]]*(max_neighborhood_size-neighborhood_sizes[k]) for k in range(self.mlp_params['batch_size'])], axis=0)
			if mode == 'train':
				self.session.run(self.update, \
								feed_dict={self.Xph: X, \
									 		self.Yph: batch[1], \
								 			self.TAN.NBph[0]: translate_array(np.take(self.TAN.neighborhoods[0], focal_node_batch, axis=0).flatten(), translate_nb_dict), \
								 			self.TAN.GFph[0]: graph_feature_padder.flatten(), \
								 			self.TAN.SGph[0]: subgraph_features, \
								 			self.dropout: 0.5,
								 			self.filter_dropout: 0.5})
			iter_cost, iter_acc = self.session.run([self.cost, tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prediction, axis=1), tf.argmax(self.Yph, axis=1)), tf.float32))], \
							feed_dict={self.Xph: X, \
								 		self.Yph: batch[1], \
								 		self.TAN.NBph[0]: translate_array(np.take(self.TAN.neighborhoods[0], focal_node_batch, axis=0).flatten(), translate_nb_dict), \
								 		self.TAN.GFph[0]: graph_feature_padder.flatten(), \
								 		self.TAN.SGph[0]: subgraph_features, \
								 		self.dropout: 0, \
								 		self.filter_dropout: 0})
		else:
			if mode == 'train':
				self.session.run(self.update, \
								feed_dict={self.Xph: batch[0], \
								 			self.Yph: batch[1], \
								 			self.dropout: 0,
								 			self.filter_dropout: 0})
			iter_cost, iter_acc = self.session.run([self.cost, tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prediction, axis=1), tf.argmax(self.Yph, axis=1)), tf.float32))], \
							feed_dict={self.Xph: batch[0], \
								 		self.Yph: batch[1], \
								 		self.dropout: 0, \
								 		self.filter_dropout: 0})
		return iter_cost, iter_acc











