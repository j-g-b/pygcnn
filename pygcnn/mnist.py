from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from pygcnn.utils import *
from pygcnn.indexing import *
from pygcnn.layers import *

# Build mnist graph
pic_size = 28
mnist_graph = nx.grid_2d_graph(pic_size, pic_size)
coord_map = {mnist_graph.nodes()[i]: mnist_graph.nodes()[i][0]*pic_size + mnist_graph.nodes()[i][1] for i in range(len(mnist_graph.nodes()))}
mnist_graph = nx.relabel_nodes(mnist_graph, coord_map)

# for node in mnist_graph.nodes():
# 	tl = node - pic_size - 1
# 	tr = node - pic_size + 1
# 	bl = node + pic_size - 1
# 	br = node + pic_size + 1
# 	nodes_to_add = [node]
# 	if tl >= 0 and not node%pic_size == 0:
# 		nodes_to_add.append(tl)
# 	if tr >= 0 and not node%pic_size == (pic_size-1):
# 		nodes_to_add.append(tr)
# 	if bl < pic_size**2 and not node%pic_size == 0:
# 		nodes_to_add.append(bl)
# 	if br < pic_size**2 and not node%pic_size == (pic_size-1):
# 		nodes_to_add.append(br)
# 	mnist_graph.add_star(nodes_to_add)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

n_node_features = 1
batch_size = 50
n_timepoints = 20
graph_hidden_layer_size = 10
filter_size = 25
n_filter_features = 32
hidden_layer_size = 1024
keep_prob = 1.0
learning_rate = 1e-5

edge_weights = [np.array([0.9, 0.9]), np.array([0.9, 0.9])]
keep_prob_ph = tf.placeholder(tf.float32)

filter_shape = [[filter_size, n_filter_features, n_node_features], [filter_size, 64, n_filter_features]]

# Placeholders
picture_x = tf.placeholder("float32", shape=(batch_size, 784))
picture_y = tf.placeholder("float32", shape=(batch_size, 10))

Convolve1 = GraphConvolution(name='first_conv', G=mnist_graph, n_layers=2, filter_shape=filter_shape, partition_resolution=[0.025, 0.2], n_timepoints=n_timepoints, edge_weights=edge_weights, act_fun=tf.nn.elu)
conv_output = Convolve1(tf.expand_dims(picture_x, 2))
convolved_features = tf.reshape(conv_output, [-1, conv_output.get_shape().as_list()[1]*conv_output.get_shape().as_list()[2]])
OutputMLP = MLP(name='dense_output', dims=[[convolved_features.get_shape().as_list()[1], hidden_layer_size], [hidden_layer_size, 10]], output_fun=tf.identity, dropout=[0, 1 - keep_prob_ph])
yhat = OutputMLP(convolved_features)

prediction = tf.argmax(yhat, axis=1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=picture_y, logits=yhat))
updates = tf.train.AdamOptimizer().minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(picture_y, axis=1)), tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss = []
test_loss = []
plt.ion()
for i in range(20000):
	picture_batch = mnist.train.next_batch(batch_size)
	picture_val_batch = mnist.validation.next_batch(batch_size)
	pictures = picture_batch[0]
	sess.run(updates, feed_dict={picture_x: pictures, picture_y: picture_batch[1], keep_prob_ph: keep_prob})
	predict = sess.run(prediction, feed_dict={picture_x: pictures, picture_y: picture_batch[1], keep_prob_ph: keep_prob})
	if i % 100 == 0:
		acc = sess.run(accuracy, feed_dict={picture_x: pictures, picture_y: picture_batch[1], keep_prob_ph: 1.0})
		loss.append(sess.run(cost, feed_dict={picture_x: pictures, picture_y: picture_batch[1], keep_prob_ph: 1.0}))
		test_loss.append(sess.run(cost, feed_dict={picture_x: picture_val_batch[0], picture_y: picture_val_batch[1], keep_prob_ph: 1.0}))
		test_acc = sess.run(accuracy, feed_dict={picture_x: picture_val_batch[0], picture_y: picture_val_batch[1], keep_prob_ph: 1.0})
		print "Accuracy: " + str(acc) + " Test accuracy: " + str(test_acc)
		first_layer_activation = sess.run(Convolve1.activation[0], feed_dict={picture_x: pictures, picture_y: picture_batch[1], keep_prob_ph: 1.0})
		plt.clf()
		plotNNFilter(first_layer_activation, sess.run(Convolve1.filter_weights[0], feed_dict={picture_x: pictures, picture_y: picture_batch[1], keep_prob_ph: 1.0}))
		plt.pause(0.01)

test_accuracy = []
for i in range(100):
	picture_test_batch = mnist.test.next_batch(batch_size)
	pictures = picture_test_batch[0]
	acc = sess.run(accuracy, feed_dict={picture_x: pictures, picture_y: picture_test_batch[1], keep_prob_ph: 1.0})
	test_accuracy.append(acc)
	print acc
print "Final test accuracy: " + str(np.mean(np.array(test_accuracy)))
