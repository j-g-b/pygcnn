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

root_node = 59
depth = 2
d_neighborhood = make_neighborhood(mnist_graph, root_node, depth, mnist_edge_fun)

A = normalized_adj(d_neighborhood)

x = np.zeros((A.shape[0], 1))
for i in range(x.size):
	if d_neighborhood.nodes()[i] == root_node:
		x[i] = 1

plt.ion()
xt = x
for i in range(100*depth):
	to_plot = xt
	to_plot_labs = d_neighborhood.nodes()
	plt.clf()
	y_pos = np.arange(len(to_plot_labs))
	bars = plt.bar(y_pos, to_plot, align='center', alpha=0.5)
	plt.xticks(y_pos, to_plot_labs)
	plt.ylim(-5,5)
	autolabel(bars)
	#nx.draw_networkx_nodes(d_neighborhood, pos=nx.shell_layout(d_neighborhood), node_color=xt, vmin=0.0, vmax=1.0, cmap='bwr')
	#nx.draw_networkx_edges(d_neighborhood, pos=nx.shell_layout(d_neighborhood))
	plt.pause(0.05)
	xt = A.dot(xt)