from pygcnn.utils import *
import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
import tensorflow as tf

weights = np.array([0.5, 0.1, -0.1, -0.1]) #np.random.uniform(0.0, 1.0, 2)
random_graph = nx.grid_2d_graph(28, 28)
root_node = (9,20)
d_neighborhood = make_neighborhood(random_graph, root_node, weights)

A = normalized_adj(d_neighborhood)

plt.ion()

x = np.zeros((A.shape[0], 1))
for i in range(x.size):
	if d_neighborhood.nodes()[i] == root_node:
		x[i] = 1

xt = x
for i in range(100*weights.size):
	to_plot = xt
	to_plot_labs = d_neighborhood.nodes()
	plt.clf()
	y_pos = np.arange(len(to_plot_labs))
	bars = plt.bar(y_pos, to_plot, align='center', alpha=0.5)
	plt.xticks(y_pos, to_plot_labs)
	plt.ylim(0,1)
	autolabel(bars)
	#nx.draw_networkx_nodes(d_neighborhood, pos=nx.shell_layout(d_neighborhood), node_color=xt, vmin=0.0, vmax=1.0, cmap='bwr')
	#nx.draw_networkx_edges(d_neighborhood, pos=nx.shell_layout(d_neighborhood))
	plt.pause(0.05)
	xt = A.dot(xt)