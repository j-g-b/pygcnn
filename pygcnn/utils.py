import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from scipy.stats.stats import pearsonr
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

def load_pubmed(data_fname, graph_fname):
	X, Y, sample_names = parse_pubmed_data(data_fname)
	pubmed_graph = parse_pubmed_graph(graph_fname)
	pubmed_graph = nx.relabel_nodes(pubmed_graph, {sample_names[i]: i for i in range(len(sample_names))})
	pubmed_graph = nx.subgraph(pubmed_graph, range(X.shape[0]))
	return X, Y, pubmed_graph

def parse_pubmed_graph(fname):
	with open(fname) as f:
		lines = f.readlines()
	split_lines = [line.strip("\n").split("\t") for line in lines[2:len(lines)]]
	edge_list = [[sl[1].strip('paper:'), sl[3].strip('paper:')] for sl in split_lines]
	return nx.from_edgelist(edge_list)

def parse_pubmed_data(fname):
	with open(fname) as f:
		lines = f.readlines()
	split_lines = [line.split("\t") for line in lines[2:len(lines)]]
	dataset = []
	word_dict = {}
	word_counter = 0
	for line in split_lines:
		word_vec = line[2:(len(line)-1)]
		for word in word_vec:
			strip_word = word.split('=')[0]
			if not strip_word in word_dict.values():
				word_dict[word_counter] = strip_word
				word_counter = word_counter + 1
		dataset.append([line[0], line[1].strip('label='), word_vec])
	inv_word_dict = {word_dict[word_dict.keys()[i]]: word_dict.keys()[i] for i in range(len(word_dict.keys()))}
	for i in range(len(dataset)):
		feature_arr = len(word_dict.keys())*[0]
		word_vec = dataset[i][2]
		for j in range(len(word_vec)):
			strip_word = word_vec[j].split('=')[0]
			value = word_vec[j].split('=')[1]
			feature_arr[inv_word_dict[strip_word]] = value
		dataset[i][2] = np.array(feature_arr, dtype='float32')
	classes = np.array(['1', '2', '3'])
	binary_classes = np.identity(len(classes))
	X = np.concatenate([np.reshape(sample[2], (1, sample[2].size)) for sample in dataset])
	Y = np.concatenate([np.reshape(binary_classes[np.where(classes == sample[1])], (1, len(classes))) for sample in dataset])
	sample_names = [sample[0] for sample in dataset]
	return X, Y, sample_names

def load_cora(data_fname, graph_fname):
	X, Y, sample_names = parse_cora_data(data_fname)
	cora_graph = parse_cora_graph(graph_fname)
	cora_graph = nx.relabel_nodes(cora_graph, {sample_names[i]: i for i in range(len(sample_names))})
	cora_graph = nx.subgraph(cora_graph, range(X.shape[0]))
	return X, Y, cora_graph

def parse_cora_graph(fname):
	with open(fname) as f:
		lines = f.readlines()
	split_lines = [line.strip("\n").split("\t") for line in lines]
	return nx.from_edgelist(split_lines)

def parse_cora_data(fname):
	with open(fname) as f:
		lines = f.readlines()
	split_lines = [line.split("\t") for line in lines]
	dataset = [[line[0], line[-1].strip("\n"), np.array(line[1:(len(line)-2)], dtype="float32")] for line in split_lines]
	classes = np.array(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])
	binary_classes = np.identity(len(classes))
	X = np.concatenate([np.reshape(sample[2], (1, sample[2].size)) for sample in dataset])
	Y = np.concatenate([np.reshape(binary_classes[np.where(classes == sample[1])], (1, len(classes))) for sample in dataset])
	sample_names = [sample[0] for sample in dataset]
	return X, Y, sample_names

def load_citeseer(data_fname, graph_fname):
	X, Y, sample_names = parse_citeseer_data(data_fname)
	citeseer_graph = parse_citeseer_graph(graph_fname)
	citeseer_graph = nx.relabel_nodes(citeseer_graph, {sample_names[i]: i for i in range(len(sample_names))})
	citeseer_graph = nx.subgraph(citeseer_graph, range(X.shape[0]))
	return X, Y, citeseer_graph

def parse_citeseer_graph(fname):
	with open(fname) as f:
		lines = f.readlines()
	split_lines = [line.strip("\n").split("\t") for line in lines]
	return nx.from_edgelist(split_lines)

def parse_citeseer_data(fname):
	with open(fname) as f:
		lines = f.readlines()
	split_lines = [line.split("\t") for line in lines]
	dataset = [[line[0], line[-1].strip("\n"), np.array(line[1:(len(line)-2)], dtype="float32")] for line in split_lines]
	classes = np.array(['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI'])
	binary_classes = np.identity(len(classes))
	X = np.concatenate([np.reshape(sample[2], (1, sample[2].size)) for sample in dataset])
	Y = np.concatenate([np.reshape(binary_classes[np.where(classes == sample[1])], (1, len(classes))) for sample in dataset])
	sample_names = [sample[0] for sample in dataset]
	return X, Y, sample_names

def translate_array(arr, translate_dict):
	return np.vectorize(translate_dict.__getitem__)(arr)

def evaluate_gradient(loss, tensor_list, session, feed_dict):
	grad_list = tf.gradients(loss, tensor_list)
	eval_grad_list = []
	for grad in grad_list:
		eval_grad_list.append(session.run(grad, feed_dict=feed_dict))
	return eval_grad_list

def align_and_concat(df_list, margin='col'):
	# assumes concat happens on opposite margin from align; margin specifies align margin
	if margin == 'row':
		df_t_list = [df.transpose() for df in df_list]
		common_fields = set(list(df_t_list[0]))
		for i in range(len(df_t_list)-1):
			common_fields = common_fields & set(list(df_t_list[i+1]))
		df_al_list = [df[list(common_fields)] for df in df_t_list]
		return pd.concat(df_al_list).transpose()
	if margin == 'col':
		common_fields = set(list(df_list[0]))
		for i in range(len(df_list)-1):
			common_fields = common_fields & set(list(df_list[i+1]))
		df_al_list = [df[list(common_fields)] for df in df_list]
		return pd.concat(df_al_list)

def draw_neighborhood(GC, node, session, fig, steps=1, highlight=None, layer=0):
	G = GC.G[layer]
	if highlight is None:
		highlight = node
	neighbors = [node]
	while steps > 0:
		for neighbor in neighbors:
			neighbors = neighbors + G.neighbors(neighbor)
		steps = steps - 1
	nodelist = list(set(neighbors))
	color = ['b' if n is not highlight else 'r' for n in nodelist]
	ax1 = fig.add_subplot(221)
	nx.draw_networkx(G.subgraph(nodelist), nodelist=nodelist, node_color=color)
	neighborhood_start = node*max(GC.neighborhood_sizes[layer])
	index = []
	for i in range(GC.neighborhood_sizes[layer][node]):
		if GC.neighborhoods[0][neighborhood_start + i] == highlight:
			index.append(sum(GC.neighborhood_sizes[layer][0:node]) + i)
	highlighted_indexing = np.take(GC.subgraph_features[layer], indices=index, axis=0)
	soft_filter_indexing = GC.indexing_mlp[layer](tf.constant(highlighted_indexing, dtype=tf.float32))
	ax2 = fig.add_subplot(222)
	plt.bar(np.arange(highlighted_indexing.shape[1]), height=highlighted_indexing.flatten())
	ax3 = fig.add_subplot(223)
	plt.imshow(soft_filter_indexing.eval(session=session), cmap="gray")
	ax3 = fig.add_subplot(224)
	plt.bar(np.arange(soft_filter_indexing.eval(session=session).shape[1]), height=soft_filter_indexing.eval(session=session).flatten())
	plt.show()

def nearest_neighbors(normalized_embeddings, session, k, int2gene):
	graph_dict = {}
	pairwise_distances = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True).eval(session=session)
	for i in range(pairwise_distances.shape[0]):
		nearest = (-pairwise_distances[i, :]).argsort()[1:k + 1]
		sorted_sim = np.sort(-pairwise_distances[i, :])[1:k + 1]
		graph_dict[int2gene[i]] = {int2gene[nearest[l]]: {'weight': -sorted_sim[l]} for l in range(k)}
	return nx.from_dict_of_dicts(graph_dict)

def find_similar_genes(similarity, query_genes, query_genes_ph, session, gene2int, int2gene):
	sim = session.run(similarity, feed_dict={query_genes_ph: np.array([gene2int[k] for k in query_genes])})
	for i in xrange(len(query_genes)):
		query_gene = query_genes[i]
		top_k = 8
		nearest = (-sim[i, :]).argsort()[1:top_k + 1]
		sorted_sim = np.sort(-sim[i, :])[1:top_k + 1]
		log_str = 'Nearest to %s:' % query_gene
		for k in xrange(top_k):
			close_gene = int2gene[nearest[k]]
			close_dist = str(-round(sorted_sim[k], 3))
			log_str = '%s %s,' % (log_str, close_gene + " (" + close_dist + ")")
		print(log_str)

def select_k_best(features, targets, k):
	selected_features = {}
	feature_names = np.array(list(features))
	for target in list(targets):
		print "\r" + target
		y = targets[target].as_matrix()
		k_best = SelectKBest(f_regression, k=k).fit(features.as_matrix(), y)
		selected_features[target] = feature_names[np.where(k_best.get_support())]
	return selected_features

def sp_matrix_from_msigdb_dict(msig_dict):
	gene_sets = list(set(msig_dict.keys()))
	genes = list(set([item for sublist in msig_dict.values() for item in sublist]))
	gene_set_index = {gene_sets[i]: i for i in range(len(gene_sets))}
	gene_index = {genes[i]: i for i in range(len(genes))}
	sp_rows = []
	sp_cols = []
	for gene_set in msig_dict.keys():
		members = msig_dict[gene_set]
		for gene in members:
			sp_rows.append(gene_set_index[gene_set])
			sp_cols.append(gene_index[gene])
	return gene_set_index, gene_index, sp.coo_matrix((np.ones(len(sp_rows)), (np.array(sp_rows), np.array(sp_cols))))

def dict_from_msigdb(msig_filename, max_size):
	f = open(msig_filename)
	lines = f.readlines()
	gene_set_dict = {}
	for line in lines:
		split_line = line[0:-1].split("\t")
		if len(split_line) < max_size - 2:
			gene_set_dict[split_line[0]] = line[0:-1].split("\t")[2:]
	return gene_set_dict

def marginal_correlation_screening(features, targets, top_n=10):
	top_mat = np.zeros((targets.shape[1], top_n))
	for target in range(targets.shape[1]):
		print "Screening target " + str(target)
		m_cors = [pearsonr(features[:, i], targets[:, target])[0] for i in range(features.shape[1])]
		top_mat[target, :] = np.argsort(np.array(m_cors))[-top_n:]
	return top_mat

def maximum_marginal_diversity(features, targets, class_labels, class_priors, top_n=10):
	top_mat = np.zeros((targets.shape[1], top_n))
	for target in range(targets.shape[1]):
		md_features = []
		for feature in range(features.shape[1]):
			md = marginal_diversity(features[:, feature], targets[:, target], class_labels, class_priors)
			md_features.append(md)
		top_mat[target, :] = np.argsort(np.array(md_features))[-top_n:]
	return top_mat

def marginal_diversity(feature, target, class_labels, class_priors):
	h, e = np.histogram(feature, bins='auto', density=True)
	hist_mat = np.zeros((h.shape[0], len(class_labels)))
	for label in class_labels:
		hist_mat[:, label], temp = np.histogram(feature[np.where(target == label)], bins=e, density=True)
	hist_mat = hist_mat + np.abs(1e-4*np.random.randn(hist_mat.shape[0], hist_mat.shape[1]))
	h_m = np.mean(hist_mat, axis=1)
	h_m = np.diag(1.0 / h_m)
	md = np.dot(class_priors, np.dot(np.transpose(hist_mat), np.log(np.dot(h_m, hist_mat))).diagonal())
	return md

def tf_task_mask(shape, soft_sharing):
	assert shape[0] > shape[1]
	# note: requires more input weights than output tasks
	shape = tuple(shape)
	task_mask = np.zeros(shape)
	weights_per_task = int(np.floor(float(soft_sharing*shape[0]) / shape[1]))
	row_counter = 0
	for task in range(shape[1]):
		task_mask[range(row_counter, row_counter + weights_per_task), task] = 1
		row_counter += weights_per_task
	task_mask[range(row_counter, shape[0]), :] = 1
	return tf.constant(task_mask)

def graph_from_data(data, dist_fun=np.corrcoef, d=4):
	dist_mat = dist_fun(data)
	np.fill_diagonal(dist_mat, 0.0)
	closest_indices = np.concatenate((np.argsort(dist_mat)[:, -d:], np.argsort(dist_mat)[:,0:d]), axis=1)
	graph_dict = {indx: list(closest_indices[indx, :]) for indx in range(closest_indices.shape[0])}
	graph = nx.from_dict_of_lists(graph_dict)
	for edge in graph.edges():
		graph.edge[edge[0]][edge[1]]['weight'] = dist_mat[edge[0], edge[1]]
	return graph

def make_neighborhood(graph, node_id, weights):
	d_neighbors = neighborhood(graph, node_id, weights.size)
	subgraph = nx.Graph(graph.subgraph(d_neighbors))
	set_edge_weights(subgraph, node_id, weights)
	return subgraph

def set_edge_weights(subgraph, node_id, weights):
	to_visit = [node_id, subgraph.neighbors(node_id)]
	visited = [node_id]
	subgraph.node[node_id]['visited_by'] = [node_id]
	subgraph.node[node_id]['depth'] = 0
	edges_to_prune = []
	while to_visit:
		parent_node = to_visit.pop(0)
		child_nodes = to_visit.pop(0)
		for child_node in child_nodes:
			if not child_node in visited:
				to_visit.append(child_node)
				to_visit.append(subgraph.neighbors(child_node))
				subgraph.node[child_node]['visited_by'] = [parent_node]
				subgraph.node[parent_node]['visited_by'].append(child_node)
				subgraph.node[child_node]['depth'] = subgraph.node[parent_node]['depth'] + 1
				subgraph.edge[parent_node][child_node]['weight'] = weights[subgraph.node[parent_node]['depth']]
				visited.append(child_node)
			elif not parent_node in subgraph.node[child_node]['visited_by']:
				min_depth = min(subgraph.node[parent_node]['depth'], subgraph.node[child_node]['depth'])
				if not min_depth == weights.size:
					subgraph.node[child_node]['visited_by'].append(parent_node)
					subgraph.node[parent_node]['visited_by'].append(child_node)
					subgraph.edge[parent_node][child_node]['weight'] = weights[min_depth]
				else:
					edges_to_prune.append((parent_node, child_node))
	subgraph.remove_edges_from(edges_to_prune)


def neighborhood(graph, node_id, max_depth):
	to_visit = [{'depth': 0, 'node_id': node_id}, graph.neighbors(node_id)]
	d_neighbors = [node_id]
	while to_visit:
		parent_node_dict = to_visit.pop(0)
		child_nodes = to_visit.pop(0)
		if not parent_node_dict['depth'] == max_depth:
			for child_node in child_nodes:
				if not child_node in d_neighbors:
					to_visit.append({'depth': parent_node_dict['depth'] + 1, 'node_id': child_node})
					to_visit.append(graph.neighbors(child_node))
					d_neighbors.append(child_node)
	return d_neighbors

def normalized_adj(subgraph, weight='None', alpha=0.5):
    degrees = np.array([subgraph.degree(node) for node in subgraph.nodes()])
    d_inv = np.power(degrees, -1.0).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    weighted_adj = nx.adjacency_matrix(subgraph)
    res = d_mat_inv.dot(weighted_adj).tocoo()
    rowsum = res.dot(np.ones(degrees.size))
    diag = 1.0 - rowsum
    res.setdiag(diag.flatten())
    return res

def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 1.05*height,
                str(round(height, 2)),
                ha='center', va='bottom')

def propagate(x0, A, t):
	xt = x0
	channels = [x0]
	while t > 0:
		xt = A.dot(xt)
		channels.append(xt)
		t -= 1
	return np.concatenate(channels, axis=1)