import networkx as nx
import numpy as np
from itertools import product
from collections import Counter
import scipy.special as spp


def poisson_pdf(x, mu):
    x = np.array(x, dtype=np.int64)
    return (x**mu / spp.factorial(x)) * np.exp(-mu)

def exp_pdf(x,mu):
    x = np.array(x, dtype=np.int64)
    return np.exp(-x / mu) / mu

def avg_degree(g):
    links = np.sum(nx.adjacency_matrix(g).toarray())
    return 2 * g.number_of_edges()/g.number_of_nodes()

def get_colours(graph) -> list:
    from pptx.dml.color import RGBColor
    out = list(dict(graph.nodes.data('color')).values())
    out = list(map(lambda x: x[4:-1], out))
    out = list(map(lambda x: x.split(','), out))
    out = list(map(lambda x: list(map(float, x)),out))
    out = list(map(lambda x: np.array(x)/255, out))
    return out

def get_sizes(graph) -> list:
    return list(np.array(list(dict(graph.nodes.data('size')).values()))*6)

def get_positions(graph) -> dict:
    positions = {}
    for p in dict(graph.nodes.data()):
        positions[p] = tuple([float(y) for y in graph.nodes.data()[p]['position'][1:-1].split(',')])
    return positions

def get_sorted_freq(graph):
    classes = list(dict(graph.nodes.data('disclass')).values())
    sorted_classes = dict(sorted(Counter(classes).items(), key=lambda x: x[1], reverse=True))
    return list(sorted_classes.keys()), list(sorted_classes.values())

def encoded_classes(graph):
    cc, _ = get_sorted_freq(graph)
    values = list(dict(graph.nodes.data('disclass')).values())
    encoded_classes = {v: i for i, v in enumerate(cc)}
    results = {i: encoded_classes[v] for i, v in enumerate(values)}
    return {'encoded_class': results}

def create_subgraph_with_classes(graph, classes: list):
    return graph.subgraph(get_subgraph_nodes(graph, classes))

def create_subgraph_with_nodes(graph, nodes: list):
    return graph.subgraph(nodes)

def get_subgraph_nodes(graph, classes: list):
    classes = set(classes)
    graph_nodes = dict(graph.nodes.data('disclass'))
    nodes_wanted = [k for k,v in graph_nodes.items() if graph_nodes[k] in classes]

    return nodes_wanted

def get_metadata_communities(graph):
    classes = set(list(dict(graph.nodes.data('disclass')).values()))
    community_list = []
    for c in classes:
        community_list.append(set(get_subgraph_nodes(graph, [c])))

    return community_list

def get_densities(graph, communities, num_ps):
    classes = range(len(communities))

    communities = sorted(communities, key=lambda x: len(x), reverse=True)
    combinations = list(product(classes, classes))
    probs = []

    for c in combinations:

        class1 = c[0]
        class2 = c[1]

        nodes1 = communities[class1]
        nodes2 = communities[class2]

        sub = create_subgraph_with_nodes(graph, list(nodes1) + list(nodes2))
        sub_edges = set(list(sub.edges()))

        sub_nodes_combs1 = list(product(nodes1, nodes2))
        sub_nodes_combs2 = list(product(nodes2, nodes1))
        sub_nodes_combs = sub_nodes_combs1 + sub_nodes_combs2

        num_inter_links = len(set(sub_nodes_combs).intersection(sub_edges))

        probs.append(num_inter_links)

    probs = np.array(probs).reshape(-1, num_ps)
    probs_diagsum = np.sum(np.diagonal(probs))
    probs = probs / probs_diagsum

    return probs
