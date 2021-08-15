import os
import networkx as nx
from csv import DictReader

os.chdir('/Users/sunho/Dropbox/Boston/CS6220/HW6/')

def build_graph(csv_filename):
    G = nx.Graph()
    with open(csv_filename, 'r') as edges_map:
        csv_dict_reader = DictReader(edges_map)
        for row in csv_dict_reader:
            node_a, node_b = int(row['0']), int(row['1'])
            G.add_edge(node_a, node_b)
    return G

def get_edge_to_remove(G):
    highest_betweenness = float('-inf')
    edge_betweenness_dict = nx.edge_betweenness_centrality(G)
    for edge, betweenness in edge_betweenness_dict.items():
        if betweenness > highest_betweenness:
            highest_betweenness = betweenness
            highest_betweenness_edge = edge
    return highest_betweenness_edge

def check_modularity(G):
    communities = list(nx.connected_components(G))
    modularity = nx.algorithms.community.modularity(G, communities)
    return modularity

def girvan_newman(G, threshold):
    iteration = 0
    modularity = float('-inf')
    while modularity <= threshold:
        iteration += 1
        node_a, node_b = get_edge_to_remove(G)
        G.remove_edge(node_a, node_b)
        print(f'iter {iteration}')
        modularity = check_modularity(G)
        print(f'modularity: {modularity}')
        num_edges = G.number_of_edges()
        print(f'num of edges: {num_edges}')
        num_connected_components = nx.number_connected_components(G)
        print(f'num of connected components: {num_connected_components}\n')

if __name__ == "__main__":
    G = build_graph('edges_sampled_map_3K.csv')
    girvan_newman(G, threshold=0.55)