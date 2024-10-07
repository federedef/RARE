#!/usr/bin/env python
"""example of CRARE method"""

from param_parser import parameter_parser
from utils import tab_printer
from C_RoleAwareRandomWalk import RoleBased2Vec
import networkx as nx
import numpy as np
import pandas as pd
import random

def read_graph(graph_path, is_directed=False):
    # with open(graph_path, "r") as f:
    #     for i,line in enumerate(f):
    #         if i > 0:
    #             line = line.strip().split("\t")
    #             edgelist1.append((line[0], line[1]))



    data = np.loadtxt(graph_path, dtype=int)
    data = data.astype(int)
    edgelist1 = []  # 1 advice

    for i in range(len(data)):
        edgelist1.append((data[i][0], data[i][1]))

    if is_directed == True:
        G1 = nx.DiGraph()
    else:
        G1 = nx.Graph()
    G1.add_edges_from(edgelist1)
    G1.remove_edges_from(nx.selfloop_edges(G1))

    # G1.remove_edges_from(G1.selfloop_edges())

    return G1

def get_node_embedding(args,G,r,t,m, community_features):
    import time
    start0 = time.time()
    print('get embedding time:')
    model = RoleBased2Vec(args, G, r,t,m, community_features, num_walks=10, walk_length=80, window_size=10)
    # w2v = model.create_embedding()
    w2v = model.train(workers=4)
    print('******************************embedding时间：{}***********************'.format(time.time() - start0))
    return w2v.wv

def get_com_features(label_path):
    com_features = {}
    with open(label_path) as f:
        for line in f:
            line = line.strip().split(" ")
            node = int(line[0])
            comm = int(line[1])
            if com_features.get(node):
                com_features[node].append(comm)
            else:
                com_features[node] = [comm]
    return com_features


if __name__ == "__main__":
    #example
    label_path = r'./data/Category/formated_cls'
    #path = self.args.input

    args = parameter_parser()
    path = args.input #r'./data/Edgelist/ecoli_STRING_700'  #
    print(path)
    tab_printer(args)
    G = read_graph(path)
    print(G)
    # node embedding
    X = get_node_embedding(args, G, r=4,t=0.25,m=1, community_features=get_com_features(args.labels))
    list_arrays=[X.get_vector(str(n)) for n in G.nodes()]
    print(X)
    # print(list_arrays)
