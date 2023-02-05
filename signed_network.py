# signed_network

# from cycles import find_all_cycles

import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd

from itertools import combinations
from numba import njit


class SignedNetwork:
    def __init__(self, df, create_using=nx.DiGraph, num_nodes=None, empty=0):
        self.df = df
        self.num_edges = len(self.df)
        self.graph = self.create_graph_from_frame(create_using)

        self.adj = nx.to_numpy_array(self.graph, dtype=float, weight="weight", nonedge=empty)
        self.num_nodes = num_nodes if num_nodes else np.shape(self.adj)[0]
        self.node_range = range(self.num_nodes)

    def create_graph_from_frame(self, graph_mode):
        return nx.from_pandas_edgelist(self.df,
                                       source = "source",
                                       target = "target",
                                       edge_attr = True,
                                       create_using=graph_mode)

    """
    def find_cycles(self, max_length=None):
        cycles = find_all_cycles(self.graph, None, 1)
        if max_length:
            cycles = [cycle for cycle in cycles if len(cycle) <= max_length]
        self.cycles = cycles
        return cycles

    def calculate_filtered_matrices(self, over=0, under=0):
        self.adj_pos_w =  self.adj*(self.adj > over)
        self.adj_neg_w = -self.adj*(self.adj < under)

        # get weighted degree matrices for subgraphs
        self.deg_pos_w = np.diag(np.sum(self.adj_pos_w, 1))
        self.deg_neg_w = np.diag(np.sum(self.adj_neg_w, 1))

    def calculate_unweighted_matrices(self):
        # times 1 to convert boolean array to int array
        self.adj_pos = 1*(self.adj > 0)
        self.adj_neg = 1*(self.adj < 0)

        # get degree matrices for subgraphs
        self.deg_pos = np.diag(np.sum(self.adj_pos, 1))
        self.deg_neg = np.diag(np.sum(self.adj_neg, 1))

    def calculate_laplacians(self):
        self.calculated_influence()
        self.lap_pos = self.deg_pos - self.adj_pos
        self.lap_neg = self.deg_neg - self.adj_neg
        self.lap_abs = self.influence_abs - self.adj

    def calculated_influence(self):
        influences = np.sum(self.adj, 1)
        self.influence = np.diag(influences)
        self.influence_abs = np.diag(np.sum(np.abs(self.adj), 1))
        self.skew = sum(influences)

    def calculate_pre_suc(self):
        # predecessors
        self.pre = np.zeros((self.num_nodes, self.num_nodes))
        self.pre_pos = [[] for n in self.node_range]
        self.pre_neg = [[] for n in self.node_range]
        self.pre_pos_w = [0]*self.num_nodes
        self.pre_neg_w = [0]*self.num_nodes

        # successors
        self.suc = np.zeros((self.num_nodes, self.num_nodes))
        self.suc_pos = [[] for n in self.node_range]
        self.suc_neg = [[] for n in self.node_range]
        self.suc_pos_w = [0]*self.num_nodes
        self.suc_neg_w = [0]*self.num_nodes

        # iterate over the adjacency matrix
        for i in self.node_range:
            for j in self.node_range:
                if not (val := self.adj[i, j]):
                    continue
                
                # there is an edge from i to j
                self.suc[i].append(j)
                self.pre[j].append(i)
                
                if val > 0:
                    self.suc_pos[i].append(j)
                    self.suc_pos_w[i] += val
                    self.pre_pos[j].append(i)
                    self.pre_pos_w[i] += val

                elif val < 0:
                    self.suc_neg[i].append(j)
                    self.suc_neg_w[i] += val
                    self.pre_neg[j].append(i)
                    self.pre_neg_w[i] += val
    """

    def calculate_in_out_deg(self, give_warning=False):
        self.adj_unweighted = 1*(self.adj != 0)
        self.in_deg = np.sum(self.adj_unweighted, axis=0)
        self.out_deg = np.sum(self.adj_unweighted, axis=1)

        if give_warning:
            if 0 in self.in_deg:
                print("There is a node with in-degree 0.")
            if 0 in self.out_deg:
                print("There is a node with out-degree 0.")

        self.in_deg_inv = np.array([1/in_deg if in_deg else 0
                                    for in_deg in self.in_deg])
        self.out_deg_inv = np.array([1/out_deg if out_deg else 0
                                     for out_deg in self.out_deg])

    def calculate_scores(self, epsilon, max_weight=1):
        goodness = np.ones(self.num_nodes, dtype=float)
        fairness = np.ones(self.num_nodes, dtype=float)
        new_goodness = np.ones(self.num_nodes, dtype=float)
        new_fairness = np.ones(self.num_nodes, dtype=float)

        r_inv = 1/(2*max_weight)
        self.calculate_in_out_deg()

        update_goodness(self.num_nodes,
                        new_goodness, fairness,
                        self.in_deg_inv, self.adj)
        update_fairness(self.num_nodes,
                        new_fairness, new_goodness,
                        self.out_deg_inv, self.adj,
                        self.adj_unweighted, r_inv)

        while (np.sum(np.abs(new_fairness - fairness)) > epsilon) \
            or (np.sum(np.abs(new_goodness - goodness)) > epsilon):
                goodness = np.array(new_goodness, copy=True)
                fairness = np.array(new_fairness, copy=True)
                update_goodness(self.num_nodes,
                                new_goodness, fairness,
                                self.in_deg_inv, self.adj)
                update_fairness(self.num_nodes,
                                new_fairness, new_goodness,
                                self.out_deg_inv, self.adj,
                                self.adj_unweighted, r_inv)
        
        return new_fairness, new_goodness

    def make_predictions(self, epsilon, value_func, max_weight=1, mode="stale"):
        if mode == "stale":
            return self.predict_fair_good(epsilon, max_weight, value_func)

        if mode == "time":
            return self.time_series_fair_good(epsilon, max_weight, value_func)

    def predict_fair_good(self, epsilon, max_weight, value_func):
        # arrays to store values
        predicted_edges = np.zeros(self.num_edges, dtype=float)
        errors = np.zeros_like(predicted_edges)

        self.calculate_in_out_deg()

        sources = np.array(self.df["source"], dtype=int)
        targets = np.array(self.df["target"], dtype=int)
        weights = np.array(self.df["weight"])

        for k in range(self.num_edges):
            # creating a sub-dataframe
            leave_in = (n for n in range(self.num_edges) if n != k)
            sample_df = self.df.iloc[leave_in]

            # calculating the goodness and fairness scores on the sub-network
            sample_net = SignedNetwork(sample_df)
            fair, good = sample_net.calculate_scores(epsilon, max_weight)

            u = sources[k]
            v = targets[k]
            weight = weights[k]

            unknown_u = self.out_deg[u] < 2
            unknown_v = self.in_deg[v] < 2
            prediction = value_func(u, v, sample_df, fair, good, unknown_u, unknown_v)

            predicted_edges[k] = prediction
            errors[k] = abs(prediction - weight)

        return predicted_edges, errors

    def time_series_fair_good(self, epsilon, max_weight, value_func):
        """
        Note to new_nodes:
            if new_nodes[k] == 0:
                both u and v are already in the network
            if new_nodes[k] == 1:
                u is a new node
            if new_nodes[k] == 2:
                v is a new node
            if new_nodes[k] == 3:
                both u and v are new nodes
        """
        predictions = np.zeros(self.num_edges, dtype=float)
        errors = np.zeros_like(predictions)
        latest_node = -1

        sources = np.array(self.df["source"], dtype=int)
        targets = np.array(self.df["target"], dtype=int)
        weights = np.array(self.df["weight"])


        for k in range(self.num_edges):
            u = sources[k]
            v = targets[k]
            weight = weights[k]

            sample_df = self.df[:k]
            sample_net = SignedNetwork(sample_df)
            fair, good = sample_net.calculate_scores(epsilon, max_weight)

            new_u = u > latest_node
            new_v = v > latest_node
            prediction = value_func(u, v, sample_df, fair, good, new_u, new_v)

            latest_node = max(u, v)
            predictions[k] = prediction
            errors[k] = abs(weight - prediction)

        return predictions, errors

@njit
def update_goodness(N, goodness, fairness, in_deg_inv, adj):
    """
    Updates the goodness scores in-place as proposed in KUMAR.
    """
    for v in range(N):
        goodness[v] = in_deg_inv[v]*np.sum(fairness*adj[:,v])

@njit
def update_fairness(N, fairness, goodness, out_deg_inv, adj, out, r_inv):
    """
    Updates the fairness scores in-place as proposed in KUMAR.
    """
    for u in range(N):
        sum_value = np.sum(np.abs(adj[u,:] - goodness*out[u,:]))
        fairness[u] = 1 - out_deg_inv[u]*sum_value*r_inv    

# @njit
# def predict_weights(edges, predictions, errors, bookkeeper,
#                     fairness, goodness, source, target, weight):
#     """
#     Predicts the weight of an edge using fairness and goodness.
#     Updates predicitons and errors in-place.
#     """
#     for edge in edges:
#         u, v, w = source[edge], target[edge], weight[edge]
#         prediction = fairness[u]*goodness[v]

#         bookkeeper[u, v] += 1
#         store_index = bookkeeper[u, v]

#         predictions[u, v, store_index] = prediction
#         errors[u, v, store_index] = abs(prediction - w)