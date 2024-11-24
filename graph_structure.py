import pandas as pd
import networkx as nx
from networkx.algorithms.flow import edmonds_karp
import json


max_edge_value = 1_000_000_000_000_000 * 1.0

class GraphStructure:
    def __init__(self, edges = None, enclosures = None, mttfs = None, mtrs = None):
        if edges is None:
            self.__init_test()
            return
        self.G = nx.DiGraph()
        self.edges = edges
        self.enclosures = enclosures
        self._add_edges()
        self.mttfs = mttfs
        self.mtrs = mtrs
    
    def __init_test(self):
        self.G = nx.DiGraph()
        self.edges = []
        self.enclosures = {}
        self.availabilities = {}
        self.redundancies = {}
        self.mttfs = {}
        self.mtrs = {}

    def _add_edges(self):
        for start, end, weight in self.edges:
            self.G.add_edge(start, end, capacity=self.parse_weight(weight), label=weight)

    def _add_availabilities(self):
        for node in self.G.nodes():
            for module, availability in self.availabilities.items():
                if module in node:
                    self.G.nodes[node]['availability'] = availability

    def parse_weight(self, weight):
        if weight.endswith('G'):
            return float(weight[:-1]) * 1_000_000_000
        elif weight.endswith('M'):
            return float(weight[:-1]) * 1_000_000
        else:
            raise ValueError("Unknown weight format")

    def calculate_max_flow(self, start_node_module, leaf_node_module):
        self.add_virtual_nodes(start_node_module, leaf_node_module)
        flow_value, flow_dict = nx.maximum_flow(self.G, 'virtual_source', 'virtual_sink',  flow_func=edmonds_karp)
        self.remove_virtual_nodes()
        return flow_value

    def add_virtual_nodes(self, start_node_module, leaf_node_module):
        self.G.add_node('virtual_source')
        self.G.add_node('virtual_sink')
        for node in self.G.nodes():
            if start_node_module in node:
                #print (start_node_module, node)
                self.G.add_edge('virtual_source', node, capacity=float(max_edge_value))
            if leaf_node_module in node:
                #if ("backend_module" in leaf_node_module):
                #    print (node)
                #print (leaf_node_module, node)
                self.G.add_edge(node, 'virtual_sink', capacity=float(max_edge_value))
    def add_virtual_nodes_for_n_leaf_nodes(self, start_node_module, leaf_node_module, n):
        self.G.add_node('virtual_source')
        self.G.add_node('virtual_sink')
        for node in self.G.nodes():
            if start_node_module in node:
                #print (start_node_module, node)
                self.G.add_edge('virtual_source', node, capacity=float(max_edge_value))
            if leaf_node_module in node:
                node_index = int(node.split('_')[-1])
                if node_index > 0 and node_index < n:
                    self.G.add_edge(node, 'virtual_sink', capacity=float(max_edge_value))
                #print (leaf_node_module, node)
                self.G.add_edge(node, 'virtual_sink', capacity=float(max_edge_value))
    def add_virtual_source(self, start_node_module):
        self.G.add_node('virtual_source')
        for node in self.G.nodes():
            if start_node_module in node:
                self.G.add_edge('virtual_source', node, capacity=float(max_edge_value))

    def add_virtual_sink(self, leaf_node_module, excpetion_nodes = []):
        self.G.add_node('virtual_sink')
        for node in self.G.nodes():
            if leaf_node_module in node and node not in excpetion_nodes:
                self.G.add_edge(node, 'virtual_sink', capacity=float(max_edge_value))

    def remove_virtual_source(self):
        if (self.G.has_node("virtual_source")):
            self.G.remove_node("virtual_source")

    def add_virtual_ssd_nodes(self, ssd_nodes, leaf_node_module, edge_value=float(max_edge_value)):
        self.G.add_node('virtual_sink')
        for node in ssd_nodes:
            if leaf_node_module in node and node in self.G.nodes():
                self.G.add_edge(node, 'virtual_sink', capacity=edge_value)
    
    def remove_virtual_sink(self):
        if (self.G.has_node("virtual_sink")):
            self.G.remove_node("virtual_sink")

    def remove_virtual_nodes(self):
        if (self.G.has_node('virtual_source')):
            self.G.remove_node('virtual_source')
        if (self.G.has_node('virtual_sink')):
            self.G.remove_node('virtual_sink')