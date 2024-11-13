import pandas as pd
import networkx as nx
from networkx.algorithms.flow import edmonds_karp
import json


max_edge_value = 1_000_000_000_000_000 * 1.0

class GraphStructure:
    def __init__(self, edges = None, enclosures = None, redundancies = None, mttfs = None, mtrs = None):
        if edges is None:
            self.__init_test()
            return
        self.G = nx.DiGraph()
        self.edges = edges
        self.enclosures = enclosures
        self.redundancies = redundancies
        self._add_edges()
        self.redundancy_groups = self._create_redundancy_groups()
        self.mttfs = mttfs
        self.mtrs = mtrs
    
    def __init_test(self):
        self.G = nx.DiGraph()
        self.edges = []
        self.enclosures = {}
        self.availabilities = {}
        self.redundancies = {}
        self.redundancy_groups = {}
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
    def _create_redundancy_groups(self):
        groups = {}
        for module, (M, K) in self.redundancies.items():
            nodes = [node for node in self.G.nodes() if node.startswith(module)]
            group_size = M + K
            for i in range(0, len(nodes), group_size):
                group = (nodes[i:i + group_size], M)
                group_name = f"{module}_group_{i // group_size}"
                groups[group_name] = group
            if (len(nodes) == 0): # if there are no nodes in the module, check enclosures
                group_size = M + K
                nodes = [enclosure for enclosure in self.enclosures if module in enclosure]
                #print (nodes)
                for i in range(0, len(nodes), group_size):
                    group = (nodes[i:i + group_size], M)
                    group_name = f"{module}_group_{i // group_size}"
                    groups[group_name] = group
        #print (groups)
        return groups
    @staticmethod

    def parse_input_from_json(file_path):
        """
        JSON 파일로부터 데이터를 파싱하여 edges, enclosures, availabilities, redundancies, mttfs, mtrs를 반환합니다.
        
        Parameters:
            file_path (str): JSON 파일의 경로.
            
        Returns:
            tuple: (edges, enclosures, availabilities, redundancies, mttfs, mtrs)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Edges: List of tuples (start, end, weight)
        edges = []
        for edge in data.get("edges", []):
            start = edge.get("start")
            end = edge.get("end")
            weight = edge.get("bandwidth")
            if start is not None and end is not None and weight is not None:
                edges.append((start, end, weight))
        
        # Enclosures: Dict of enclosure name to list of nodes
        enclosures = data.get("enclosures", {})
                
        # MTTFs: Dict of module to MTTF
        mttfs = data.get("mttf", {})
        
        # MTRs: Dict of module to MTR
        mtrs = data.get("mtr", {})
        
        # Redundancies: Dict of module to tuple (M, K)
        redundancies = {}
        for module, redundancy in data.get("redundancies", {}).items():
            M = redundancy.get("M")
            K = redundancy.get("K")
            if M is not None and K is not None:
                redundancies[module] = (M, K)
        
        options = data.get("options", {})
        return edges, enclosures, redundancies, mttfs, mtrs, options

    @staticmethod
    def parse_input_from_excel(file_path, sheet_name, start_cell, enclosure_start_cell, availability_sheet):
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        start_row = start_cell[1] - 1
        start_col = ord(start_cell[0].upper()) - ord('A')
        enclosure_start_row = enclosure_start_cell[1] - 1
        enclosure_start_col = ord(enclosure_start_cell[0].upper()) - ord('A')

        edges = []
        for i in range(start_row, len(df)):
            if pd.isna(df.iloc[i, start_col]):
                break
            start = df.iloc[i, start_col]
            end = df.iloc[i, start_col + 1]
            weight = df.iloc[i, start_col + 2]
            edges.append((start, end, weight))
        
        enclosures = {}
        for i in range(enclosure_start_row, len(df)):
            if pd.isna(df.iloc[i, enclosure_start_col]):
                break
            enclosure = df.iloc[i, enclosure_start_col]
            nodes = [str(df.iloc[i, enclosure_start_col + j]) for j in range(1, len(df.columns) - enclosure_start_col) if not pd.isna(df.iloc[i, enclosure_start_col + j])]
            enclosures[enclosure] = nodes

        avail_df = pd.read_excel(file_path, sheet_name=availability_sheet, header=1, usecols="B,C,D,E,F,G")
        mttfs = {str(row['module']): row['MTTF'] for _, row in avail_df.iterrows()}
        mtrs = {str(row['module']): row['MTR'] for _, row in avail_df.iterrows()}
        redundancies = {str(row['module']): (row['M'], row['K']) for _, row in avail_df.iterrows()}
        return edges, enclosures, redundancies, mttfs, mtrs
