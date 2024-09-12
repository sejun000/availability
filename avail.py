import pandas as pd
import networkx as nx

class GraphStructure:
    def __init__(self, file_path, architecture_sheet, start_cell, enclosure_start_cell, availability_sheet):
        self.G = nx.DiGraph()
        self.enclosures = {}
        self.load_data(file_path, architecture_sheet, start_cell, enclosure_start_cell, availability_sheet)

    def load_data(self, file_path, architecture_sheet, start_cell, enclosure_start_cell, availability_sheet):
        # Load edges and enclosures from the architecture sheet
        df = pd.read_excel(file_path, sheet_name=architecture_sheet, header=None)
        for index, row in df.iterrows():
            start, end, weight = row[start_cell[1]-1], row[start_cell[1]], row[start_cell[1]+1]
            if pd.notna(start) and pd.notna(end) and pd.notna(weight):
                self.G.add_edge(start, end, weight=weight, capacity=self.parse_weight(weight))
        
        # Load enclosures
        enclosure_df = pd.read_excel(file_path, sheet_name=architecture_sheet, header=None)
        for index, row in enclosure_df.iterrows():
            enclosure = row[enclosure_start_cell[1]-1]
            nodes = row[enclosure_start_cell[1]:]
            if pd.notna(enclosure):
                self.enclosures[enclosure] = [node for node in nodes if pd.notna(node)]

        # Load availability data
        availability_df = pd.read_excel(file_path, sheet_name=availability_sheet, header=None)
        for index, row in availability_df.iterrows():
            node, availability = row[1], row[3]  # Assuming availability data starts from second column
            if pd.notna(node) and pd.notna(availability):
                for n in self.G.nodes:
                    if node in n:
                        self.G.nodes[n]['availability'] = availability

    def parse_weight(self, weight):
        if weight.endswith('G'):
            return float(weight[:-1]) * 1_000_000_000
        elif weight.endswith('M'):
            return float(weight[:-1]) * 1_000_000
        else:
            raise ValueError("Unknown weight format")

    def add_virtual_nodes(self):
        self.G.add_node('virtual_source')
        self.G.add_node('virtual_sink')
        for node in self.G.nodes:
            if "switch" in node:
                self.G.add_edge('virtual_source', node, capacity=float('inf'))
            if leaf_node_module in node:
                self.G.add_edge(node, 'virtual_sink', capacity=float('inf'))

    def remove_virtual_nodes(self):
        self.G.remove_node('virtual_source')
        self.G.remove_node('virtual_sink')

    def get_graph(self):
        return self.G

    def calculate_max_flow(self):
        self.add_virtual_nodes()
        flow_value, flow_dict = edmonds_karp(self.G, 'virtual_source', 'virtual_sink')
        self.remove_virtual_nodes()
        return flow_value
