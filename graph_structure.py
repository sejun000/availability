import pandas as pd
import networkx as nx

class GraphStructure:
    def __init__(self, edges, enclosures, availabilities, redundancies, mttfs, mtrs):
        self.G = nx.DiGraph()
        self.edges = edges
        self.enclosures = enclosures
        self.availabilities = availabilities
        self.redundancies = redundancies
        self._add_edges()
        self._add_availabilities()
        self.redundancy_groups = self._create_redundancy_groups()
        self.mttfs = mttfs
        self.mtrs = mtrs

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

    def calculate_max_flow(self):
        self.add_virtual_nodes()
        flow_value, flow_dict = nx.maximum_flow(self.G, 'virtual_source', 'virtual_sink')
        self.remove_virtual_nodes()
        return flow_value

    def add_virtual_nodes(self):
        self.G.add_node('virtual_source')
        self.G.add_node('virtual_sink')
        for node in self.G.nodes():
            if "switch" in node:
                self.G.add_edge('virtual_source', node, capacity=float('inf'))
            if "dfm" in node:
                self.G.add_edge(node, 'virtual_sink', capacity=float('inf'))

    def remove_virtual_nodes(self):
        self.G.remove_node('virtual_source')
        self.G.remove_node('virtual_sink')
    def _create_redundancy_groups(self):
        groups = {}
        for module, (M, K) in self.redundancies.items():
            nodes = [node for node in self.G.nodes() if module in node]
            group_size = M + K
            #print (module, M, K)
            for i in range(0, len(nodes), group_size):
                group = (nodes[i:i + group_size], M)
                group_name = f"{module}_group_{i // group_size}"
                groups[group_name] = group
        return groups
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
        availabilities = {str(row['module']): row['Availability'] for _, row in avail_df.iterrows()}
        mttfs = {str(row['module']): row['MTTF'] for _, row in avail_df.iterrows()}
        mtrs = {str(row['module']): row['MTR'] for _, row in avail_df.iterrows()}
        redundancies = {str(row['module']): (row['M'], row['K']) for _, row in avail_df.iterrows()}
        #print (redundancies)
        #redundancy_df = pd.read_excel(file_path, sheet_name=availability_sheet, header=1, usecols="B,F,G")
        #redundancies = {}
        #for _, row in redundancy_df.iterrows():
        #    redundancies[str(row['module'])] = (int(row['M']), int(row['K']))
        return edges, enclosures, availabilities, redundancies, mttfs, mtrs
