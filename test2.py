import networkx as nx

# Create the directed graph
G = nx.DiGraph()

# Add edges with capacities
edges_with_capacity = [
    ("io_module0", "ssd0", 1969230769.2307692),
    ("io_module0", "ssd1", 1969230769.2307692),
    ("io_module0", "ssd2", 1969230769.2307692),
    ("io_module0", "ssd3", 1969230769.2307692),
    ("io_module0", "ssd4", 1969230769.2307692),
    ("io_module0", "ssd5", 1969230769.2307692),
    ("io_module0", "ssd6", 1992307692.3076923),
    ("io_module0", "ssd7", 2000000000.0),
    ("io_module0", "ssd8", 2000000000.0),
    ("io_module0", "ssd9", 2000000000.0),
    ("io_module0", "ssd10", 2000000000.0),
    ("io_module0", "ssd11", 2000000000.0),
    ("io_module0", "ssd12", 2000000000.0),
    ("io_module0", "ssd13", 2000000000.0),
    ("io_module0", "ssd14", 2000000000.0),
    ("io_module0", "ssd15", 2000000000.0),
    ("io_module0", "ssd16", 2000000000.0),
    ("io_module0", "ssd17", 2000000000.0),
    ("io_module0", "ssd18", 2000000000.0),
    ("io_module0", "ssd19", 2000000000.0),
    ("io_module0", "ssd20", 2000000000.0),
    ("io_module0", "ssd21", 2000000000.0),
    ("io_module0", "ssd22", 2000000000.0),
    ("io_module0", "ssd23", 2000000000.0),
    ("io_module0", "ssd24", 2000000000.0),
    ("io_module0", "ssd25", 2000000000.0),
    ("io_module0", "ssd26", 2000000000.0),
    ("io_module0", "ssd27", 2000000000.0),
    ("io_module0", "ssd28", 2000000000.0),
    ("io_module0", "ssd29", 2000000000.0),
    ("io_module0", "ssd30", 2000000000.0),
    ("io_module0", "ssd31", 2000000000.0),
    ("io_module1", "ssd0", 1969230769.2307692),
    ("io_module1", "ssd1", 1969230769.2307692),
    ("io_module1", "ssd2", 1969230769.2307692),
    ("io_module1", "ssd3", 1969230769.2307692),
    ("io_module1", "ssd4", 1969230769.2307692),
    ("io_module1", "ssd5", 1969230769.2307692),
    ("io_module1", "ssd6", 1992307692.3076923),
    ("io_module1", "ssd7", 2000000000.0),
    ("io_module1", "ssd8", 2000000000.0),
    ("io_module1", "ssd9", 2000000000.0),
    ("io_module1", "ssd10", 2000000000.0),
    ("io_module1", "ssd11", 2000000000.0),
    ("io_module1", "ssd12", 2000000000.0),
    ("io_module1", "ssd13", 2000000000.0),
    ("io_module1", "ssd14", 2000000000.0),
    ("io_module1", "ssd15", 2000000000.0),
    ("io_module1", "ssd16", 2000000000.0),
    ("io_module1", "ssd17", 2000000000.0),
    ("io_module1", "ssd18", 2000000000.0),
    ("io_module1", "ssd19", 2000000000.0),
    ("io_module1", "ssd20", 2000000000.0),
    ("io_module1", "ssd21", 2000000000.0),
    ("io_module1", "ssd22", 2000000000.0),
    ("io_module1", "ssd23", 2000000000.0),
    ("io_module1", "ssd24", 2000000000.0),
    ("io_module1", "ssd25", 2000000000.0),
    ("io_module1", "ssd26", 2000000000.0),
    ("io_module1", "ssd27", 2000000000.0),
    ("io_module1", "ssd28", 2000000000.0),
    ("io_module1", "ssd29", 2000000000.0),
    ("io_module1", "ssd30", 2000000000.0),
    ("io_module1", "ssd31", 2000000000.0),
    ("switch1", "io_module0", 12307692307.692308),
    ("switch1", "io_module1", 12307692307.692308),
    ("ssd16", "virtual_sink", 1000000000000000.0),
    ("ssd17", "virtual_sink", 1000000000000000.0),
    ("ssd18", "virtual_sink", 1000000000000000.0),
    ("ssd19", "virtual_sink", 1000000000000000.0),
    ("ssd20", "virtual_sink", 1000000000000000.0),
    ("virtual_source", "switch1", 1000000000000000.0)
]

# Add all edges to the graph with their respective capacities
for u, v, capacity in edges_with_capacity:
    G.add_edge(u, v, capacity=capacity)
print ("edge added")
# Compute the maximum flow using Edmonds-Karp algorithm
flow_value, flow_dict = nx.maximum_flow(G, 'virtual_source', 'virtual_sink') # flow_func=nx.algorithms.flow.edmonds_karp)
print (flow_value)