import random
import pandas as pd
import networkx as nx
from graph_structure import GraphStructure

def monte_carlo_simulation(graph_structure, num_simulations, time_period):
    results = []

    for i in range(num_simulations):
        up_time = 0
        down_time = 0
        current_time = 0
        events = []
        removed_in_edges = {}
        removed_out_edges = {}

        # Generate failure and repair events
        for node in list(graph_structure.G.nodes()):
            matched_module = None
            for module in graph_structure.mttfs.keys():
                if module in node:
                    matched_module = module
                    break
            current_time = 0
            if matched_module:
                while (current_time < time_period):
                    mttf = graph_structure.mttfs[matched_module]
                    mtr = graph_structure.mtrs[matched_module]
                    failure_time = random.expovariate(1 / mttf)
                    repair_time = failure_time + random.expovariate(1 / mtr)
                    # print (mttf, mtr ,failure_time, repair_time)
                    events.append((current_time + failure_time, 'fail', node))
                    events.append((current_time + repair_time, 'repair', node))
                    current_time = current_time + repair_time

        # Sort events by time
        events.sort()
        current_time = 0
        min_flow = 99999999999
       # print(len(events))
        # Process events
        for idx, event in enumerate(events):
            event_time, event_type, node = event
            # Calculate the time difference from the previous event
            prev_time = 0
            if idx > 0:
                prev_time = events[idx - 1][0]
            
            break_flag = False
            if (event_time > time_period):
                event_time = time_period
                break_flag = True
            time_diff = event_time - prev_time
            graph_structure.add_virtual_nodes()
            for node in list(graph_structure.G.nodes()):
                if ("dfm" in node):
                    max_flow = nx.has_path(graph_structure.G, 'virtual_source', 'virtual_sink') #graph_structure.calculate_max_flow()
            graph_structure.remove_virtual_nodes()
            #if (min_flow > max_flow):
            #    min_flow = max_flow
            if max_flow == True:
                up_time += time_diff
            else:
                down_time += time_diff
            if (break_flag == True):
                break
                

            # Update current time
            current_time = event_time

            # Handle the event
            if event_type == 'fail':
                # Save the edges before removing the node
                removed_in_edges[node] = list(graph_structure.G.in_edges(node, data=True))
                removed_out_edges[node]= list(graph_structure.G.out_edges(node, data=True))
                graph_structure.G.remove_node(node)
                #print("event time : ", current_time)
                #print("removed : ", node, removed_in_edges[node], removed_out_edges[node])
                #print("maxflow : ", graph_structure.calculate_max_flow())
             #   exit(1)
            elif event_type == 'repair':
                graph_structure.G.add_node(node)
                if node in removed_in_edges:
                    for u, v, edge_data in removed_in_edges[node]:
                  #      print ("in", u, v, graph_structure.G.has_node(u), graph_structure.G.has_node(v))
                        if graph_structure.G.has_node(u) and graph_structure.G.has_node(v):
                            graph_structure.G.add_edge(u, v, **edge_data)
                    for u, v, edge_data in removed_out_edges[node]:
                 #       print ("out", u, v, graph_structure.G.has_node(u), graph_structure.G.has_node(v))
                        if graph_structure.G.has_node(u) and graph_structure.G.has_node(v):
                            graph_structure.G.add_edge(u, v, **edge_data)
                #print("event time : ", current_time)
                #print ("repair : ", node)
                #print("maxflow : ", graph_structure.calculate_max_flow())
                #exit(1)

        # Calculate the availability
        availability = up_time / (up_time + down_time)
        results.append((up_time, down_time, availability))

        # Print progress to standard output
        print(f"Simulation {i+1}/{num_simulations} completed. Total period: {time_period}, Up Time: {up_time}, Down Time: {down_time}, Availability: {availability:.8f}, MinFlow : {min_flow}")

    return results

if __name__ == "__main__":
    file_path = 'availability.xlsx'
    sheet_name = 'HW Architecture'
    start_cell = ('B', 2)  # Corresponds to cell B2
    enclosure_start_cell = ('F', 2)  # Corresponds to cell F2
    availability_sheet = 'Availability'
    num_simulations = 10000
    time_period = 10 * 365 * 24 * 60 * 60  # 10 years in seconds

    edges, enclosures, availabilities, mttfs, mtrs = GraphStructure.parse_input_from_excel(file_path, sheet_name, start_cell, enclosure_start_cell, availability_sheet)
    graph_structure = GraphStructure(edges, enclosures, availabilities)
    graph_structure.mttfs = mttfs
    graph_structure.mtrs = mtrs

    results = monte_carlo_simulation(graph_structure, num_simulations, time_period)

    with open("monte_carlo.output", "w") as f:
        for up_time, down_time, availability in results:
            f.write(f"Up Time: {up_time}, Down Time: {down_time}, Availability: {availability:.8f}\n")
