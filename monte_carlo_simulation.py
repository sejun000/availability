import random
import pandas as pd
import networkx as nx
from graph_structure import GraphStructure
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import os
import threading
import multiprocessing

def monte_carlo_simulation(graph_structure_origin, time_period, simulation_idx):
    total_up_time = 0
    total_down_time = 0
    global lock
    global batch_size
    global completed
    global num_simulations
    graph_structure = copy.deepcopy(graph_structure_origin)
    for local_simul in range(0, batch_size):
        # when you add / delete graph, you need to copy here
        """
        graph_structure = copy.deepcopy(graph_structure_origin)
        """
        up_time = 0
        down_time = 0
        current_time = 0
        events = []
        removed_in_edges = {}
        removed_out_edges = {}
        failed_node = {}

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
                    failure_time = current_time + random.expovariate(1 / mttf)
                    repair_time = failure_time + random.expovariate(1 / mtr)
                    #print (matched_module, node, mttf, mtr, current_time, failure_time, repair_time)
                    events.append((failure_time, 'fail', node))
                    events.append((repair_time, 'repair', node))
                    current_time = repair_time
        # Sort events by time
        events.sort()
        #for time, event, node in events:
        #    print (time, event, node)
        current_time = 0
        # Process events
        for idx, event in enumerate(events):
            #print (event)
            event_time, event_type, event_node = event
            # Calculate the time difference from the previous event
            prev_time = 0
            if idx > 0:
                prev_time = events[idx - 1][0]
            break_flag = False
            if (event_time > time_period):
                event_time = time_period
                break_flag = True
            time_diff = event_time - prev_time
            """
            graph_structure.add_virtual_nodes()
            """
            connected = True
            """
            for module, (M, K) in graph_structure.redundancies.items(): 
                if ("dfm" == module):
                    for group_name, nodes in graph_structure.redundancy_groups.items():
                        if ("dfm" in group_name):
                            connected_count = 0
                            for node in nodes:
                                if (graph_structure.G.has_node(node) and nx.has_path(graph_structure.G, 'virtual_source', node)):
                                    connected_count += 1
                            if (connected_count < M):
                                connected = False
                            #print (connected_count)
                    break
            """
            for group_name, (nodes, M) in graph_structure.redundancy_groups.items():
                connected_count = 0
                for node in nodes:
                    if (not node in failed_node):
                        connected_count += 1
                #print (failed_node, group_name, M, connected_count)
                if (connected_count < M):
                    connected = False
                        

            """
            graph_structure.remove_virtual_nodes()
            """
            if connected == True:
                up_time += time_diff
            else:
                down_time += time_diff
            if (break_flag == True):
                break
                

            # Update current time
            current_time = event_time

            # Handle the event
            if event_type == 'fail':
                failed_node[event_node] = 1
            elif event_type == 'repair' and event_node in failed_node:
                del failed_node[event_node]
            """
            # remove and repair node in Graph
            if event_type == 'fail':
                # Save the edges before removing the node
                removed_in_edges[node] = list(graph_structure.G.in_edges(node, data=True))
                removed_out_edges[node]= list(graph_structure.G.out_edges(node, data=True))
                if graph_structure.G.has_node(node):
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
            """

        # Calculate the availability
        availability = up_time / (up_time + down_time)
        total_up_time += up_time
        total_down_time += down_time
        completed += 1
        #if (completed % 10000 == 0):
        print ("completed "+ str(completed / num_simulations) + "%")
    #    results.append((up_time, down_time, availability))
        # Print progress to standard output
        #print(f"Simulation {simulation_idx+1} completed. Total period: {time_period}, Up Time: {up_time}, Down Time: {down_time}, Availability: {availability:.8f}, MinFlow : {min_flow}")
    queue.put((total_up_time, total_down_time))
    return ""

if __name__ == "__main__":
    lock = threading.Lock()
    completed = 0
    num_simulations = 40
    time_period = 2000000 * 365 * 24 
    file_path = 'availability.xlsx'
    sheet_name = 'HW Architecture'
    start_cell = ('B', 2)  # Corresponds to cell B2
    enclosure_start_cell = ('F', 2)  # Corresponds to cell F2
    availability_sheet = 'Availability'
    redundancy_sheet = 'Redundancy'

    edges, enclosures, availabilities, redundancies, mttfs, mtrs = GraphStructure.parse_input_from_excel(file_path, sheet_name, start_cell, enclosure_start_cell, availability_sheet)
    graph_structure_origin = GraphStructure(edges, enclosures, availabilities, redundancies, mttfs, mtrs)
    # Determine the number of CPU cores
    procs = 40  #os.cpu_count()
    batch_size = (num_simulations + procs - 1) // procs
    jobs = []
    queue = multiprocessing.Queue()
    for i in range(0, procs):
    #for i in range(0, 1):
        out_list = list()
        process = multiprocessing.Process(target=monte_carlo_simulation, 
                                          args=(graph_structure_origin, time_period, i))
        jobs.append(process)

    # Start the processes (i.e. calculate the random number lists)      
    for j in jobs:
        j.start()

    # Ensure all of the processes have finished
    for j in jobs:
        j.join()
    results = []
    while not queue.empty():
        results.append(queue.get())
    total_up_time = 0
    total_down_time = 0
    for up_time, down_time in results:
        total_up_time += up_time
        total_down_time += down_time
    print (total_up_time, total_down_time, total_up_time / (total_up_time+total_down_time))
    #with open("monte_carlo.output", "w") as f:
    #    for up_time, down_time, availability in results:
    #        f.write(f"Up Time: {up_time}, Down Time: {down_time}, Availability: {availability:.8f}\n")
