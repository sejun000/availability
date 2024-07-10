import random
import pandas as pd
import networkx as nx
from graph_structure import GraphStructure
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import os
import threading
import multiprocessing
import itertools
import time
from collections import OrderedDict

num_simulations = 4000000
time_period = 20 * 365 * 24
file_path = 'availability.xlsx'
sheet_name = 'HW Architecture'
start_cell = ('B', 2)  # Corresponds to cell B2
enclosure_start_cell = ('F', 2)  # Corresponds to cell F2
availability_sheet = 'Availability'
redundancy_sheet = 'Redundancy'

def _calculate_max_flow(graph_structure, current_failures, key):
    global max_flow_table
    graph_structure = copy.deepcopy(graph_structure_origin)
    for group_name, num_failures in current_failures.items():
        nodes, M = graph_structure.redundancy_groups[group_name]
        for node in nodes[:num_failures]:
            if (node in graph_structure.G.nodes()):
                graph_structure.G.remove_node(node)
            # if first character of node is capital, it is enclosure
            if (is_enclosure(node)):
                for modules_node in graph_structure.enclosures[node]:
                    if (modules_node in graph_structure.G.nodes()):
                        graph_structure.G.remove_node(modules_node)

    flow = graph_structure.calculate_max_flow()
    
    global max_flow_table
    max_flow_table[key] = flow

def monte_carlo_simulation(graph_structure_origin, time_period, simulation_idx):
    total_up_time = 0
    total_down_time = 0
    total_effective_up_time = 0
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
        effective_up_time = 0
        down_time = 0
        current_time = 0
        events = []
        removed_in_edges = {}
        removed_out_edges = {}
        failed_node = {}
        current_failures = OrderedDict()
        for group_name, (nodes, M) in graph_structure.redundancy_groups.items():
            current_failures[group_name] = 0

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

        # Generate enclosure failure and repair events
        for enclosure in list(graph_structure.enclosures):
            matched_module = None
            for module in graph_structure.mttfs.keys():
                if module in enclosure:
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
                    for node in enclosure:
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
            key = frozenset(current_failures.items())
            coeff = 1.0
            if not key in max_flow_table:
                _calculate_max_flow(graph_structure, current_failures, key)
            
            max_flow = max_flow_table[key]
            if (max_flow <= 0):
                connected = False
            else:
                coeff = max_flow / max_bw

            for group, (nodes, M) in graph_structure.redundancy_groups.items():
                if current_failures[group] > len(nodes) - M:
                    # print(group, current_failures[group], len(nodes) - M)
                    connected = False
                    break
            if connected == True:
                up_time += time_diff
                effective_up_time += time_diff * coeff
            else:
                down_time += time_diff
            if (break_flag == True):
                break

            # Update current time
            current_time = event_time

            # Handle the event
            if event_type == 'fail':
                for group_name, (nodes, M) in graph_structure.redundancy_groups.items():
                    #print (nodes)
                    if event_node in nodes:
                        current_failures[group_name] += 1
                failed_node[event_node] = 1
            elif event_type == 'repair' and event_node in failed_node:
                for group_name, (nodes, M) in graph_structure.redundancy_groups.items():
                    if event_node in nodes:
                        current_failures[group_name] -= 1
                del failed_node[event_node]
        # Calculate the availability
        availability = up_time / (up_time + down_time)
        total_up_time += up_time
        total_down_time += down_time
        total_effective_up_time += effective_up_time
        completed += 1
        #if (completed % 10000 == 0):
        if (simulation_idx == 0 and (completed * 1000) % batch_size == 0):
            print ("completed "+ str(completed / batch_size * 100) + "%")
    #    results.append((up_time, down_time, availability))
        # Print progress to standard output
        #print(f"Simulation {simulation_idx+1} completed. Total period: {time_period}, Up Time: {up_time}, Down Time: {down_time}, Availability: {availability:.8f}, MinFlow : {max_flow}")
    queue.put((total_up_time, total_down_time, total_effective_up_time))
    return ""

def is_enclosure(node):
    return node[0].isupper()

redundancy_group_dependency = []
def explore_failures_recursive(graph_structure_origin, groups, current_group_index=0):
    group_name, (nodes, M) = groups[current_group_index]
    N = len(nodes)
    global current_failures
    global partial_failure_mark
    global cnt
    for num_failures in range(N - M + 1):
       # print (num_failures, group_name)
        current_failures[group_name] = num_failures
        if current_group_index == len(groups) - 1:
            # Calculate the flow for the current configuration
            
            graph_structure = copy.deepcopy(graph_structure_origin)
            for group_name, num_failures in current_failures.items():
                nodes, M = graph_structure.redundancy_groups[group_name]
                for node in nodes[:num_failures]:
                    if (node in graph_structure.G.nodes()):
                        graph_structure.G.remove_node(node)
                    # if first character of node is capital, it is enclosure
                    if (is_enclosure(node)):
                        for modules_node in graph_structure.enclosures[node]:
                            if (modules_node in graph_structure.G.nodes()):
                                graph_structure.G.remove_node(modules_node)

            flow = graph_structure.calculate_max_flow()
            
            global max_flow_table
            key = frozenset(current_failures.items())
        #    print (key)
            max_flow_table[key] = flow
        else:
            explore_failures_recursive(graph_structure_origin, groups, current_group_index + 1)

def initial_computation(graph_structure_origin):
    global max_bw
    max_bw = graph_structure_origin.calculate_max_flow()
    graph_structure = copy.deepcopy(graph_structure_origin)
    groups = list(graph_structure.redundancy_groups.items())
    start_time = time.time()
   # explore_failures_recursive(graph_structure, groups)
    end_time = time.time()
    print ("elapsed time : ", end_time - start_time)


if __name__ == "__main__":
    lock = threading.Lock()
    completed = 0
    max_flow_table = {}
    current_failures = OrderedDict()
    max_bw = 0

    edges, enclosures, availabilities, redundancies, mttfs, mtrs = GraphStructure.parse_input_from_excel(file_path, sheet_name, start_cell, enclosure_start_cell, availability_sheet)
    graph_structure_origin = GraphStructure(edges, enclosures, availabilities, redundancies, mttfs, mtrs)
    # Initial computation
    print ("Initial computation....")
    initial_computation(graph_structure_origin)

    # Reset the current failures
    current_failures = OrderedDict()
    # Determine the number of CPU cores
    procs = 40  #os.cpu_count()
    batch_size = (num_simulations + procs - 1) // procs
    jobs = []
    queue = multiprocessing.Queue()
    for i in range(0, procs):
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
    total_effective_up_time = 0
    print (graph_structure_origin.enclosures)
    for up_time, down_time, effective_up_time in results:
        total_up_time += up_time
        total_down_time += down_time
        total_effective_up_time += effective_up_time
    print ("Total Up Time : ", total_up_time)
    print ("Total Down Time : ", total_down_time)
    print ("Total Effective Up Time : ", total_effective_up_time)
    print ("Availability : ", total_up_time / (total_up_time + total_down_time))
    print ("Effective Availability : ", total_effective_up_time / (total_up_time + total_down_time))
    #with open("monte_carlo.output", "w") as f:
    #    for up_time, down_time, availability in results:
    #        f.write(f"Up Time: {up_time}, Down Time: {down_time}, Availability: {availability:.8f}\n")
