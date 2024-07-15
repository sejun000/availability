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
from enum import Enum, auto
import heapq

num_simulations = 4000000
time_period = 20 * 365 * 24
SSD_capacity = 1_000_000_000_000
SSD_speed = 200_000_000
file_path = 'availability.xlsx'
sheet_name = 'HW Architecture'
start_cell = ('B', 2)  # Corresponds to cell B2
enclosure_start_cell = ('F', 2)  # Corresponds to cell F2
availability_sheet = 'Availability'
redundancy_sheet = 'Redundancy'

"""

class RebuildType():
    def __init__(self):
        DONT_REBUILD_SSD = 0
        LOCAL_REBUILD = 1
        NETWORK_REBUILD = 2
        EXTERNAL_REBUILD = 3
        rebuild_type = DONT_REBUILD_SSD
    def set_rebuild_type(self, rebuild_type):
        # if more severe rebuild type is requested, update the rebuild type
        if (self.rebuild_type < rebuild_type):
            self.rebuild_type = rebuild_type
"""

def _calculate_max_flow(graph_structure_origin, current_failures, key):
    global max_flow_table
    graph_structure = copy.deepcopy(graph_structure_origin)
    failed_ssd = dict()
    for group_name, num_failures in current_failures.items():
        nodes, M = graph_structure.redundancy_groups[group_name]
        
        if ("ssd" in group_name and len(nodes) - M >= num_failures and num_failures > 0):
            for node in nodes:
                # all group shall be 80% 
                if (node in graph_structure.G.nodes()):
                    for u, v, edge_data in graph_structure.G.in_edges(node, data=True):
                        if 'capacity' in graph_structure.G[u][v]:
                            graph_structure.G[u][v]['capacity'] *= 0.8
        else:
            for node in nodes[:num_failures]:
                if (node in graph_structure.G.nodes()):
                    graph_structure.G.remove_node(node)
                # if first character of node is capital, it is enclosure
                if (is_enclosure(node)):
                    for modules_node in graph_structure.enclosures[node]:
                        if (modules_node in graph_structure.G.nodes()):
                            graph_structure.G.remove_node(modules_node)
    """
    for group_name, num_failures in current_failures.items():
        if ("ssd" in group_name):
            nodes, M = graph_structure.redundancy_groups[group_name]
            # locally recoverable
            if (len(nodes) - M < num_failures):
                for node in nodes[num_failures:len(nodes)]:
                    failure_ssd[node] = group_name
            for node in nodes[num_failures:len(nodes)]:
                failure_ssd[node] = group_name
    """
    flow = graph_structure.calculate_max_flow()
    
    global max_flow_table
    max_flow_table[key] = flow

def is_enclosure(node):
    return node[0].isupper()

def generate_first_failure_events(graph_structure, time_period, matched_module):
    events = []
    for node in list(graph_structure.G.nodes()):
        module = matched_module[node]
        if module:
            # generate only once
            mttf = graph_structure.mttfs[module]
            mtr = graph_structure.mtrs[module]
            failure_time = random.expovariate(1 / mttf)
            #print (module, node, mttf, mtr, current_time, failure_time, repair_time)
            heapq.heappush(events, (failure_time, 'fail', node))
    # Generate enclosure failure and repair events
    for enclosure in list(graph_structure.enclosures):
        module = matched_module[enclosure]
        if module:
            mttf = graph_structure.mttfs[module]
            mtr = graph_structure.mtrs[module]
            failure_time = random.expovariate(1 / mttf)
            #print (module, node, mttf, mtr, current_time, failure_time, repair_time)
            # failure all node in the enclosure
            heapq.heappush(events, (failure_time, 'fail', enclosure))
            #print (enclosure)
    return events

def push_repair_event(repair_events, event_node, current_time, matched_module, graph_structure):
    #print (event_node, current_time)
    module = matched_module[event_node]
    mtr = graph_structure.mtrs[module]
    if ("ssd" in module):
        repair_time = current_time + random.expovariate(1 / (SSD_capacity / (SSD_speed * 0.2) / 3600.0))
    else:
        repair_time = current_time + random.expovariate(1 / mtr)
        #print (repair_time)
    heapq.heappush(repair_events, (repair_time, 'repair', event_node))

def push_failed_event(failed_events, event_node, current_time, matched_module, graph_structure):
    module = matched_module[event_node]
    mttf = graph_structure.mttfs[module]
    failure_time = current_time + random.expovariate(1 / mttf)
    heapq.heappush(failed_events, (failure_time, 'fail', event_node))

def pop_event(events, repair_events):
    if (len(repair_events) > 0 and len(events) == 0):
        popped_event = heapq.heappop(repair_events)
    elif (len(repair_events) == 0 and len(events) > 0):
        popped_event = heapq.heappop(events)
    else:
        repair_event = repair_events[0]
        event = events[0]
        if (repair_event[0] < event[0]):
            popped_event = heapq.heappop(repair_events)
        else:
            popped_event = heapq.heappop(events)
    return popped_event

def monte_carlo_simulation(graph_structure_origin, time_period, simulation_idx):
    total_up_time = 0
    total_down_time = 0
    total_effective_up_time = 0
    global lock
    global batch_size
    global completed
    global num_simulations
    graph_structure = copy.deepcopy(graph_structure_origin)

    matched_module = {}
    SSDs = {}
    for enclosure in list(graph_structure.enclosures):
        for module in graph_structure.mttfs.keys():
            if module in enclosure:
                matched_module[enclosure] = module
                break
    for node in list(graph_structure.G.nodes()):
        for module in graph_structure.mttfs.keys():
            if module in node:
                matched_module[node] = module
                break
    for node in list(graph_structure.G.nodes()):
        if "ssd" in node:
            # speed and capacity
            # if SSDs[node][0] is less than SSD_speed, some rebuild process happens
            # if SSDs[node][1] is less than SSD_capacity, some rebuild process happens (0 means not rebuilded SSD)
            SSDs[node] = (SSD_speed, SSD_capacity)

    for local_simul in range(0, batch_size):
        # when you add / delete graph, you need to copy here
        """
        graph_structure = copy.deepcopy(graph_structure_origin)
        """
        up_time = 0
        effective_up_time = 0
        down_time = 0
        current_time = 0
        failed_events = []
        repair_events = []
        removed_in_edges = {}
        removed_out_edges = {}
        failed_node = {}
        current_failures = OrderedDict()
        for group_name, (nodes, M) in graph_structure.redundancy_groups.items():
            current_failures[group_name] = 0
        # Generate failure and repair events
        failed_events = generate_first_failure_events(graph_structure, time_period, matched_module)
        current_time = 0
        prev_time = 0
        # Process events
        while(1):    
            event_time, event_type, event_node = pop_event(failed_events, repair_events)
            # Calculate the time difference from the previous event
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

            # Handle the event
            if event_type == 'fail':
                if (event_node in failed_node):
                    failed_node[event_node] += 1
                else: # newly failed node
                    failed_node[event_node] = 1
                for group_name, (nodes, M) in graph_structure.redundancy_groups.items():
                    #print (nodes)
                    if event_node in nodes:
                        current_failures[group_name] += 1
            elif event_type == 'repair' and event_node in failed_node:
                failed_node[event_node] -= 1
                if failed_node[event_node] == 0:
                    for group_name, (nodes, M) in graph_structure.redundancy_groups.items():
                        if event_node in nodes:
                            current_failures[group_name] -= 1
                    del failed_node[event_node]
            # Push the next repair event
            if event_type == 'fail':
                push_repair_event(repair_events, event_node, event_time, matched_module, graph_structure)
            if event_type == 'repair':
                push_failed_event(failed_events, event_node, event_time, matched_module, graph_structure)
            prev_time = event_time
        # Calculate the availability
        availability = up_time / (up_time + down_time)
        total_up_time += up_time
        total_down_time += down_time
        total_effective_up_time += effective_up_time
        completed += 1
        #if (completed % 10000 == 0):
        if (simulation_idx == 0 and (completed * 1000) % batch_size == 0):
            print ("completed "+ str(completed / batch_size * 100) + "%")
    queue.put((total_up_time, total_down_time, total_effective_up_time))
    return ""

def initial_computation(graph_structure):
    global max_bw
    max_bw = graph_structure.calculate_max_flow()

if __name__ == "__main__":
    lock = threading.Lock()
    completed = 0
    max_flow_table = {}
    current_failures = OrderedDict()
    max_bw = 0

    edges, enclosures, availabilities, redundancies, mttfs, mtrs = GraphStructure.parse_input_from_excel(file_path, sheet_name, start_cell, enclosure_start_cell, availability_sheet)
    graph_structure_origin = GraphStructure(edges, enclosures, availabilities, redundancies, mttfs, mtrs)
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