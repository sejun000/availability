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
import logging

num_simulations = 4000000
time_period = 20 * 365 * 24
SSD_capacity = 64_000_000_000_000
SSD_speed = 200_000_000
near_zero_value = 0.0000000000001
network_redundancy = False
network_only = False
rebuild_overhead = 0.2
file_path = 'availability_single_switched_hdd.xlsx'
sheet_name = 'HW Architecture'
start_cell = ('B', 2)  # Corresponds to cell B2
enclosure_start_cell = ('F', 2)  # Corresponds to cell F2
availability_sheet = 'Availability'
redundancy_sheet = 'Redundancy'

def _calculate_connected_ssd(graph_structure_origin, key, failed_node):
    global disconnected_ssd_table
    if (key in disconnected_ssd_table):
        return
    graph_structure = copy.deepcopy(graph_structure_origin)
    for node in failed_node:
        if (not "ssd" in node):
            if (node in graph_structure.G.nodes()):
                graph_structure.G.remove_node(node)
            # if first character of node is capital, it is enclosure
            if (is_enclosure(node)):
                for modules_node in graph_structure.enclosures[node]:
                    if (not "ssd" in modules_node and modules_node in graph_structure.G.nodes()):
                        graph_structure.G.remove_node(modules_node)
    graph_structure.add_virtual_nodes()
    disconnected_ssd = {}
    for node in graph_structure.G.nodes():
        if ("ssd" in node):
            connected = nx.has_path(graph_structure.G, 'virtual_source', node)
            if (connected == False):
                # If controller is failed, SSD shall be rebuilded
                disconnected_ssd[node] = 1
    disconnected_ssd_table[key] = disconnected_ssd
                    

def _calculate_max_flow(graph_structure_origin, current_failures, key, failed_node, failed_events, repair_events, matched_module, matched_group, matched_network_group, event_time = 0):
    global max_flow_table
    global max_flow_for_rebuild_table
    global max_flow_for_network_rebuild_table
    if (key in max_flow_table):
        return
    graph_structure = copy.deepcopy(graph_structure_origin)
    rebuilding_flow = dict() # group to rebuild bandwidth mapping
    network_rebuild_flow = dict()
    ssd_under_rebuilding = dict()
    skip_failure_group = dict()
    data_loss = False
    
    for node in failed_node:
        if (not "ssd" in node):
            if (node in graph_structure.G.nodes()):
                graph_structure.G.remove_node(node)
            # if first character of node is capital, it is enclosure
            if (is_enclosure(node)):
                for modules_node in graph_structure.enclosures[node]:
                    if (not "ssd" in modules_node and modules_node in graph_structure.G.nodes()):
                        graph_structure.G.remove_node(modules_node)

    # configure edge's capacity except network_level rebuild case
    if (network_only == False):
        for group_name, num_failures in current_failures.items():
            if ("disconnected" in group_name):
                continue
            nodes, M = graph_structure.redundancy_groups[group_name]
            # local rebuild is possible
            if ("ssd" in group_name and len(nodes) - M >= num_failures and num_failures > 0 and current_failures[get_disconnected_name(group_name)] == 0):
                skip_failure_group[group_name] = 1
                idx = 0
                for node in nodes:
                    # all group shall be 80% 
                    if (node in graph_structure.G.nodes()):
                        for u, v, edge_data in graph_structure.G.in_edges(node, data=True):
                            if 'capacity' in graph_structure.G[u][v]:
                                ssd_under_rebuilding[node] = 1
                                # if controller is exist
                                if (idx == 0): 
                                    if (group_name in rebuilding_flow):
                                        rebuilding_flow[group_name] += rebuild_overhead * graph_structure.G[u][v]['capacity']    
                                    else:
                                        rebuilding_flow[group_name] = rebuild_overhead * graph_structure.G[u][v]['capacity']
                                graph_structure.G[u][v]['capacity'] *= (1 - rebuild_overhead)
                        idx += 1
                    else:
                        logging.critical("node is not in the graph", node)

    # if network level rebuilding is necessary, get maximum flow
    graph_structure.add_virtual_nodes()
   
    for group_name, num_failures in current_failures.items():
        if ("disconnected" in group_name):
            continue
        if ("ssd" in group_name and current_failures[get_disconnected_name(group_name)] > 0):
            data_loss = True
        if ("ssd" in group_name and num_failures > 0 and current_failures[get_disconnected_name(group_name)] == 0):
            nodes, M = graph_structure.redundancy_groups[get_original_group_name(group_name)]
            for node in nodes:
                # it can be already rebuild with local rebuild, ignore, and just rebuild with local rebuild
                if node in ssd_under_rebuilding:
                    break
                data_loss = True
                flow_value, flow_dict = nx.maximum_flow(graph_structure.G, 'virtual_source', node)
                rebuilding_flow_each = flow_value * rebuild_overhead
                for u in flow_dict:
                    for v, flow in flow_dict[u].items():
                        if flow > 0:
                            rebuilding_flow_each = min(rebuilding_flow_each, graph_structure.G[u][v]['capacity'])
                for u in flow_dict:
                    for v, flow in flow_dict[u].items():
                        if flow > 0:
                            graph_structure.G[u][v]['capacity'] -= rebuilding_flow_each
                            #print (u, v, graph_structure.G[u][v]['capacity'], graph_structure_origin.G[u][v]['capacity'], flow_value * 0.2)
                network_rebuild_flow[node] = rebuilding_flow_each
                # if network level rebuild cannot recover the network, remove related nodes
    flow = 0
    if (data_loss == True and network_redundancy == False):
        flow = 0
    else:
        flow = graph_structure.calculate_max_flow()
        #print (failed_node)
        #print (flow)
    max_flow_table[key] = flow
    max_flow_for_rebuild_table[key] = rebuilding_flow
    max_flow_for_network_rebuild_table[key] = network_rebuild_flow

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
            heapq.heappush(events, (failure_time, 'fail', node, 0))
    # Generate enclosure failure and repair events
    for enclosure in list(graph_structure.enclosures):
        module = matched_module[enclosure]
        if module:
            mttf = graph_structure.mttfs[module]
            mtr = graph_structure.mtrs[module]
            failure_time = random.expovariate(1 / mttf)
            #print (module, node, mttf, mtr, current_time, failure_time, repair_time)
            # failure all node in the enclosure
            heapq.heappush(events, (failure_time, 'fail', enclosure, 0))
            #print (enclosure)
    return events

def update_repair_event(repair_events, current_time, matched_module, matched_group, graph_structure, key, SSDs, disconnected_ssds):
    global max_flow_for_rebuild_table
    updated_repair_events = []
    while(1):
        if (len(repair_events) == 0): break
        popped_event = heapq.heappop(repair_events)
        repair_time, repair_event, repair_node, prev_time = popped_event
        if (not "ssd" in repair_node):
            updated_repair_events.append(popped_event)
            continue
        else:
            if (repair_node in disconnected_ssds):
                # disconnected ssd cannot be repaired
                # do not enqueue and continue
                continue
            module = matched_module[repair_node]
            mtr = graph_structure.mtrs[module]
            group = matched_group[repair_node]
            if (key in max_flow_for_rebuild_table):
                if (group in max_flow_for_rebuild_table[key]):
                    rebuild_bandwidth = max_flow_for_rebuild_table[key][group]
                    if (rebuild_bandwidth == 0):
                        logging.critical("warning 1 update_repair_events !!!")
                        updated_repair_time = current_time + 1 / near_zero_value
                    else:
                        remaining_capacity = SSDs[repair_node][1] - (current_time - prev_time) * SSDs[repair_node][0] * 3600
                        updated_repair_time = current_time + remaining_capacity / rebuild_bandwidth / 3600.0
                        #print (repair_node, current_time, repair_time, updated_repair_time, SSDs[repair_node][0], rebuild_bandwidth)
                        SSDs[repair_node] = (rebuild_bandwidth, remaining_capacity)
                elif (repair_node in max_flow_for_network_rebuild_table[key]):
                    # network level rebuild
                    rebuild_bandwidth = max_flow_for_network_rebuild_table[key][repair_node]
                    if (rebuild_bandwidth == 0):
                        logging.critical("warning 2 update_repair_events !!!")
                        updated_repair_time = current_time + 1 / near_zero_value
                    else:
                        remaining_capacity = SSDs[repair_node][1] - (current_time - prev_time) * SSDs[repair_node][0] * 3600
                        updated_repair_time = current_time + remaining_capacity / rebuild_bandwidth / 3600.0
                        #print (repair_node, current_time, repair_time, updated_repair_time, SSDs[repair_node][0], rebuild_bandwidth)
                        SSDs[repair_node] = (rebuild_bandwidth, remaining_capacity)
                        logging.info (repair_node, current_time, repair_time, updated_repair_time)
                else:
                    # rebuild is not possible, data loss, so, we just switch SSDs
                    logging.info (repair_node, "-----------", disconnected_ssds)
                    logging.info (max_flow_for_rebuild_table[key], max_flow_for_network_rebuild_table[key])
                    logging.critical("warning 3 update_repair_events !!!")
                    updated_repair_time = current_time + 1 / near_zero_value
            updated_repair_events.append((updated_repair_time, repair_event, repair_node, current_time))
            #print (repair_node, current_time, repair_time, updated_repair_time, case_flag)
    for event in updated_repair_events:
        heapq.heappush(repair_events, event)

def push_repair_event(repair_events, event_node, current_time, matched_module, matched_group, graph_structure, key, SSDs, disconnected_ssds, failed_node):
    global max_flow_for_rebuild_table
    module = matched_module[event_node]
    mtr = graph_structure.mtrs[module]
    if ("ssd" in module):
        if (event_node in disconnected_ssds):
                # disconnected ssd cannot be repaired
                # do not enqueue and continue
                return
        group = matched_group[event_node]
        rebuild_bandwidth = 0
        if (key in max_flow_for_rebuild_table):
            if (group in max_flow_for_rebuild_table[key]):
                rebuild_bandwidth = max_flow_for_rebuild_table[key][group]
                if (rebuild_bandwidth == 0):
                    logging.critical ("warning 1 repaired_event !!!")
                    repair_time = current_time + 1 / near_zero_value
                else:
                    repair_time = current_time + SSD_capacity / rebuild_bandwidth / 3600.0
            elif (event_node in max_flow_for_network_rebuild_table[key]):
                # network level rebuild
                rebuild_bandwidth = max_flow_for_network_rebuild_table[key][event_node]
                if (rebuild_bandwidth == 0):
                    logging.critical ("warning 2 repaired_event !!!")
                    repair_time = current_time + 1 / near_zero_value
                else:
                    repair_time = current_time + SSD_capacity / rebuild_bandwidth / 3600.0
                    #print (SSD_capacity / rebuild_bandwidth / 3600.0)
            else:
                logging.info (event_node, "-----------", disconnected_ssds)
                logging.info (max_flow_for_rebuild_table[key], max_flow_for_network_rebuild_table[key], failed_node)
                logging.critical ("warning 3 repaired_event !!!")
                repair_time = current_time + random.expovariate(1 / mtr)
                
            SSDs[event_node] = (rebuild_bandwidth, SSD_capacity)
        else:
            assert False
    else:
        repair_time = current_time + random.expovariate(1 / mtr)
    heapq.heappush(repair_events, (repair_time, 'repair', event_node, current_time))

def push_failed_event_now(failed_events, event_node, current_time):
    heapq.heappush(failed_events, (current_time, 'fail', event_node, current_time))

def push_failed_event(failed_events, event_node, current_time, matched_module, graph_structure):
    module = matched_module[event_node]
    mttf = graph_structure.mttfs[module]
    failure_time = current_time + random.expovariate(1 / mttf)
    heapq.heappush(failed_events, (failure_time, 'fail', event_node, current_time))

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

def count_failed_node(event_type, event_node, failed_node, current_failures_except_ssd, matched_group):
    if event_type == 'fail':
        if (event_node in failed_node):
            #print ("+++++++++++++++", failed_node)
            failed_node[event_node] += 1
        else: # newly failed node
            failed_node[event_node] = 1
            group_name = matched_group[event_node]
            if (not "ssd" in group_name):
                current_failures_except_ssd[group_name] += 1
    elif event_type == 'repair' and event_node in failed_node:
        failed_node[event_node] -= 1
        if failed_node[event_node] == 0:
            group_name = matched_group[event_node]
            if (not "ssd" in group_name):
                current_failures_except_ssd[group_name] -= 1
            #print ("-------------", failed_node)
            del failed_node[event_node]

def count_current_failures(graph_structure, disconnected_ssds, failed_node, current_failures, matched_group):
    for group_name, (nodes, M) in graph_structure.redundancy_groups.items():
        current_failures[group_name] = 0
        if ("ssd" in group_name):
            current_failures[get_disconnected_name(group_name)] = 0
    for node in failed_node:
        group_name = matched_group[node]
        current_failures[group_name] += 1
    for ssd in disconnected_ssds:
        group_name = get_disconnected_name(matched_group[ssd])
        current_failures[group_name] += 1


def get_disconnected_name(group_name):
    return group_name + "_disconnected"

def get_original_group_name(group_name):
    if ("_disconnected" in group_name):
        return group_name[:len(group_name) - len("_disconnected")]
    return group_name


def monte_carlo_simulation(graph_structure_origin, time_period, simulation_idx):
    total_up_time = 0
    total_down_time = 0
    total_effective_up_time = 0
    global lock
    global batch_size
    global completed
    global num_simulations
    global max_flow_table
    global max_flow_for_rebuild_table
    global max_flow_for_network_rebuild_table
    global disconnected_ssd_table

    graph_structure = copy.deepcopy(graph_structure_origin)

    matched_module = {}
    matched_group = {}
    matched_network_group = OrderedDict()
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
    for group, (nodes, M) in graph_structure.redundancy_groups.items():
        if ("network_level" in group):
            for node in nodes:
                matched_network_group[node] = group
        else:
            for node in nodes:
                matched_group[node] = group
    #print(matched_network_group)
    #assert False
    for node in list(graph_structure.G.nodes()):
        if "ssd" in node:
            # speed and capacity
            # if SSDs[node][0] is less than SSD_speed, some rebuild process happens
            # if SSDs[node][1] is less than SSD_capacity, some rebuild process happens (0 means not rebuilded SSD)
            SSDs[node] = (SSD_speed, 0)

    for local_simul in range(0, batch_size):
        # when you add / delete graph, you need to copy here
        """
        graph_structure = copy.deepcopy(graph_structure_origin)
        """
        up_time = 0
        effective_up_time = 0
        down_time = 0
        failed_events = []
        repair_events = []
        prev_disconnected_ssds = OrderedDict()
        failed_node = OrderedDict()
        current_failures = OrderedDict()
        current_failures_except_ssd = OrderedDict()
        for group_name, (nodes, M) in graph_structure.redundancy_groups.items():
            current_failures[group_name] = 0
            if (not "ssd" in group_name):
                current_failures_except_ssd[group_name] = 0
            if ("ssd" in group_name):
                current_failures[get_disconnected_name(group_name)] = 0

        # Generate failure and repair events
        failed_events = generate_first_failure_events(graph_structure, time_period, matched_module)
        current_time = 0
        prev_time = 0
        # maximum flow for initial state
        key1 = frozenset(current_failures_except_ssd.items())
        key2 = frozenset(current_failures.items())
        _calculate_connected_ssd(graph_structure, key1, failed_node)
        _calculate_max_flow(graph_structure, current_failures, key2, failed_node, failed_events, repair_events, matched_module, matched_group, matched_network_group)

        # Process events
        while(1):    
            event_time, event_type, event_node, prev_time = pop_event(failed_events, repair_events)
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
            coeff = 1.0
            
            max_flow = max_flow_table[key2]
            if (max_flow <= 0):
                connected = False
            else:
                coeff = max_flow / max_bw
            """
            if (network_redundancy == True):
                for group, (nodes, M) in graph_structure.redundancy_groups.items():
                    if current_failures[group] > len(nodes) - M:
                        # print(group, current_failures[group], len(nodes) - M)
                        connected = False
                        break
            """
            if connected == True:
                up_time += time_diff
                effective_up_time += time_diff * coeff
            else:
                down_time += time_diff
            if (break_flag == True):
                break

            # Handle the event
            count_failed_node(event_type, event_node, failed_node, current_failures_except_ssd, matched_group)
            # Update the maximum flow if necessary

            key1 = frozenset(current_failures_except_ssd.items())
            _calculate_connected_ssd(graph_structure, key1, failed_node)
            
            for ssd in prev_disconnected_ssds:
                if ssd not in disconnected_ssds:
                    count_failed_node('fail', ssd, failed_node, current_failures_except_ssd, matched_group)
            

            disconnected_ssds = disconnected_ssd_table[key1]
            count_current_failures(graph_structure, disconnected_ssds, failed_node, current_failures, matched_group)
            key2 = frozenset(current_failures.items())
            _calculate_max_flow(graph_structure, current_failures, key2, failed_node, failed_events, repair_events, matched_module, matched_group, matched_network_group, event_time)

            if (not "ssd" in event_node):
                update_repair_event(repair_events, event_time, matched_module, matched_group, graph_structure, key2, SSDs, disconnected_ssds)
            # Push the next repair event
            if event_type == 'fail':
                push_repair_event(repair_events, event_node, event_time, matched_module, matched_group, graph_structure, key2, SSDs, disconnected_ssds, failed_node)
            if event_type == 'repair':
                push_failed_event(failed_events, event_node, event_time, matched_module, graph_structure)
            
            for ssd in prev_disconnected_ssds:
                if ssd not in disconnected_ssds:
                    # needs to be rebuild
                    push_failed_event_now(failed_events, ssd, event_time)
            
            prev_time = event_time
            prev_disconnected_ssds = copy.deepcopy(disconnected_ssds)
            #print (prev_disconnected_ssds)
        # Calculate the availability
        availability = up_time / (up_time + down_time)
        total_up_time += up_time
        total_down_time += down_time
        total_effective_up_time += effective_up_time
        completed += 1
        #if (completed % 10000 == 0):
        if (simulation_idx == 0 and (completed * 100) % batch_size == 0):
            print ("completed "+ str(completed * 100 // batch_size) + "%")
    queue.put((total_up_time, total_down_time, total_effective_up_time))
    return ""

def initial_computation(graph_structure):
    global max_bw
    max_bw = graph_structure.calculate_max_flow()

if __name__ == "__main__":
    lock = threading.Lock()
    completed = 0
    max_flow_table = {}
    max_flow_for_rebuild_table = {}
    disconnected_ssd_table = OrderedDict()
    max_flow_for_network_rebuild_table = {}
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