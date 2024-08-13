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
import argparse
import math

def parse_arguments():
    parser = argparse.ArgumentParser(description='Monte Carlo simulation parameters.')
    parser.add_argument('--num_simulations', type=int, default=4000000, help='Number of simulations to run')
    parser.add_argument('--time_period', type=int, default=20 * 365 * 24, help='Time period for the simulation in hours')
    parser.add_argument('--SSD_capacity', type=int, default=64_000_000_000_000, help='SSD capacity in bytes')
    parser.add_argument('--SSD_speed', type=int, default=200_000_000, help='SSD speed in bytes per second')
    parser.add_argument('--rebuild_overhead', type=float, default=0.2, help='Rebuild overhead as a fraction')
    parser.add_argument('--SSD_m', type=int, default=0, help='number of data chunks (declustered parity)')
    parser.add_argument('--SSD_M', type=int, default=0, help='number of SSD data chunks')
    parser.add_argument('--SSD_K', type=int, default=0, help='number of SSD parity chunks')
    parser.add_argument('--network_m', type=int, default=0, help='number of network data chunks (declustered parity)')
    parser.add_argument('--network_M', type=int, default=1, help='number of network data chunks')
    parser.add_argument('--network_K', type=int, default=0, help='number of network parity chunks')
    parser.add_argument('--generate_fault_injection', action='store_true', help='generate fault injection file, this is only available when network_K is 0')
    parser.add_argument('--total_nodes', type=int, default=1, help='number of network nodes, it shall be M * K\'s multiple')
    parser.add_argument('--network_only', action='store_true', help='Flag to indicate if only network level rebuild is considered')
    parser.add_argument('--file_path', type=str, default='availability_single_switched_hdd.xlsx', help='Path to the Excel file')
    parser.add_argument('--SSD_type', type=str, default="tlc", help='SSD type, e.g., tlc, qlc.')
    parser.add_argument('--SSD_write_percentage', type=int, default=0, help='SSD write percentage')
    parser.add_argument('-o', '--output', type=str, help='Output file path to save results')
    parser.add_argument('--flow_cdf', type=str, default='flow_cdf.txt', help='File to save flow CDF')
    args = parser.parse_args()
    return args

args = parse_arguments()

num_simulations = args.num_simulations
time_period = args.time_period
SSD_capacity = args.SSD_capacity
SSD_speed = args.SSD_speed
SSD_type = args.SSD_type
flow_cdf_file = args.flow_cdf

if (not SSD_type == "tlc" and not SSD_type == "qlc"):
    print ("SSD type is not supported")
    assert False
SSD_write_percentage = args.SSD_write_percentage
if (SSD_write_percentage < 0 or SSD_write_percentage > 100):
    print ("SSD write percentage is not in the range of 0 ~ 100")
    assert False
SSD_write_mttf_per_1tb = 10128 # for 200 MB/s writes
if (SSD_type == "qlc"):
    SSD_write_mttf_per_1tb = 2640

rebuild_overhead = args.rebuild_overhead
network_M = args.network_M
network_K = args.network_K
ssd_M = 0
ssd_K = 0
ssd_m = 0
network_m = 0
check_repair_time_for_ssd_switching = True

if args.SSD_M:
    ssd_M = args.SSD_M
if args.SSD_K:
    ssd_K = args.SSD_K
if args.SSD_m:
    ssd_m = args.SSD_m
if args.network_m:
    network_m = args.network_m

if (network_m > network_M + network_K):
    print ("network_m is larger than the sum of network_M and network_K")
    assert False

total_nodes = args.total_nodes
if (total_nodes == 1):
    total_nodes = network_M + network_K
if (total_nodes % (network_M + network_K) != 0):
    print ("total_nodes is not multiple of M + K, total_nodes : ", total_nodes, "M + K : ", network_M + network_K)
    assert False

network_redundancy = network_M + network_K
network_only = args.network_only
file_path = args.file_path
output_file = args.output
network_injection_file = 'network_injection.txt'

near_zero_value = 0.00000000000000000001
input_ssd_availability = -1
network_availability = -1
sheet_name = 'HW Architecture'
start_cell = ('B', 2)  # Corresponds to cell B2
enclosure_start_cell = ('F', 2)  # Corresponds to cell F2
availability_sheet = 'Availability'
redundancy_sheet = 'Redundancy'
generate_fault_injection = args.generate_fault_injection
if (network_K > 0):
    print ("network_K is larger than 0, so fault injection is not supported")
    generate_fault_injection = False

def combinations_count(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

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
    global input_ssd_availability
    global network_availability
    global network_availability_if_im_alive
    global network_availability_table
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
                    rebuild_speed_up = 1
                    if (ssd_m > 0):
                        if (ssd_m > len(nodes)):
                            logging.critical("ssd_m is larger than the number of nodes in the group")
                            assert False
                        rebuild_speed_up = ssd_m + (len(nodes) - M) - 1
                    if (node in graph_structure.G.nodes()):
                        for u, v, edge_data in graph_structure.G.in_edges(node, data=True):
                            if 'capacity' in graph_structure.G[u][v]:
                                # if controller is exist
                                if (idx == 0):
                                    #print(group_name)
                                    if (group_name in rebuilding_flow):
                                        rebuilding_flow[group_name] += rebuild_overhead * rebuild_speed_up * graph_structure.G[u][v]['capacity']    
                                    else:
                                        rebuilding_flow[group_name] = rebuild_overhead * rebuild_speed_up * graph_structure.G[u][v]['capacity']
                                if (ssd_m > 0):
                                    graph_structure.G[u][v]['capacity'] -=  rebuild_overhead * 2 * graph_structure.G[u][v]['capacity']
                                else:
                                    graph_structure.G[u][v]['capacity'] -= rebuild_overhead * graph_structure.G[u][v]['capacity']
                        #print ("local : ", u, v, graph_structure.G[u][v]['capacity'], graph_structure_origin.G[u][v]['capacity'], rebuilding_flow[group_name])
                        idx += 1
                    else:
                        logging.critical("node is not in the graph", node)
    # if network level rebuilding is necessary, get maximum flow
    graph_structure.add_virtual_source()
    total_network_availability = 0
    total_network_availability_if_im_alive = 0
    for group_name, num_failures in current_failures.items():
        if ("disconnected" in group_name):
            continue
        if ("ssd" in group_name and current_failures[get_disconnected_name(group_name)] > 0):
            total_network_availability += current_failures[get_disconnected_name(group_name)]
            #total_network_availability *= (network_availability ** current_failures[get_disconnected_name(group_name)])
            data_loss = True
        if (group_name in skip_failure_group):
            continue
        if ("ssd" in group_name and num_failures == 0 and current_failures[get_disconnected_name(group_name)] == 0):
            nodes, M = graph_structure.redundancy_groups[get_original_group_name(group_name)]
            total_network_availability_if_im_alive += len(nodes)
            #total_network_availability_if_im_alive *= (network_availability_if_im_alive ** len(nodes))

        if ("ssd" in group_name and num_failures > 0 and current_failures[get_disconnected_name(group_name)] == 0):
            #total_network_availability *= (network_availability ** num_failures)
            total_network_availability += num_failures
            data_loss = True
            nodes, M = graph_structure.redundancy_groups[get_original_group_name(group_name)]
            #total_network_availability_if_im_alive *= (network_availability_if_im_alive ** (len(nodes) - num_failures))
            total_network_availability_if_im_alive += len(nodes) - num_failures
            failed_ssd_nodes = nodes[0:num_failures]
            graph_structure.add_virtual_ssd_nodes(failed_ssd_nodes)
            flow_value, flow_dict = nx.maximum_flow(graph_structure.G, 'virtual_source', "virtual_sink")
            rebuild_speed_up = 1
            if (network_m > 0):
                rebuild_speed_up = network_m + (len(nodes) - M) - 1
            for u in flow_dict:
                for v, flow in flow_dict[u].items():
                    if (u == 'virtual_source'): continue
                    if (v == 'virtual_sink'): continue
                    if flow > 0:
                        # rebuild can be executed source and destination, so we multiply network_redundancy + 1
                        if network_m > 0:
                            graph_structure.G[u][v]['capacity'] -= rebuild_overhead * 2 * flow
                        else:
                            graph_structure.G[u][v]['capacity'] -= rebuild_overhead * flow
                            
            if (len(nodes) == M): # there is no parity
                # IO is not happen, so all bandwidth can be used for IO
                network_rebuild_flow[group_name] = flow_value * rebuild_overhead / num_failures
            else:
                network_rebuild_flow[group_name] = flow_value * rebuild_overhead / (network_redundancy) / num_failures * rebuild_speed_up
            #print ("network : ", num_failures, group_name, network_rebuild_flow[group_name], flow_value)
            graph_structure.remove_virtual_sink()
            #if (group_name == "ssd_group_1"):
            #    assert False
                # if network level rebuild cannot recover the network, remove related nodes
    flow = 0
    if (data_loss == True and network_K == 0):
        flow = 0
    else:
        flow = graph_structure.calculate_max_flow()
        #print (failed_node)
        #print (flow)
    max_flow_table[key] = flow
    max_flow_for_rebuild_table[key] = rebuilding_flow
    max_flow_for_network_rebuild_table[key] = network_rebuild_flow
    #network_availability_table_if_im_alive[key] = total_network_availability_if_im_alive
    #print (total_network_availability, total_network_availability_if_im_alive, network_availability, network_availability_if_im_alive)
    #print (input_ssd_availability ** 31)
    #network_availability_table[key] = total_network_availability * total_network_availability_if_im_alive
    network_availability_table[key] = (network_availability ** total_network_availability) * (network_availability_if_im_alive ** total_network_availability_if_im_alive)

def is_enclosure(node):
    return node[0].isupper()

def generate_first_failure_events(graph_structure, time_period, matched_module):
    events = []
    for node in list(graph_structure.G.nodes()):
        push_failed_event(events, node, 0, matched_module, graph_structure, time_period)
        
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
                    if (rebuild_bandwidth == 0 or rebuild_bandwidth < 0):
                        logging.critical("warning 1 update_repair_events !!!")
                        updated_repair_time = current_time + 1 / near_zero_value
                    else:
                        repaired_time = current_time - prev_time - SSDs[repair_node][2]
                        updated_switch_time_for_a_failed_ssd = 0
                        if (repaired_time < 0):
                            updated_switch_time_for_a_failed_ssd = -repaired_time
                            # not yet switched
                            repaired_time = 0
                        remaining_capacity = SSDs[repair_node][1] - repaired_time * SSDs[repair_node][0] * 3600
                        updated_repair_time = current_time + remaining_capacity / rebuild_bandwidth / 3600.0
                        if remaining_capacity < 0:
                            remaining_capacity = 0
                            updated_repair_time = current_time + near_zero_value
                        #print (repair_node, current_time, repair_time, updated_repair_time, SSDs[repair_node][0], rebuild_bandwidth)
                        SSDs[repair_node] = (rebuild_bandwidth, remaining_capacity, updated_switch_time_for_a_failed_ssd)
                elif (group in max_flow_for_network_rebuild_table[key]):
                    # network level rebuild
                    rebuild_bandwidth = max_flow_for_network_rebuild_table[key][group]
                    if (rebuild_bandwidth == 0 or rebuild_bandwidth < 0):
                        logging.critical("warning 2 update_repair_events !!!")
                        updated_repair_time = current_time + 1 / near_zero_value
                    else:
                        repaired_time = current_time - prev_time - SSDs[repair_node][2]
                        updated_switch_time_for_a_failed_ssd = 0
                        if (repaired_time < 0):
                            updated_switch_time_for_a_failed_ssd = -repaired_time
                            # not yet switched
                            repaired_time = 0
                        remaining_capacity = SSDs[repair_node][1] - repaired_time * SSDs[repair_node][0] * 3600
                        updated_repair_time = current_time + remaining_capacity / rebuild_bandwidth / 3600.0
                        if remaining_capacity < 0:
                            remaining_capacity = 0
                            updated_repair_time = current_time + near_zero_value
                        #print (repair_node, current_time, repair_time, updated_repair_time, SSDs[repair_node][0], rebuild_bandwidth)
                        SSDs[repair_node] = (rebuild_bandwidth, remaining_capacity, updated_switch_time_for_a_failed_ssd)
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

def push_repair_event(repair_events, event_node, current_time, matched_module, matched_group, graph_structure, key, SSDs, disconnected_ssds, failed_node, time_period):
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
                if (rebuild_bandwidth == 0 or rebuild_bandwidth < 0):
                    logging.critical ("warning 1 repaired_event !!!")
                    repair_time = current_time + 1 / near_zero_value
                else:
                    repair_time = current_time + SSD_capacity / rebuild_bandwidth / 3600.0
            elif (group in max_flow_for_network_rebuild_table[key]):
                # network level rebuild
                rebuild_bandwidth = max_flow_for_network_rebuild_table[key][group]
                if (rebuild_bandwidth == 0 or rebuild_bandwidth < 0):
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
            switch_time_for_a_failed_ssd = 0
            if check_repair_time_for_ssd_switching == True:
                switch_time_for_a_failed_ssd = random.expovariate(1 / mtr)
            SSDs[event_node] = (rebuild_bandwidth, SSD_capacity, switch_time_for_a_failed_ssd)
        else:
            assert False
        if (check_repair_time_for_ssd_switching == True):
            repair_time += SSDs[event_node][2]
    else:
        repair_time = current_time + random.expovariate(1 / mtr)
    #if (repair_time < time_period):
    heapq.heappush(repair_events, (repair_time, 'repair', event_node, current_time))

def push_failed_event_now(failed_events, event_node, current_time):
    heapq.heappush(failed_events, (current_time, 'fail', event_node, current_time))

def push_failed_event(failed_events, event_node, current_time, matched_module, graph_structure, time_period):
    module = matched_module[event_node]
    mttf = graph_structure.mttfs[module]
    if ("ssd" in module):
        write_mttf = SSD_write_mttf_per_1tb * SSD_capacity / 1_000_000_000_000
        write_ratio = SSD_write_percentage / 100
        mttf = 1 / ((1 - write_ratio) / mttf + write_ratio / write_mttf)
       # print (mttf)
    failure_time = current_time + random.expovariate(1 / mttf)
    #if (failure_time < time_period):
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

def count_failed_node(event_type, event_node, failed_node, current_failures_except_ssd, current_failures, matched_group):
    if event_type == 'fail':
        if (event_node in failed_node):
            #print ("+++++++++++++++", failed_node)
            failed_node[event_node] += 1
        else: # newly failed node
            failed_node[event_node] = 1
            group_name = matched_group[event_node]
            if (not "ssd" in group_name):
                current_failures_except_ssd[group_name] += 1
            current_failures[group_name] += 1
    elif event_type == 'repair' and event_node in failed_node:
        failed_node[event_node] -= 1
        if failed_node[event_node] == 0:
            group_name = matched_group[event_node]
            if (not "ssd" in group_name):
                current_failures_except_ssd[group_name] -= 1
            current_failures[group_name] -= 1
            #print ("-------------", failed_node)
            del failed_node[event_node]

def count_current_failures(graph_structure, disconnected_ssds, current_failures, matched_group):
    for group_name, (nodes, M) in graph_structure.redundancy_groups.items():
        if "ssd" in group_name:
            current_failures[get_disconnected_name(group_name)] = 0
    for ssd in disconnected_ssds:
        group_name = get_disconnected_name(matched_group[ssd])
        current_failures[group_name] += 1
        #print (group_name, current_failures)
    #print (current_failures)


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
    total_an_ssd_up_time = 0
    global lock
    global batch_size
    global completed
    global num_simulations
    global max_flow_table
    global max_flow_for_rebuild_table
    global max_flow_for_network_rebuild_table
    global disconnected_ssd_table
    global network_availability_table
    #global network_availability_table_if_im_alive

    graph_structure = copy.deepcopy(graph_structure_origin)

    matched_module = {}
    matched_group = {}
    matched_network_group = OrderedDict()
    max_flow_cdf = OrderedDict()
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
            # SSDs[node][2] indicates the switching time for a failed SSD
            SSDs[node] = (SSD_speed, 0, 0)

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
        an_ssd_up_time = 0
        an_ssd_up = True
        _calculate_connected_ssd(graph_structure, key1, failed_node)
        _calculate_max_flow(graph_structure, current_failures, key2, failed_node, failed_events, repair_events, matched_module, matched_group, matched_network_group)

        # Process events
        while(1):    
            event_time, event_type, event_node, temp = pop_event(failed_events, repair_events)
            # Calculate the time difference from the previous event
            break_flag = False
            if (event_time > time_period):
                event_time = time_period
                break_flag = True
            time_diff = event_time - prev_time
            if (time_diff < 0):
                print (event_type, event_node, event_time, prev_time)
                assert False
            """
            graph_structure.add_virtual_nodes()
            """
            connected = True
            coeff = 1.0
            
            max_flow = max_flow_table[key2]
            value = random.random()
            # if network case, we need to consider network availability
            # else, network_availability_table[key2] is 1
            #if (max_flow >= 0 and value > network_availability_table_if_im_alive[key2]):
            #    connected = False
            if (value > network_availability_table[key2]):
                connected = False
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
                max_flow_cdf[int(max_flow)] = max_flow_cdf.get(int(max_flow), 0) + time_diff
                up_time += time_diff
                effective_up_time += time_diff * coeff
            else:
                down_time += time_diff
            if (an_ssd_up == True):
                an_ssd_up_time += time_diff
            if (break_flag == True):
                break

            # Handle the event
            count_failed_node(event_type, event_node, failed_node, current_failures_except_ssd, current_failures, matched_group)
            # Update the maximum flow if necessary

            key1 = frozenset(current_failures_except_ssd.items())
            _calculate_connected_ssd(graph_structure, key1, failed_node)
            
            for ssd in prev_disconnected_ssds:
                if ssd not in disconnected_ssds:
                    count_failed_node('fail', ssd, failed_node, current_failures_except_ssd, matched_group)
            

            disconnected_ssds = disconnected_ssd_table[key1]
            count_current_failures(graph_structure, disconnected_ssds, current_failures, matched_group)
            key2 = frozenset(current_failures.items())
            _calculate_max_flow(graph_structure, current_failures, key2, failed_node, failed_events, repair_events, matched_module, matched_group, matched_network_group, event_time)

            #if (not "ssd" in event_node):
            update_repair_event(repair_events, event_time, matched_module, matched_group, graph_structure, key2, SSDs, disconnected_ssds)
            # Push the next repair event
            if event_type == 'fail':
                if ("ssd0" in event_node and "ssd0" not in disconnected_ssds):
                    an_ssd_up = False
                push_repair_event(repair_events, event_node, event_time, matched_module, matched_group, graph_structure, key2, SSDs, disconnected_ssds, failed_node, time_period)
            if event_type == 'repair':
                if ("ssd0" in event_node and "ssd0" not in disconnected_ssds):
                    an_ssd_up = True
                push_failed_event(failed_events, event_node, event_time, matched_module, graph_structure, time_period)
            
            if ("ssd0" in disconnected_ssds):
                an_ssd_up = False
            # if maximum flow is more than 0, an ssd can be rebuilt with local level
            if (max_flow_table[key2] > 0):
                an_ssd_up = True
            
            for ssd in prev_disconnected_ssds:
                if ssd not in disconnected_ssds:
                    # needs to be rebuild
                    push_failed_event_now(failed_events, ssd, event_time)
            
            prev_time = event_time
            prev_disconnected_ssds = copy.deepcopy(disconnected_ssds)
            #print (prev_disconnected_ssds)
        # Calculate the availability
        total_up_time += up_time
        total_down_time += down_time
        total_effective_up_time += effective_up_time
        total_an_ssd_up_time += an_ssd_up_time
        completed += 1
        #if (completed % 10000 == 0):
        if (simulation_idx == 0 and (completed * 100) % batch_size == 0):
            #print ("completed "+ str(completed * 100 // batch_size) + "%" + str(event_time), up_time + down_time)
            print ("completed "+ str(completed * 100 // batch_size) + "%")
        assert (up_time + down_time >= time_period - 1 and up_time + down_time <= time_period + 1)
    queue.put((total_up_time, total_down_time, total_effective_up_time, total_an_ssd_up_time, max_flow_cdf))
    return ""

def initial_computation(graph_structure, configuration):
    global max_bw
    global input_ssd_availability
    global network_availability
    global network_availability_if_im_alive
    max_bw = graph_structure.calculate_max_flow()
    if (configuration != "local_active + network_backup"):
        with open(network_injection_file, 'r') as file:
            # not ssd_availability alone, but considering local level rebuild
            input_ssd_availability = float(file.read())
    network_availability = 0
    network_availability_if_im_alive = 0
    for k in range(0, network_K + 1):
        # print (network_redundancy - 1, k)
        if (k == network_K):
            print (combinations_count(network_redundancy - 1, k))
            print (network_redundancy - 1 - k)
            print (input_ssd_availability ** (network_redundancy - 1 - k))
            print ((1 - input_ssd_availability) ** k)
            network_availability_if_im_alive = network_availability + combinations_count(network_redundancy - 1, k) * (input_ssd_availability ** (network_redundancy - 1 - k)) * ((1 - input_ssd_availability) ** k)
        else:
            network_availability += combinations_count(network_redundancy - 1, k) * (input_ssd_availability ** (network_redundancy - 1 - k)) * ((1 - input_ssd_availability) ** k)

if __name__ == "__main__":
    lock = threading.Lock()
    completed = 0
    max_flow_table = {}
    max_flow_for_rebuild_table = {}
    disconnected_ssd_table = OrderedDict()
    max_flow_for_network_rebuild_table = {}
    network_availability_table = {}
    network_availability_table_if_im_alive = {}
    current_failures = OrderedDict()
    max_bw = 0

    configuration = "local_active + network_backup"
    if network_redundancy > 1:
        if network_only:
            configuration = "network_only"
        else:
            configuration = "local_active + network_active"

    edges, enclosures, availabilities, redundancies, mttfs, mtrs = GraphStructure.parse_input_from_excel(file_path, sheet_name, start_cell, enclosure_start_cell, availability_sheet)
    if (ssd_M != 0):
        redundancies['ssd'] = (ssd_M, ssd_K)
    print (redundancies)
    graph_structure_origin = GraphStructure(edges, enclosures, availabilities, redundancies, mttfs, mtrs)
    print (graph_structure_origin.redundancy_groups)
    initial_computation(graph_structure_origin, configuration)
    # Reset the current failures
    current_failures = OrderedDict()
    # Determine the number of CPU cores
    procs = 40 #os.cpu_count()
    batch_size = (num_simulations + procs - 1) // procs
    jobs = []
    queue = multiprocessing.Queue(maxsize=100000000)
    for i in range(0, procs):
        out_list = list()
        process = multiprocessing.Process(target=monte_carlo_simulation, 
                                          args=(graph_structure_origin, time_period, i))
        jobs.append(process)

    # Start the processes (i.e. calculate the random number lists)      
    for j in jobs:
        j.start()

    # Ensure all of the processes have finished
    received = 0
    results = []
    while True:
        result = queue.get()
        if (result != None):
            results.append(result)
            received +=1
        if received == procs:
            break
    for j in jobs:
        j.join()
    

    total_up_time = 0
    total_down_time = 0
    total_effective_up_time = 0
    total_an_ssd_up_time = 0
    merged_max_flow_cdf = OrderedDict()
    print (graph_structure_origin.enclosures)
    for up_time, down_time, effective_up_time, an_ssd_up_time, max_flow_cdf in results:
        total_up_time += up_time
        total_down_time += down_time
        total_effective_up_time += effective_up_time
        total_an_ssd_up_time += an_ssd_up_time
        for key, value in max_flow_cdf.items():
            merged_max_flow_cdf[key] = merged_max_flow_cdf.get(key, 0) + value
    
    model = file_path.rstrip('.xlsx')
    
    num_network_groups = total_nodes // (network_M + network_K)
    output_data = (f"{model} | {configuration} | Total Up Time: {total_up_time:.2f} | Total Down Time: {total_down_time:.2f} | Total Effective Up Time: {total_effective_up_time:.2f} | "
                   f"Availability: {(total_up_time / (total_up_time + total_down_time)) ** num_network_groups:.12f} | Effective Availability: {(total_effective_up_time / (total_up_time + total_down_time)) ** num_network_groups:.12f} | "
                   f"Num Simulations: {num_simulations} | Time Period: {time_period} | SSD Capacity: {SSD_capacity} | SSD Speed: {SSD_speed} | "
                   f"Rebuild Overhead: {rebuild_overhead} | Network Redundancy: {network_redundancy} | Network Only: {network_only} | an SSD Availability : {total_an_ssd_up_time / (total_up_time + total_down_time)} | Network_M: {network_M} | Network_K: {network_K} | SSD_write_percentage: {SSD_write_percentage} | SSD_type: {SSD_type} | Network_m: {network_m} | Total Nodes: {total_nodes} | Network Availability: {network_availability} | Input SSD Availability: {input_ssd_availability} | "
                   f"SSD_M: {ssd_M} | SSD_K: {ssd_K} | SSD_m: {ssd_m} | Network_m: {network_m} | SSD Write MTTF: {SSD_write_mttf_per_1tb}\n")
    if output_file and generate_fault_injection == False:
        with open(output_file, 'a') as f:
            f.write(output_data)
    else:
        print(output_data)

    with open(flow_cdf_file, 'w') as f:
        for key in sorted(merged_max_flow_cdf, reverse=True):
            f.write(f"{key} {merged_max_flow_cdf[key]}\n")
    
    if generate_fault_injection:
        with open(network_injection_file, 'w') as file:
            file.write(f"{total_an_ssd_up_time / (total_up_time + total_down_time)}")
