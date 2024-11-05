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
from networkx.algorithms.flow import edmonds_karp

def parse_arguments():
    parser = argparse.ArgumentParser(description='Monte Carlo simulation parameters.')
    parser.add_argument('--num_simulations', type=int, default=4000000, help='Number of simulations to run')
    parser.add_argument('--time_period', type=int, default=20 * 365 * 24, help='Time period for the simulation in hours')
    parser.add_argument('--SSD_capacity', type=int, default=64_000_000_000_000, help='SSD capacity in bytes')
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
    parser.add_argument('--file_path', type=str, default='3tier.xlsx', help='Path to the Excel file')
    parser.add_argument('--SSD_type', type=str, default="qlc", help='SSD type, e.g., tlc, qlc.')
    parser.add_argument('--host_DWPD', type=float, default=1, help='host writes as DWPD')
    parser.add_argument('-o', '--output', type=str, help='Output file path to save results')
    parser.add_argument('--flow_cdf', type=str, default='flow_cdf.txt', help='File to save flow CDF')
    parser.add_argument('--leaf_node_module', type=str, default='ssd', help='Leaf node module name')
    parser.add_argument('--start_node_module', type=str, default='switch', help='Start node module name')
    parser.add_argument('--local_level_module', type=str, default='io_module,ssd', help='Local level module names, Comma-separated. First module is top module of the local modules')
    parser.add_argument('--lowest_common_level_module', type=str, default='backend_module', help='Lowest common level module name')
    parser.add_argument('--verbose', action='store_true', help='Print logs')
    args = parser.parse_args()
    return args

args = parse_arguments()

num_simulations = args.num_simulations
time_period = args.time_period
SSD_capacity = args.SSD_capacity
SSD_speed = 7_064_000_000
if (args.SSD_type == "qlc"):
    SSD_speed = 1_400_000_000
SSD_type = args.SSD_type
flow_cdf_file = args.flow_cdf

if (not SSD_type == "tlc" and not SSD_type == "qlc"):
    print ("SSD type is not supported")
    assert False
host_DWPD = args.host_DWPD
if (host_DWPD <= 0):
    print ("host_DWPD is less than 0")
    assert False
SSD_DWPD_limit = 1 
if (SSD_type == "qlc"):
    SSD_DWPD_limit = 0.26

rebuild_overhead = args.rebuild_overhead
network_M = args.network_M
network_K = args.network_K
ssd_M = 0
ssd_K = 0
ssd_m = 0
network_m = 0
max_bw = 0
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
guaranteed_years = 5
input_ssd_availability = -1
network_availability = -1
sheet_name = 'HW Architecture'
start_cell = ('B', 2)  # Corresponds to cell B2
enclosure_start_cell = ('F', 2)  # Corresponds to cell F2
availability_sheet = 'Availability'
redundancy_sheet = 'Redundancy'
generate_fault_injection = args.generate_fault_injection
leaf_node_module = args.leaf_node_module
start_node_module = args.start_node_module
local_level_module = set(args.local_level_module.split(','))
local_level_top_module = args.local_level_module.split(',')[0]
verbose = args.verbose
lowest_common_level_module = args.lowest_common_level_module

if (verbose == False):
    logging.disable(logging.CRITICAL + 1)

if (network_K > 0 and generate_fault_injection == True):
    print ("network_K is larger than 0, so fault injection is not supported")
    generate_fault_injection = False

if (generate_fault_injection == True):
    print ("local_level_top_module : ", local_level_top_module, "is start_node of this test when generating fault injection file")
    start_node_module = local_level_top_module

def combinations_count(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

def _calculate_connected_ssd(graph_structure_origin, key, failed_node, lowest_common_level_module, disconnected_ssd_table, lowest_common_level_module_connected_table):
    if (key in disconnected_ssd_table):
        return
    graph_structure = copy.deepcopy(graph_structure_origin)
    for node in failed_node:
        if (not leaf_node_module in node):
            if (node in graph_structure.G.nodes()):
                graph_structure.G.remove_node(node)
            # if first character of node is capital, it is enclosure
            if (is_enclosure(node)):
                for modules_node in graph_structure.enclosures[node]:
                    if (not leaf_node_module in modules_node and modules_node in graph_structure.G.nodes()):
                        graph_structure.G.remove_node(modules_node)

    graph_structure.add_virtual_nodes(start_node_module, leaf_node_module)
    disconnected_ssd = {}
    for node in graph_structure.G.nodes():
        if (leaf_node_module in node):
            connected = nx.has_path(graph_structure.G, 'virtual_source', node)
            if (connected == False):
                # If controller is failed, SSD shall be rebuilded
                disconnected_ssd[node] = 1
    disconnected_ssd_table[key] = disconnected_ssd
    graph_structure.remove_virtual_nodes()
    graph_structure.add_virtual_nodes(start_node_module, lowest_common_level_module)
    connected = nx.has_path(graph_structure.G, 'virtual_source', 'virtual_sink')
    #print (connected, lowest_common_level_module, disconnected_ssd)
    lowest_common_level_module_connected_table[key] = connected
                    

def _calculate_max_flow(graph_structure_origin, current_failures, key, failed_node, max_flow_table, max_flow_for_rebuild_table, max_flow_for_network_rebuild_table, network_availability, network_availability_if_im_alive, catastrophic_failures_table, first_group_available_table):
    if (key in max_flow_table):
        return
    graph_structure = copy.deepcopy(graph_structure_origin)
    rebuilding_flow = dict() # group to rebuild bandwidth mapping
    network_rebuild_flow = dict()
    skip_failure_group = dict()
    catastrophic_failure_group = dict()
    
    for node in failed_node:
        if (not leaf_node_module in node):
            if (node in graph_structure.G.nodes()):
                graph_structure.G.remove_node(node)
            # if first character of node is capital, it is enclosure
            if (is_enclosure(node)):
                for modules_node in graph_structure.enclosures[node]:
                    if (not leaf_node_module in modules_node and modules_node in graph_structure.G.nodes()):
                        graph_structure.G.remove_node(modules_node)
            """
            if node in matched_enclosure and "Component" in matched_enclosure[node]:
                enclosure = matched_enclosure[node]
                for modules_node in graph_structure.enclosures[enclosure]:
                    if (not leaf_node_module in modules_node and modules_node in graph_structure.G.nodes()):
                        graph_structure.G.remove_node(modules_node)
            """

    # configure edge's capacity except network_level rebuild case
    if (network_only == False):
        for group_name, num_failures in current_failures.items():
            if ("disconnected" in group_name):
                continue
            nodes, M = graph_structure.redundancy_groups[group_name]
            # local rebuild is possible
            if (leaf_node_module in group_name and len(nodes) - M >= num_failures and num_failures > 0 and current_failures[get_disconnected_name(group_name)] == 0):
                skip_failure_group[group_name] = 1
                idx = 0
                for node in nodes:
                    # all group shall be 80% 
                    rebuild_speed_up = 1
                    # declustered parity?
                    if (ssd_m > 0):
                        if (ssd_m > len(nodes)):
                            logging.critical("ssd_m is larger than the number of nodes in the group")
                            assert False
                        rebuild_speed_up = ssd_m + (len(nodes) - M) - 1
                    if (node in graph_structure.G.nodes()):
                        for u, v, edge_data in graph_structure.G.in_edges(node, data=True):
                            if 'capacity' in graph_structure.G[u][v]:
                                # if controller is exist
                                write_speed = min(SSD_speed, graph_structure.G[u][v]['capacity'])
                                if (idx == 0):
                                    #print(group_name)
                                    if (group_name in rebuilding_flow):
                                        rebuilding_flow[group_name] += rebuild_overhead * rebuild_speed_up * write_speed    
                                    else:
                                        rebuilding_flow[group_name] = rebuild_overhead * rebuild_speed_up * write_speed
                                if (ssd_m > 0):
                                    graph_structure.G[u][v]['capacity'] -= rebuild_overhead * 2 * write_speed
                                else:
                                    graph_structure.G[u][v]['capacity'] -= rebuild_overhead * write_speed
                        #print ("local : ", u, v, graph_structure.G[u][v]['capacity'], graph_structure_origin.G[u][v]['capacity'], rebuilding_flow[group_name])
                        idx += 1
                    else:
                        logging.critical("node is not in the graph", node)
    # if network level rebuilding is necessary, get maximum flow
    graph_structure.add_virtual_source(start_node_module)
    non_catastrophic_failed_group_count = 0
    catastrophic_failed_group_count = 0
    for group_name, num_failures in current_failures.items():
        if ("disconnected" in group_name):
            continue
        if (leaf_node_module in group_name and current_failures[get_disconnected_name(group_name)] > 0):
            catastrophic_failure_group[group_name] = 1
            catastrophic_failed_group_count += 1
        if (group_name in skip_failure_group):
            non_catastrophic_failed_group_count += 1
            continue
        if (leaf_node_module in group_name and num_failures == 0 and current_failures[get_disconnected_name(group_name)] == 0):
            nodes, M = graph_structure.redundancy_groups[get_original_group_name(group_name)]
            non_catastrophic_failed_group_count += 1
        if (leaf_node_module in group_name and num_failures > 0 and current_failures[get_disconnected_name(group_name)] == 0):
            catastrophic_failure_group[group_name] = 1
            catastrophic_failed_group_count += 1
            nodes, M = graph_structure.redundancy_groups[get_original_group_name(group_name)]
            failed_ssd_nodes = nodes[0:num_failures]
            graph_structure.add_virtual_ssd_nodes(failed_ssd_nodes, leaf_node_module, edge_value=SSD_speed)
            try:
                flow_value, flow_dict = nx.maximum_flow(graph_structure.G, 'virtual_source', 'virtual_sink', flow_func=edmonds_karp)
            except Exception as e:
                print(f"An error occurred: {e}")
                try:
                    flow_value, flow_dict = nx.maximum_flow(graph_structure.G, 'virtual_source', 'virtual_sink')
                except Exception as e:
                    print(f"An error occurred: {e}")
                connected = nx.has_path(graph_structure.G, 'virtual_source', 'virtual_sink')
                ancestors = nx.ancestors(graph_structure.G, 'virtual_sink')
                print (ancestors)
                edges = graph_structure.G.edges(data=True)
                for u, v, data in edges:
                    print(f"Edge ({u}, {v}) has capacity: {data.get('capacity', 'No capacity')}")
                print ("nodes : ", graph_structure.G.nodes())
                print ("connected : ", connected)
                print ("failed_ssd_nodes : ", failed_ssd_nodes)
                print ("num_failures : ", num_failures)
                print ("group_name : ", group_name)
                print ("flow_value : ", flow_value)
                print ("flow_dict : ", flow_dict)
                #assert False
            
            rebuild_speed_up = 1
            if (network_m > 0):
                rebuild_speed_up = network_m + (len(nodes) - M) - 1
            for u in flow_dict:
                for v, flow in flow_dict[u].items():
                    if (u == 'virtual_source'): continue
                    if (v == 'virtual_sink'): continue
                    # if common module, bandwidth is shared by all network nodes
                    bandwidth_divider = 1
                    if (v in local_level_module):
                        bandwidth_divider = network_redundancy
                    if flow > 0:
                        # rebuild can be executed source and destination, so we multiply network_redundancy + 1
                        if network_m > 0:
                            # 2 means that we need to read and write for SSD multiply in case of declustered parity
                            graph_structure.G[u][v]['capacity'] -= rebuild_overhead * 2 * flow  / num_failures * rebuild_speed_up / bandwidth_divider
                        else:
                            graph_structure.G[u][v]['capacity'] -= rebuild_overhead * flow / num_failures / bandwidth_divider
                            
            if (network_K == 0): # there is no network parity
                # IO is not happen, so all bandwidth can be used for IO
                network_rebuild_flow[group_name] = flow_value * rebuild_overhead / num_failures / bandwidth_divider
            else:
                network_rebuild_flow[group_name] = flow_value * rebuild_overhead / (network_redundancy) / num_failures * rebuild_speed_up / bandwidth_divider
            graph_structure.remove_virtual_sink()
    flows = {}
    graph_structure.remove_virtual_nodes()
    graph_structure.add_virtual_source(start_node_module)
    for group_name, num_failures in current_failures.items():
        if ("disconnected" in group_name):
            continue
        if (leaf_node_module in group_name and current_failures[get_disconnected_name(group_name)] == 0):
             nodes, M = graph_structure.redundancy_groups[get_original_group_name(group_name)]
             graph_structure.add_virtual_ssd_nodes(nodes, leaf_node_module)
             flow, flow_dict = nx.maximum_flow(graph_structure.G, 'virtual_source', 'virtual_sink', flow_func=edmonds_karp)
             flows[group_name] = flow
             graph_structure.remove_virtual_sink()
    graph_structure.remove_virtual_nodes()
    graph_structure.add_virtual_nodes(start_node_module, leaf_node_module)
    flow_value, flow_dict = nx.maximum_flow(graph_structure.G, 'virtual_source', 'virtual_sink', flow_func=edmonds_karp)
    #print(flow_value)
    flows['max_flow'] = flow_value
    max_flow_table[key] = flows
    max_flow_for_rebuild_table[key] = rebuilding_flow
    max_flow_for_network_rebuild_table[key] = network_rebuild_flow
    first_group_availabile = False
    if (not (leaf_node_module + "_group_0") in catastrophic_failure_group):
        first_group_availabile = True
    first_group_available_table[key] = first_group_availabile
    catastrophic_failures_table[key] = catastrophic_failure_group
    #print ("##########", catastrophic_failures_table[key], catastrophic_failed_group_count, network_availability, non_catastrophic_failed_group_count, network_availability_if_im_alive, skip_failure_group, current_failures)
    

def is_enclosure(node):
    return node[0].isupper()

def generate_first_failure_events(graph_structure, matched_module, matched_enclosure):
    events = []
    for node in list(graph_structure.G.nodes()):
        # each module's failure is ignored, only component's failure is considered
        #if (node in matched_enclosure and "Component" in matched_enclosure[node]):
        #   continue
        push_failed_event(events, node, 0, matched_module, graph_structure)
        
    # Generate enclosure failure and repair events
    for enclosure in list(graph_structure.enclosures):
        module = matched_module[enclosure]
        if module:
            mttf = graph_structure.mttfs[module]
            mtr = graph_structure.mtrs[module]
            failure_time = random.expovariate(1 / mttf)
            heapq.heappush(events, (failure_time, 'fail', enclosure, 0))
    return events

def update_repair_event(repair_events, current_time, matched_module, matched_group, graph_structure, key, SSDs, disconnected_ssds, max_flow_for_rebuild_table, max_flow_for_network_rebuild_table):
    updated_repair_events = []
    while(1):
        if (len(repair_events) == 0): break
        popped_event = heapq.heappop(repair_events)
        repair_time, repair_event, repair_node, prev_time = popped_event
        if (not leaf_node_module in repair_node):
            updated_repair_events.append(popped_event)
            continue
        else:
            if (repair_node in disconnected_ssds):
                # disconnected ssd cannot be repaired
                # do not enqueue and continue
                continue
            module = matched_module[repair_node]
            mtr = graph_structure.mtrs[module]
            mtr += near_zero_value
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
                        print ("rebuild_bandwidth : ", rebuild_bandwidth)
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
                    #print ("-----------", disconnected_ssds)
                    #print (key, group, repair_node)
                    logging.critical("warning 3 update_repair_events !!!")
                    updated_repair_time = current_time + 1 / near_zero_value
            updated_repair_events.append((updated_repair_time, repair_event, repair_node, current_time))
            #print (repair_node, current_time, repair_time, updated_repair_time, case_flag)
    for event in updated_repair_events:
        heapq.heappush(repair_events, event)

def push_repair_event(repair_events, event_node, current_time, matched_module, matched_group, graph_structure, key, SSDs, disconnected_ssds, failed_node, time_period, max_flow_for_rebuild_table, max_flow_for_network_rebuild_table):
    module = matched_module[event_node]
    mtr = graph_structure.mtrs[module]
    mtr += near_zero_value
    if (leaf_node_module in module):
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
                #print (repair_time - current_time)
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
            #print (repair_time - current_time)
    else:
        repair_time = current_time + random.expovariate(1 / mtr)
    #if (repair_time < time_period):
    heapq.heappush(repair_events, (repair_time, 'repair', event_node, current_time))

def push_failed_event_now(failed_events, event_node, current_time):
    heapq.heappush(failed_events, (current_time, 'fail', event_node, current_time))

def push_failed_event(failed_events, event_node, current_time, matched_module, graph_structure):
    module = matched_module[event_node]
    mttf = graph_structure.mttfs[module]
    if (leaf_node_module in module):
        mttf = 24 * 365 * guaranteed_years * SSD_DWPD_limit / host_DWPD
       # print(mttf)
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

def count_failed_node(event_type, event_node, failed_node, current_failures_except_ssd, current_failures, matched_group):
    if event_type == 'fail':
        if (event_node in failed_node):
            failed_node[event_node] += 1
        else: # newly failed node
            failed_node[event_node] = 1
            group_name = matched_group[event_node]
            if (not leaf_node_module in group_name):
                current_failures_except_ssd[group_name] += 1
            current_failures[group_name] += 1
    elif event_type == 'repair' and event_node in failed_node:
        failed_node[event_node] -= 1
        if failed_node[event_node] == 0:
            group_name = matched_group[event_node]
            if (not leaf_node_module in group_name):
                current_failures_except_ssd[group_name] -= 1
            current_failures[group_name] -= 1
            #print ("-------------", failed_node)
            del failed_node[event_node]

def count_current_failures(graph_structure, disconnected_ssds, current_failures, matched_group):
    for group_name, (nodes, M) in graph_structure.redundancy_groups.items():
        if leaf_node_module in group_name:
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


def monte_carlo_simulation(graph_structure_origin, time_period, simulation_idx, batch_size, network_availability, network_availability_if_im_alive):
    total_up_time = 0
    total_time = 0
    total_effective_up_time = 0
    total_an_ssd_up_time = 0
    completed = 0
    max_flow_table = {}
    max_flow_for_rebuild_table = {}
    disconnected_ssd_table = OrderedDict()
    lowest_common_level_module_connected_table = OrderedDict()
    max_flow_for_network_rebuild_table = {}
    catastrophic_failures_table = {}
    first_group_available_table = {}

    graph_structure = copy.deepcopy(graph_structure_origin)

    matched_module = {}
    matched_enclosure = {}
    matched_group = {}
    matched_network_group = OrderedDict()
    max_flow_cdf = OrderedDict()
    SSDs = {}
    for enclosure in list(graph_structure.enclosures):
        for module in graph_structure.mttfs.keys():
            if module in enclosure:
                matched_module[enclosure] = module
                matched_enclosure[module] = enclosure
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
        if leaf_node_module in node:
            # speed and capacity
            # if SSDs[node][0] is less than SSD_speed, some rebuild process happens
            # if SSDs[node][1] is less than SSD_capacity, some rebuild process happens (0 means not rebuilded SSD)
            # SSDs[node][2] indicates the switching time for a failed SSD
            SSDs[node] = (SSD_speed, 0, 0)
    current_failures = OrderedDict()
    current_failures_except_ssd = OrderedDict()
    for local_simul in range(0, batch_size):
        # when you add / delete graph, you need to copy here
        """
        graph_structure = copy.deepcopy(graph_structure_origin)
        """
        up_time = 0
        effective_up_time = 0
        time_for_a_simulation = 0
        failed_events = []
        repair_events = []
        prev_disconnected_ssds = OrderedDict()
        failed_node = OrderedDict()
        for group_name, (nodes, M) in graph_structure.redundancy_groups.items():
            current_failures[group_name] = 0
            if (not leaf_node_module in group_name):
                current_failures_except_ssd[group_name] = 0
            if (leaf_node_module in group_name):
                current_failures[get_disconnected_name(group_name)] = 0

        # Generate failure and repair events
        failed_events = generate_first_failure_events(graph_structure, matched_module, matched_enclosure)
        prev_time = 0
        # maximum flow for initial state
        key1 = frozenset(current_failures_except_ssd.items())
        key2 = frozenset(current_failures.items())
        an_ssd_up_time = 0
        _calculate_connected_ssd(graph_structure, key1, failed_node, lowest_common_level_module, disconnected_ssd_table, lowest_common_level_module_connected_table)
        _calculate_max_flow(graph_structure, current_failures, key2, failed_node, max_flow_table, max_flow_for_rebuild_table, max_flow_for_network_rebuild_table, network_availability, network_availability_if_im_alive, catastrophic_failures_table, first_group_available_table)

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
            connected = False
            coeff = 1.0
            
            max_flows = max_flow_table[key2]
            flow_for_time_diff = 0
            total_groups = len(max_flows.items())
            for group_name, flow in max_flows.items():
                connected = False
                random_value = random.random()
                #print (lowest_common_level_module_connected_table[key1])
                #if (lowest_common_level_module_connected_table[key1] == False):
                    #continue
                if (group_name in catastrophic_failures_table[key2]):
                    #print (group_name, key2)
                    if (network_K > 0 and random_value < network_availability):
                        connected = True
                else:
                    connected = True
                if (connected == True):
                    flow_for_time_diff = min(max_flows['max_flow'] / total_groups, flow)
                    coeff = flow_for_time_diff / max_bw
                    max_flow_cdf[int(flow_for_time_diff)] = max_flow_cdf.get(int(flow_for_time_diff), 0) + time_diff
                    up_time += (time_diff / total_groups)
                    effective_up_time += coeff * time_diff
                    #print (effective_up_time, flow)
                    if ("_group_0" in group_name):
                        an_ssd_up_time += time_diff
           
            time_for_a_simulation += time_diff
            if (break_flag == True):
                break
            # Handle the event
            count_failed_node(event_type, event_node, failed_node, current_failures_except_ssd, current_failures, matched_group)
            # Update the maximum flow if necessary
            key1 = frozenset(current_failures_except_ssd.items())
            _calculate_connected_ssd(graph_structure, key1, failed_node, lowest_common_level_module, disconnected_ssd_table, lowest_common_level_module_connected_table)
            
            for ssd in prev_disconnected_ssds:
                if ssd not in disconnected_ssds:
                    count_failed_node('fail', ssd, failed_node, current_failures_except_ssd, matched_group)
                    

            disconnected_ssds = disconnected_ssd_table[key1]
            count_current_failures(graph_structure, disconnected_ssds, current_failures, matched_group)
            key2 = frozenset(current_failures.items())
            _calculate_max_flow(graph_structure_origin, current_failures, key2, failed_node, max_flow_table, max_flow_for_rebuild_table, max_flow_for_network_rebuild_table, network_availability, network_availability_if_im_alive, catastrophic_failures_table, first_group_available_table)
            
            #if (not leaf_node_module in event_node):
            #print (current_failures['ssd_group_0'], current_failures['ssd_group_1'])
            update_repair_event(repair_events, event_time, matched_module, matched_group, graph_structure, key2, SSDs, disconnected_ssds, max_flow_for_rebuild_table, max_flow_for_network_rebuild_table)
            # Push the next repair event
            if event_type == 'fail':
                push_repair_event(repair_events, event_node, event_time, matched_module, matched_group, graph_structure, key2, SSDs, disconnected_ssds, failed_node, time_period, max_flow_for_rebuild_table, max_flow_for_network_rebuild_table)
            if event_type == 'repair':
                push_failed_event(failed_events, event_node, event_time, matched_module, graph_structure)
            
            for ssd in prev_disconnected_ssds:
                if ssd not in disconnected_ssds:
                    # needs to be rebuild
                    push_failed_event_now(failed_events, ssd, event_time)
            
            prev_time = event_time
            prev_disconnected_ssds = disconnected_ssds
            #print (prev_disconnected_ssds)
        # Calculate the availability
        total_up_time += up_time
        total_time += time_for_a_simulation
        total_effective_up_time += effective_up_time
        total_an_ssd_up_time += an_ssd_up_time
        completed += 1
        #if (completed % 10000 == 0):
        if (simulation_idx == 0 and (completed * 100) % batch_size == 0):
            #print ("completed "+ str(completed * 100 // batch_size) + "%" + str(event_time), up_time + time_for_a_simulation)
            print ("completed "+ str(completed * 100 // batch_size) + "%")
        assert (time_for_a_simulation >= time_period - 1 and time_for_a_simulation <= time_period + 1)
    queue.put((total_up_time, total_time, total_effective_up_time, total_an_ssd_up_time, max_flow_cdf))
    return ""

def initial_computation(graph_structure, configuration, input_ssd_availability = -1):
    global max_bw
    max_bw = graph_structure.calculate_max_flow(start_node_module, leaf_node_module)
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
            print ("network_redundancy :", k, network_availability)

    return network_availability, network_availability_if_im_alive, input_ssd_availability

if __name__ == "__main__":
    network_availability = 0
    network_availability_if_im_alive = 0
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
    network_availability, network_availability_if_im_alive, input_ssd_availability = initial_computation(graph_structure_origin, configuration)
    # Determine the number of CPU cores
    procs = 40 #os.cpu_count()
    batch_size = (num_simulations + procs - 1) // procs
    jobs = []
    queue = multiprocessing.Queue(maxsize=100000000)
    for i in range(0, procs):
        out_list = list()
        process = multiprocessing.Process(target=monte_carlo_simulation, 
                                          args=(graph_structure_origin, time_period, i, batch_size, network_availability, network_availability_if_im_alive))
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
    total_time = 0
    total_effective_up_time = 0
    total_an_ssd_up_time = 0
    merged_max_flow_cdf = OrderedDict()
    print (graph_structure_origin.enclosures)
    for up_time, time_for_a_simulation, effective_up_time, an_ssd_up_time, max_flow_cdf in results:
        total_up_time += up_time
        total_time += time_for_a_simulation
        total_effective_up_time += effective_up_time
        total_an_ssd_up_time += an_ssd_up_time
        for key, value in max_flow_cdf.items():
            merged_max_flow_cdf[key] = merged_max_flow_cdf.get(key, 0) + value
    
    model = file_path.rstrip('.xlsx')
    
    num_network_groups = total_nodes // (network_M + network_K)
    output_data = (f"{model} | {configuration} | Total Up Time: {total_up_time:.2f} | Total Down Time: {total_time - total_up_time:.2f} | Total Effective Up Time: {total_effective_up_time:.2f} | "
                   f"Availability: {(total_up_time /  total_time)} | Effective Availability: {(total_effective_up_time / total_time)} | "
                   f"Num Simulations: {num_simulations} | Time Period: {time_period} | SSD Capacity: {SSD_capacity} | SSD Speed: {SSD_speed} | "
                   f"Rebuild Overhead: {rebuild_overhead} | Network Redundancy: {network_redundancy} | Network Only: {network_only} | an SSD Availability : {total_an_ssd_up_time / (total_time)} | Network_M: {network_M} | Network_K: {network_K} | Host_DWPD: {host_DWPD} | SSD_type: {SSD_type} | Network_m: {network_m} | Total Nodes: {total_nodes} | Network Availability: {network_availability} | Input SSD Availability: {input_ssd_availability} | "
                   f"SSD_M: {ssd_M} | SSD_K: {ssd_K} | SSD_m: {ssd_m} | Network_m: {network_m} \n")
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
            file.write(f"{total_an_ssd_up_time / (total_time)}")
