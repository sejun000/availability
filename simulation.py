import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from collections import OrderedDict
import heapq
import random
import networkx as nx
import math
import utils
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

SSD_Enclosure = "NVMeEnclosure"

big_number = 1_000_000_000_000_000
## SSD state
SSD_state_normal = "normal"
# not switched to standby

SSD_state_intra_rebuilding = "intra_rebuilding"
SSD_state_inter_rebuilding = "inter_rebuilding"
SSD_state_inter_degraded = "inter_degraded"

# need to backup
SSD_state_data_loss = "data_loss" 

network_state_normal = "normal"
network_state_degraded = "degraded"
# if network_k == failure_count in network except the simulated node, data loss is up to the simulated node.
network_state_max_degraded = "max_degraded"
network_state_data_loss = "data_loss"

# SSD failure by intra failure or inter failure
# if network_K = 0, it is intra failure
# otherwise, it is inter failure

def calculate_hardware_graph(hardware_graph, failed_nodes_and_enclosures, enclosure_to_node_map, options, failed_hardware_graph_table, disconnected_table, key1):
    if (key1 in failed_hardware_graph_table):
        return failed_hardware_graph_table[key1]
    hardware_graph_copy = copy.deepcopy(hardware_graph)
    for failed_node_or_enclosure in failed_nodes_and_enclosures:
        if (failed_nodes_and_enclosures[failed_node_or_enclosure] == True):
            if failed_node_or_enclosure in hardware_graph_copy.G.nodes():
                hardware_graph_copy.G.remove_node(failed_node_or_enclosure)
            else: # enclosure
                for node in enclosure_to_node_map[failed_node_or_enclosure]:
                    # node can be already removed
                    if node in hardware_graph_copy.G.nodes():
                        hardware_graph_copy.G.remove_node(node)
    hardware_graph_copy.add_virtual_nodes(options["start_module"], options["end_module"])
    connected = nx.has_path(hardware_graph_copy.G, 'virtual_source', 'virtual_sink')
    hardware_graph_copy.remove_virtual_nodes()
    disconnected_table[key1] = {}
    disconnected_table[key1]['local_module'] = not connected
    hardware_graph_copy.add_virtual_nodes(options["start_module"], options["lowest_common_module"])
    connected = nx.has_path(hardware_graph_copy.G, 'virtual_source', 'virtual_sink')
    hardware_graph_copy.remove_virtual_nodes()
    disconnected_table[key1]['common_module'] = not connected
    failed_hardware_graph_table[key1] = hardware_graph_copy

def is_catastrophic_failure(failure_info, ssd_k, local_module_disconnected):
    return failure_info['failure_count'] > ssd_k or local_module_disconnected == True

def is_other_nodes_catastrophic_failure_and_recoverable(failure_info, ssd_k, network_k, disconnected):
    if (failure_info['network_failure_count'] > 0 and (not is_catastrophic_failure(failure_info, ssd_k, disconnected)) and failure_info['network_failure_count'] <= network_k and network_k > 0):
        return True
    
def judge_state_from_failure_info(failure_info, ssd_k, network_k, disconnected):
    if disconnected['common_module']:
        return SSD_state_data_loss
    local_module_disconnected = disconnected['local_module']
    if failure_info['failure_count'] > 0 and (not is_catastrophic_failure(failure_info, ssd_k, local_module_disconnected)):
        return SSD_state_intra_rebuilding
    elif (is_catastrophic_failure(failure_info, ssd_k, local_module_disconnected) and failure_info['network_failure_count'] <= network_k - 1 and network_k > 0):
        return SSD_state_inter_rebuilding
    elif ((network_k > 0 and failure_info['network_failure_count'] > network_k) or (is_catastrophic_failure(failure_info, ssd_k, local_module_disconnected) and failure_info['network_failure_count'] > network_k - 1)):
        return SSD_state_data_loss
    else:
        return SSD_state_normal
# input : failed_hardware_graph_table, ssd groups' failure
# output : flows_and_speed_table
def calculate_flows_and_speed(df, hardware_graph_copy, failure_info_per_ssd_group, ssd_m, ssd_k, ssd_total_count, network_m, network_k, ssd_read_bw, ssd_write_bw, ssd_read_latency, network_latency, options, flows_and_speed_table, disconnected, key2):
    # local rebuild
    if (key2 in flows_and_speed_table):
        return
    total_read_bw_for_ssds = 0

    tables = {}
    tables['intra_rebuilding_bw'] = {}
    tables['inter_rebuilding_bw'] = {}
    tables['backup_rebuild_speed'] = 0
    tables['max_read_performance'] = {}

    availability_ratio = 1
    dram_bandwidth = utils.KMG_to_bytes(options["dram_bandwidth"])
    if (options["nic_to_ssd_direct"] == False):
        dram_bandwidth /= 2

    hardware_graph_copy.add_virtual_nodes(options["lowest_common_module"], options["end_module"])
    hardware_max_flow = nx.maximum_flow_value(hardware_graph_copy.G, 'virtual_source', 'virtual_sink')
    hardware_graph_copy.remove_virtual_nodes()
    hardware_graph_copy.add_virtual_nodes(options["start_module"], options["lowest_common_module"])
    common_module_max_flow = nx.maximum_flow_value(hardware_graph_copy.G, 'virtual_source', 'virtual_sink')
    hardware_graph_copy.remove_virtual_nodes()

    normal_ssds_count = 0
    latencies = defaultdict(int)
    normal_latency = ssd_read_latency + network_latency
    for failure_info in failure_info_per_ssd_group.values():
        degraded_ssd_count = ssd_m - failure_info['failure_count']
        total_read_bw_for_ssds += ssd_read_bw * degraded_ssd_count
        normal_ssds_count += degraded_ssd_count
        latencies[normal_latency] += (ssd_m + ssd_k)
    
    total_read_bw_for_ssds = min(total_read_bw_for_ssds, dram_bandwidth)
    # if dram bandwidth is smaller than ssd read bw, we need to reduce ssd read bw
    ssd_read_bw = min(ssd_read_bw, total_read_bw_for_ssds / normal_ssds_count)

    # calculate bottlneck read performance to calculate degraded and rebuilding speed
    bottleneck_read_bw = min(total_read_bw_for_ssds, hardware_max_flow)
    # common module's flow is shared by all network nodes, so we need to divide it by the number of network nodes
    bottleneck_read_bw_per_ssd = bottleneck_read_bw / normal_ssds_count
    #print (bottleneck_read_bw, total_read_bw_for_ssds, common_module_max_flow, hardware_max_flow)
  #  print (bottleneck_read_bw, total_read_bw_for_ssds)
    for failure_info in failure_info_per_ssd_group.values():
        # failed ssd shall be read from the other ssds
        if judge_state_from_failure_info(failure_info, ssd_k, network_k, disconnected) == SSD_state_intra_rebuilding:
            local_failure_count = failure_info['failure_count']
            degraded_bw = calculate_bottleneck_speed(df, ssd_m, local_failure_count, [ssd_read_bw], options)
            rebuilding_bw = calculate_bottleneck_speed(df, ssd_m, local_failure_count, [ssd_read_bw, ssd_write_bw], options)
            tables['intra_rebuilding_bw'][local_failure_count] = rebuilding_bw
            degraded_ssd_count = ssd_m - local_failure_count
            total_read_bw_for_ssds = total_read_bw_for_ssds - degraded_bw * degraded_ssd_count
            latencies[normal_latency] -= local_failure_count
            
            intra_degreaded_latency = ssd_read_latency + network_latency + utils.get_encoding_latency_usec(df, ssd_m, local_failure_count)
            latencies[intra_degreaded_latency] += local_failure_count

        # catastrophic failure for the simulated nodes
        elif judge_state_from_failure_info(failure_info, ssd_k, network_k, disconnected) == SSD_state_inter_rebuilding:
            # failed ssd shall be read from the other network
            # we can assume that another network nodes also have similar status with the simulated nodes,
            # so we use bottlenck_read_bw_per_ssd as other degraded bw for the other network nodes
            network_failure_count = failure_info['network_failure_count']
            # +1 means that the simulated node is also failed
            degraded_bw = calculate_bottleneck_speed(df, network_m, network_failure_count + 1, [bottleneck_read_bw_per_ssd], options)
            rebuilding_bw = calculate_bottleneck_speed(df, network_m, network_failure_count + 1, [bottleneck_read_bw_per_ssd, ssd_write_bw], options)
            tables['inter_rebuilding_bw'][network_failure_count] = rebuilding_bw
            latencies[normal_latency] -= network_failure_count + 1
            inter_degraded_latency = ssd_read_latency + network_latency + utils.get_encoding_latency_usec(df, network_m, network_failure_count + 1)
            latencies[inter_degraded_latency] += network_failure_count + 1
        # data loss because of catastrophic failure is more than network_k
        elif judge_state_from_failure_info(failure_info, ssd_k, network_k, disconnected) == SSD_state_data_loss:
            rebuilding_bw = calculate_bottleneck_speed(df, network_m, 0, [bottleneck_read_bw_per_ssd, ssd_write_bw], options)
            tables['backup_rebuild_speed'] = rebuilding_bw
            #print (disconnected, bottleneck_read_bw_per_ssd, rebuilding_bw)
            # all read is removed from data loss failure, but bottleneck read bw is not changed (it is used for other ssds)
            degraded_ssd_count = ssd_m - failure_info['failure_count']
            total_read_bw_for_ssds = total_read_bw_for_ssds - ssd_read_bw * degraded_ssd_count
            availability_ratio -= (ssd_m + ssd_k) / ssd_total_count
            latencies[normal_latency] -= (ssd_m + ssd_k)
            latencies[big_number] += (ssd_m + ssd_k)

        # catastrophic failure for the other nodes. reduce the read bw for the simulated nodes
        if is_other_nodes_catastrophic_failure_and_recoverable(failure_info, ssd_k, network_k, disconnected):
             # failed ssd shall be read from this network to rebuild
            degraded_bw = calculate_bottleneck_speed(df, network_m, failure_info['network_failure_count'], [bottleneck_read_bw_per_ssd], options)
            degraded_ssd_count = ssd_m - failure_info['failure_count']
            bottleneck_read_bw = bottleneck_read_bw - degraded_bw * degraded_ssd_count

    max_read_performance = min(bottleneck_read_bw, total_read_bw_for_ssds)
    max_read_performance = min(max_read_performance, common_module_max_flow / (network_m + network_k))
    #print (max_read_performance)
    tables['max_read_performance'] = max_read_performance
    tables['availability_ratio'] = availability_ratio
    sum = 0
    for key in latencies.keys():
        sum += latencies[key]
    assert sum == ssd_total_count
    tables['latencies'] = latencies
    flows_and_speed_table[key2] = tables

def generate_first_failure_events(hardware_graph, node_to_module_map, ssd_total_count, n, ssd_mttf, network_mttf_table):
    events = []
    for node in list(hardware_graph.G.nodes()):
        push_failed_event(events, node, 0, node_to_module_map, hardware_graph, ssd_mttf)
    # Generate enclosure failure and repair events
    for enclosure in list(hardware_graph.enclosures):
        push_failed_event(events, enclosure, 0, node_to_module_map, hardware_graph, ssd_mttf)
    for ssd_index in range(0, ssd_total_count):
        push_failed_event(events, get_ssd_name(ssd_index), 0, node_to_module_map, hardware_graph, ssd_mttf)
    #for network_index in range(0, ssd_total_count / n):
    #    push_failed_event(events, get_network_group_name(network_index), 0, node_to_module_map, hardware_graph, ssd_mttf, network_mttf_table)
    return events

def calculate_bottleneck_speed(df, m, k, other_bws, options, use_erasure_coding = True):
    erasure_coding_latency = 0.00001
    if (k > 0 and use_erasure_coding):
        erasure_coding_latency = utils.get_encoding_latency_sec(df, m, k)
    erasure_coding_speed = 256_000 / erasure_coding_latency
    min_speed = erasure_coding_speed
    for other_bw in other_bws:
        if (other_bw * options["rebuild_bw_ratio"] < min_speed):
            min_speed = other_bw * options["rebuild_bw_ratio"]
    return min_speed

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


def update_failure_info(event_type, event_node, failure_info_per_ssd_group, failed_nodes_and_enclosures, n):
    if (SSD_module_name in event_node):
        ssd_index = get_ssd_index(event_node)
        ssd_group_name = get_ssd_group_name(ssd_index // n)
        if (event_type == 'fail'):
            failure_info_per_ssd_group[ssd_group_name]['failure_count'] += 1    
        elif (event_type == 'repair'):
            failure_info_per_ssd_group[ssd_group_name]['failure_count'] -= 1
    else:
        if (event_type == 'fail'):
            failed_nodes_and_enclosures[event_node] = True
        elif (event_type == 'repair'):
            failed_nodes_and_enclosures[event_node] = False

def get_SSD_changed_state(failure_info, ssd_k, network_k, disconnected):
    return judge_state_from_failure_info(failure_info, ssd_k, network_k, disconnected)

# only SSDs info shall be output and can be modified
def update_all_ssd_states(failure_info_per_ssd_group, SSDs, m, k, total_ssds, network_k, disconnected):
    for ssd_index in range(0, total_ssds):
        if SSDs[ssd_index]['failed'] == False:
            continue
        ssd_group_index = ssd_index // (m + k)
        failure_info = failure_info_per_ssd_group[get_ssd_group_name(ssd_group_index)]
        changed_state = get_SSD_changed_state(failure_info, k, network_k, disconnected)
        SSDs[ssd_index]['state'] = changed_state

def update_ssd_state(ssd_name, failure_info_per_ssd_group, SSDs, capacity, event_type, prep_time_for_rebuilding, m, k, network_k, disconnected):
    ssd_index = get_ssd_index(ssd_name)
    if (event_type == 'fail'):
        assert SSDs[ssd_index]['failed'] == False
        SSDs[ssd_index]['failed'] = True
        SSDs[ssd_index]['remaining_capacity_to_rebuild'] = capacity
        SSDs[ssd_index]['rebuild_speed'] = 0
        SSDs[ssd_index]['remaining_prep_time_for_rebuilding'] = random.expovariate(1 / prep_time_for_rebuilding)
        
    elif (event_type == 'repair'):
        assert SSDs[ssd_index]['failed'] == True
        SSDs[ssd_index]['failed'] = False

    n = m + k
    group_index = ssd_index // n
    failure_info = failure_info_per_ssd_group[get_ssd_group_name(group_index)]
        
    changed_state = get_SSD_changed_state(failure_info, k, network_k, disconnected)

    for i in range(group_index * n, group_index * n + n):
        #print (k, failure_info['failure_count'], network_k)
        if SSDs[i]['failed'] == False:
            continue
        #print ("changed", state_updated_ssd_name, changed_state)
        assert changed_state != SSD_state_normal
        SSDs[i]['state'] = changed_state    

def combinations_count(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

def generate_network_failure_table(network_n, availability_without_network_parity, network_mttf_table):
    probability = 0
    single_availability = availability_without_network_parity
    for failed_count in range(0, network_n):
        probability += combinations_count(network_n - 1, failed_count) * ((1 - single_availability) ** failed_count) * ((single_availability) ** (network_n - 1 - failed_count))
        network_mttf_table[failed_count] = probability
    #print (network_mttf_table)

def update_network_state(failure_info_per_ssd_group, n, total_ssds, network_n, network_mttf_table):
    changed = False
    for group_index in range(0, total_ssds // n):    
        random_value = random.uniform(0, 1)
        failure_info = failure_info_per_ssd_group[get_ssd_group_name(group_index)]
        for failed_count in range(0, network_n):
            if (random_value <= network_mttf_table[failed_count]):
                if failed_count != failure_info['network_failure_count']:
                    failure_info['network_failure_count'] = failed_count
                    changed = True
                break
    return changed

def push_failed_event(failed_events, event_node, current_time, node_to_module_map, hardware_graph, ssd_mttf):
    if (SSD_module_name in event_node):
        mttf = ssd_mttf
    else:
        module = node_to_module_map[event_node]
        mttf = hardware_graph.mttfs[module]
    failure_time = current_time + random.expovariate(1 / mttf)
    heapq.heappush(failed_events, (failure_time, 'fail', event_node, current_time))

def push_repair_event(repair_events, event_node, current_time, node_to_module_map, hardware_graph):
    if (SSD_module_name in event_node):
        heapq.heappush(repair_events, (current_time + big_number, 'repair', event_node, current_time))
        # update_repair_event_for_SSDs shall be called right after this function
    else:
        module = node_to_module_map[event_node]
        repair_time = current_time + hardware_graph.mtrs[module]
        heapq.heappush(repair_events, (repair_time, 'repair', event_node, current_time))

def update_repair_event_for_SSDs(repair_events, current_time, SSDs, flows_and_speed_entry, failure_info_per_ssd_group, n):
    updated_repair_events = []
    while(1):
        if (len(repair_events) == 0): break
        popped_event = heapq.heappop(repair_events)
        _, repair_event, repair_node, prev_time = popped_event
        assert repair_event == 'repair'
        if (not SSD_module_name in repair_node):
            updated_repair_events.append(popped_event)
            continue
        repaired_ssd_index = get_ssd_index(repair_node)
        current_ssd_info = SSDs[repaired_ssd_index]
        remaining_consumed_time = current_time - prev_time

        # before the repair, ssd shall be switched to standby
        if (current_ssd_info['remaining_prep_time_for_rebuilding'] > 0):
            remaining_prep_time_for_rebuilding = current_ssd_info['remaining_prep_time_for_rebuilding'] - remaining_consumed_time
            if (remaining_prep_time_for_rebuilding <= 0):   
                current_ssd_info['remaining_prep_time_for_rebuilding'] = 0
                remaining_consumed_time = -remaining_prep_time_for_rebuilding
            else:
                current_ssd_info['remaining_prep_time_for_rebuilding'] = remaining_prep_time_for_rebuilding
                remaining_consumed_time = 0

        # calculate the remaining capacity to previous rebuild speed
        if (remaining_consumed_time > 0):
            rebuild_speed = current_ssd_info['rebuild_speed']
            rebuild_speed_per_hour = rebuild_speed * 3600
            remaining_capacity_to_rebuild = current_ssd_info['remaining_capacity_to_rebuild'] - remaining_consumed_time * rebuild_speed_per_hour
            assert remaining_capacity_to_rebuild >= -0.1
            current_ssd_info['remaining_capacity_to_rebuild'] = remaining_capacity_to_rebuild    
            #print (repair_node, current_time, repair_time, updated_repair_time, case_flag)
        # change rebuild speed to the current state
        rebuild_speed = 0
        ssd_group_index = repaired_ssd_index // n
        ssd_failure_count = failure_info_per_ssd_group[get_ssd_group_name(ssd_group_index)]['failure_count']
        network_failure_count = failure_info_per_ssd_group[get_ssd_group_name(ssd_group_index)]['network_failure_count']

        #print (ssd_index, repair_node, current_ssd_info, flows_and_speed_entry['intra_rebuilding_bw'], failure_info_per_ssd_group)
        if (current_ssd_info['state'] == SSD_state_intra_rebuilding): # it can be reconstructed
            try:
                rebuild_speed = flows_and_speed_entry['intra_rebuilding_bw'][ssd_failure_count]
            except:
                print (repair_node, current_ssd_info, ssd_failure_count, flows_and_speed_entry, failure_info_per_ssd_group)
                assert False
        elif (current_ssd_info['state'] == SSD_state_inter_rebuilding):
            try:
                rebuild_speed = flows_and_speed_entry['inter_rebuilding_bw'][network_failure_count]
            except:
                print (repair_node, current_ssd_info, network_failure_count, flows_and_speed_entry, failure_info_per_ssd_group)
                assert False
        elif (current_ssd_info['state'] == SSD_state_data_loss):
            rebuild_speed = flows_and_speed_entry['backup_rebuild_speed']
        else:
            assert False
        current_ssd_info['rebuild_speed'] = rebuild_speed
        assert (current_ssd_info['remaining_prep_time_for_rebuilding'] >= 0)
        if (rebuild_speed == 0):
            # it may be disconnected, so, set rebuild_speed as small value now
            rebuild_speed = 1 / big_number
        rebuild_speed_per_hour = rebuild_speed * 3600
        updated_repair_event_time = current_ssd_info['remaining_prep_time_for_rebuilding'] + current_ssd_info['remaining_capacity_to_rebuild'] / rebuild_speed_per_hour
        #updated_repair_event_time = 1 / updated_repair_event_time)
        if (updated_repair_event_time < 0):
            print (repair_node, current_ssd_info, rebuild_speed, current_ssd_info['remaining_prep_time_for_rebuilding'], current_ssd_info['remaining_capacity_to_rebuild'])
            assert False
        updated_repair_events.append((current_time + updated_repair_event_time, 'repair', repair_node, current_time))

    for event in updated_repair_events:
        heapq.heappush(repair_events, event)

def initialize_simulation(hardware_graph, ssd_read_bw, total_ssd_count, options, network_n):
    node_to_module_map = {}
    enclosure_to_node_map = {}
    for enclosure in list(hardware_graph.enclosures):
        for module in hardware_graph.mttfs.keys():
            if module in enclosure:
                node_to_module_map[enclosure] = module
                break
        enclosure_to_node_map[enclosure] = hardware_graph.enclosures[enclosure]
    for node in list(hardware_graph.G.nodes()):
        for module in hardware_graph.mttfs.keys():
            if module in node:
                node_to_module_map[node] = module
                break
    
    # calculate the maximum flow without any failure
    hardware_graph_copy = copy.deepcopy(hardware_graph)
    hardware_graph_copy.add_virtual_nodes(options["start_module"], options["end_module"])
    max_flow_hardware = nx.maximum_flow_value(hardware_graph_copy.G, 'virtual_source', 'virtual_sink')
    hardware_graph_copy.remove_virtual_nodes()
    hardware_graph_copy.add_virtual_nodes(options["start_module"], options["lowest_common_module"])
    common_module_max_flow = nx.maximum_flow_value(hardware_graph_copy.G, 'virtual_source', 'virtual_sink')
    hardware_graph_copy.remove_virtual_nodes()
    ssd_max_read_performance = min(max_flow_hardware, total_ssd_count * ssd_read_bw)
    ssd_max_read_performance = min(ssd_max_read_performance, common_module_max_flow / network_n)
    dram_bandwidth = utils.KMG_to_bytes(options["dram_bandwidth"])
    ssd_max_read_performance = min(ssd_max_read_performance, dram_bandwidth)
    
    return node_to_module_map, enclosure_to_node_map, ssd_max_read_performance

def get_key1(failed_nodes_and_enclosures):
    return frozenset(failed_nodes_and_enclosures.items())

def get_key2(failed_nodes_and_enclosures, failure_info_per_ssd_group):
    frozen_set_list = []
    for _, failure_info in failure_info_per_ssd_group.items():
        #print (frozenset([ssd_group_name, frozenset(failure_info.items())]))
        frozen_set_list.append(frozenset(failure_info.items()))
    return frozenset([get_key1(failed_nodes_and_enclosures), frozenset(frozen_set_list)])

def get_percentile_value(raw_datas, ascending=True):
    df = pd.DataFrame(list(raw_datas.items()), columns=["value", "interval"])
    df = df.sort_values(by="value", ascending=ascending).reset_index(drop=True)

    df["cumulative_time"] = df["interval"].cumsum()

    total_time = df["cumulative_time"].iloc[-1]

    percentile_99_time = total_time * 0.99
    p99 = df[df["cumulative_time"] >= percentile_99_time].iloc[0]["value"]

    percentile_99_9_time = total_time * 0.999
    p99_9 = df[df["cumulative_time"] >= percentile_99_9_time].iloc[0]["value"]

    percentile_99_99_time = total_time * 0.9999
    p99_99 = df[df["cumulative_time"] >= percentile_99_99_time].iloc[0]["value"]

    median_time = total_time * 0.5
    median = df[df["cumulative_time"] >= median_time].iloc[0]["value"]

    average = (df["value"] * df["interval"]).sum() / df["interval"].sum()
    print (average, median, p99, p99_9, p99_99)
    return (average, median, p99, p99_9, p99_99)

def get_latencies_from_options(options, params_and_results):
    ssd_read_latency = 0
    is_qlc = params_and_results['qlc']
    if (is_qlc):
        ssd_read_latency = utils.convert_to_microseconds(options["qlc_read_latency"])
    else:
        ssd_read_latency = utils.convert_to_microseconds(options["tlc_read_latency"])
    network_latency = utils.convert_to_microseconds(options["network_latency"])
    return ssd_read_latency, network_latency

def monte_carlo_simulation(params_and_results, graph_structure_origin, batch_size, options):
    procs = 20 #os.cpu_count()
    batch_size = (batch_size + procs - 1) // procs
    jobs = []
    queue = multiprocessing.Queue(maxsize=100000000)
    for i in range(0, procs):
        process = multiprocessing.Process(target=simulation_per_core, 
                                          args=(i, params_and_results, graph_structure_origin, batch_size, options, queue))
        jobs.append(process)

    # Start the processes (i.e. calculate the random number lists)      
    for j in jobs:
        j.start()

    # Ensure all of the processes have finished
    received = 0
    results_from_proc = []
    while True:
        result = queue.get()
        if (result != None):
            results_from_proc.append(result)
            received +=1
        if received == procs:
            break
    for j in jobs:
        j.join()
    total_up_time = 0
    total_time = 0
    total_effective_up_time = 0
    effective_availabilities_dict = defaultdict(int)
    latencies_dict = defaultdict(int)
    ssd_read_latency, network_latency = get_latencies_from_options(options, params_and_results)
    normal_latency = ssd_read_latency + network_latency
    for up_time, simulation_time, effective_up_time, effective_availabilities, latencies in results_from_proc:
        total_up_time += up_time
        total_time += simulation_time
        total_effective_up_time += effective_up_time
        for key in effective_availabilities.keys():
            effective_availabilities_dict[key] += effective_availabilities[key]
        for key in latencies.keys():
            latencies_dict[normal_latency / key] += latencies[key]

    _, _, p99, p99_9, p99_99 = get_percentile_value(effective_availabilities_dict, False)
    avg, _, _, _, _= get_percentile_value(latencies_dict, False)

    params_and_results['up_time'] = total_up_time
    params_and_results['simulation_time'] = total_time
    params_and_results['effective_up_time'] = total_effective_up_time
    params_and_results['availability'] = total_up_time / total_time
    params_and_results['effective_availability'] = total_effective_up_time / total_time
    params_and_results['p99'] = p99
    params_and_results['p99_9'] = p99_9
    params_and_results['p99_99'] = p99_99
    params_and_results['latency_availability'] = avg

    print (params_and_results)
    if (params_and_results['network_k'] == 0):
        # open "last_avail.txt" file and write the availability
        with open(options["avail_file"], "w") as f:
            f.write(str(params_and_results['availability']))
    
def simulation_per_core(simulation_idx, params_and_results, graph_structure_origin, batch_size, options, queue):
    m = params_and_results['m']
    k = params_and_results['k']
    total_ssds = params_and_results['total_ssds']
    cached_m = params_and_results['cached_m']
    cached_k = params_and_results['cached_k']
    cached_network_m = params_and_results['cached_network_m']
    cached_network_k = params_and_results['cached_network_k']
    cached_ssds = params_and_results['cached_ssds']
    inter_replicas = params_and_results['inter_replica']
    intra_replicas = params_and_results['intra_replica']
    dwpd = params_and_results['dwpd']
    dwpd_limit = params_and_results['dwpd_limit']
    guaranteed_years = params_and_results['guaranteed_years']
    read_bw = params_and_results['ssd_read_bw']
    write_bw = params_and_results['ssd_write_bw']
    capacity = params_and_results['capacity']
    use_tbwpd = params_and_results['use_tbwpd']
    tbwpd = params_and_results['tbwpd']
    network_m = params_and_results['network_m']
    network_k = params_and_results['network_k']
    df = params_and_results['df']
    ssd_read_latency, network_latency = get_latencies_from_options(options, params_and_results)
    # In static analysis, standby_ssd is not considered
    n = m + k
    ssd_mttf = guaranteed_years * 365 * 24 * dwpd_limit / dwpd
    network_mttf_table = {}
    if (use_tbwpd):
        ssd_mttf = guaranteed_years * 365 * 24 * (dwpd_limit * capacity / 1_000_000_000_000) / tbwpd
    # effective dwpd is amplified by the number of parity chunks
    dwpd = dwpd * n / m
    # only 20% of the bandwidth is used for writing

    hardware_graph = copy.deepcopy(graph_structure_origin)
    node_to_module_map, enclosure_to_node_map, max_read_performance_without_any_failure = initialize_simulation(hardware_graph, read_bw, total_ssds, options, network_m + network_k)
    flows_and_speed_table = {}
    disconnected_table = {}
    failed_hardware_graph_table = {}
    effective_availabilities = defaultdict(int)
    latencies = defaultdict(int)

    total_up_time = 0
    total_time = 0
    total_effective_up_time = 0
    completed = 0

    for enclosure in list(hardware_graph.enclosures):
        for module in hardware_graph.mttfs.keys():
            if module in enclosure:
                node_to_module_map[enclosure] = module
                break
        enclosure_to_node_map[enclosure] = graph_structure_origin.enclosures[enclosure]
    for node in list(hardware_graph.G.nodes()):
        for module in hardware_graph.mttfs.keys():
            if module in node:
                node_to_module_map[node] = module
                break
    # open avail file and read the availability
    availability_without_network_parity = 0
    try:
        with open(options["avail_file"], "r") as f:
            availability_without_network_parity = float(f.readline())
    except:
        assert (network_k == 0)
    
    generate_network_failure_table(network_m + network_k, availability_without_network_parity, network_mttf_table)

    # to reduce remaining time from simulation, we just batch_size as 1 and multiply the simulation time by batch_size
    
    SSDs = {}
    for index in range(0, total_ssds):
        SSDs[index] = {}
        SSDs[index]['failed'] = False
        SSDs[index]['remaining_prep_time_for_rebuilding'] = 0
        SSDs[index]['remaining_capacity_to_rebuild'] = 0
        SSDs[index]['rebuild_speed'] = 0

    up_time = 0
    effective_up_time = 0
    timestamp = 0
    failed_events = []
    repair_events = []
    
    failed_nodes_and_enclosures = OrderedDict()
    failure_info_per_ssd_group = OrderedDict()
    for node in list(hardware_graph.G.nodes()):
        failed_nodes_and_enclosures[node] = False
    for i in range(0, total_ssds // n):
        ssd_group_name = get_ssd_group_name(i)
        failure_info_per_ssd_group[ssd_group_name] = {}
        failure_info_per_ssd_group[ssd_group_name]['failure_count'] = 0
        failure_info_per_ssd_group[ssd_group_name]['network_failure_count'] = 0
        
    # Generate failure and repair events
    failed_events = generate_first_failure_events(hardware_graph, node_to_module_map, total_ssds, n, ssd_mttf, network_mttf_table)
    prev_time = 0
    # maximum flow for initial state
    
    last_disconnected = {}
    last_disconnected['local_module'] = False
    last_disconnected['common_module'] = False

    key1 = get_key1(failed_nodes_and_enclosures)
    key2 = get_key2(failed_nodes_and_enclosures, failure_info_per_ssd_group)
    calculate_hardware_graph(hardware_graph, failed_nodes_and_enclosures, enclosure_to_node_map, options, failed_hardware_graph_table, disconnected_table, key1)
    calculate_flows_and_speed(df, failed_hardware_graph_table[key1], failure_info_per_ssd_group, m, k, total_ssds, network_m, network_k, read_bw, write_bw, ssd_read_latency, network_latency, options, flows_and_speed_table, last_disconnected, key2)
    simulation_hours = options["simulation_years"] * 365 * 24
    percentage = percentage_increasing = 1
    # Process events
    while(1):    
        event_time, event_type, event_node, temp = pop_event(failed_events, repair_events)
        # Calculate the time difference from the previous event
        break_flag = False
        if (event_time > simulation_hours * batch_size):
            event_time = simulation_hours * batch_size
            break_flag = True
        time_diff = event_time - prev_time
        if (time_diff < 0):
            print (event_type, event_node, event_time, prev_time)
            assert False

        # Calculate the availability
        flows_and_speed_entry = flows_and_speed_table[key2]
        availability_ratio = flows_and_speed_entry['availability_ratio']
        latency_entries = flows_and_speed_entry['latencies']
        effective_availability_ratio = flows_and_speed_entry['max_read_performance'] / max_read_performance_without_any_failure
        
        effective_up_time += time_diff * effective_availability_ratio
        up_time += time_diff * availability_ratio
        timestamp += time_diff
        effective_availabilities[effective_availability_ratio] += time_diff
        #print (effective_availabilities)
        for latency, count in latency_entries.items():
            latencies[latency] += count
            
        if (break_flag == True):
            break

        update_failure_info(event_type, event_node, failure_info_per_ssd_group, failed_nodes_and_enclosures, n)
        network_state_changed = update_network_state(failure_info_per_ssd_group, n, total_ssds, network_m + network_k, network_mttf_table)

        key1 = get_key1(failed_nodes_and_enclosures)
        calculate_hardware_graph(hardware_graph, failed_nodes_and_enclosures, enclosure_to_node_map, options, failed_hardware_graph_table, disconnected_table, key1)
        disconnected = disconnected_table[key1]

        key2 = get_key2(failed_nodes_and_enclosures, failure_info_per_ssd_group)
        calculate_flows_and_speed(df, failed_hardware_graph_table[key1], failure_info_per_ssd_group, m, k, total_ssds, network_m, network_k, read_bw, write_bw, ssd_read_latency, network_latency, options, flows_and_speed_table, disconnected, key2)

        if (SSD_module_name in event_node):
            update_ssd_state(event_node, failure_info_per_ssd_group, SSDs, capacity, event_type, options["prep_time_for_rebuilding"], m, k, network_k, disconnected)
        if (frozenset(last_disconnected.items()) != frozenset(disconnected.items()) or network_state_changed):
            update_all_ssd_states(failure_info_per_ssd_group, SSDs, m, k, total_ssds, network_k, disconnected)
            last_disconnected = disconnected
    
        #if (not leaf_node_module in event_node):
        if event_type == 'fail':
            push_repair_event(repair_events, event_node, event_time, node_to_module_map, hardware_graph)
        if event_type == 'repair':
            push_failed_event(failed_events, event_node, event_time, node_to_module_map, hardware_graph, ssd_mttf)

        flows_and_speed_entry = flows_and_speed_table[key2]
        #print (key2, flows_and_speed_entry)
        update_repair_event_for_SSDs(repair_events, event_time, SSDs, flows_and_speed_entry, failure_info_per_ssd_group, n)
        
        prev_time = event_time
        if (simulation_idx == 0 and (event_time * 100) // (simulation_hours * batch_size) > percentage):
            print ("completed ", percentage ,"%")
            percentage += percentage_increasing

        #print (prev_disconnected_ssds)
    # Calculate the availability
    total_up_time += up_time
    total_time += timestamp
    #print (timestamp, up_time, effective_availability_ratio, availability_ratio)
    total_effective_up_time += effective_up_time
    completed += 1
    #if (completed % 10000 == 0):
    assert (timestamp >= simulation_hours * batch_size - 1 and timestamp <= simulation_hours * batch_size + 1)

    queue.put((total_up_time, total_time, total_effective_up_time, effective_availabilities, latencies))
    
    return ""
    