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

import ssd
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
    
def judge_state_from_failure_info(failure_info, ssd_redun_scheme, disconnected, cached):
    ssd_k = ssd_redun_scheme.get_k(cached)
    network_k = ssd_redun_scheme.get_network_k(cached)
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
def calculate_flows_and_speed(df, hardware_graph_copy, failure_info_per_ssd_group, ssd_redun_scheme, tcp_stack_latency, network_latency, options, flows_and_speed_table, max_read_performance_without_any_failure, write_through, disconnected, key2):
    # local rebuild
    if (key2 in flows_and_speed_table):
        return
    
    tables = {}
    tables['intra_rebuilding_bw'] = {}
    tables['inter_rebuilding_bw'] = {}
    tables['backup_rebuild_speed'] = 0
    tables['cached_intra_rebuilding_bw'] = {}
    tables['cached_inter_rebuilding_bw'] = {}
    tables['cached_backup_rebuild_speed'] = 0

    availability_ratio = {}
    dram_bandwidth = get_dram_bandwidth(options)

    hardware_graph_copy.add_virtual_nodes(options["lowest_common_module"], options["end_module"])
    hardware_max_flow = nx.maximum_flow_value(hardware_graph_copy.G, 'virtual_source', 'virtual_sink')
    hardware_graph_copy.remove_virtual_nodes()
    hardware_graph_copy.add_virtual_nodes(options["start_module"], options["lowest_common_module"])
    common_module_max_flow = nx.maximum_flow_value(hardware_graph_copy.G, 'virtual_source', 'virtual_sink')
    hardware_graph_copy.remove_virtual_nodes()
    critical_path_length = 0
    if (hardware_max_flow != 0):
        critical_path = nx.dag_longest_path(hardware_graph_copy.G, weight=None) 
        critical_path_length = len(critical_path) - 1
        
    normal_ssds_count = 0
    latencies = {}
    latencies['cached_latencies'] = defaultdict(int)
    latencies['latencies'] = defaultdict(int)
    total_read_bw_for_ssds = 0
    end_module_degraded = hardware_graph_copy.get_module_degraded(options["end_module"])

    for group_index, failure_info in failure_info_per_ssd_group.items():
        cached = ssd_redun_scheme.is_ssd_group_index_cached(group_index)
        ssd_m = ssd_redun_scheme.get_m(cached)
        ssd_k = ssd_redun_scheme.get_k(cached)
        ssd_l = ssd_redun_scheme.get_l(cached)
        ssd_read_latency = ssd_redun_scheme.get_read_latency(cached)
        ssd_read_bw = ssd_redun_scheme.get_read_bw(cached)
        if (end_module_degraded):
            ssd_read_bw = ssd_read_bw / 2
        normal_latency = ssd_read_latency + network_latency * critical_path_length + tcp_stack_latency
        prefix = ssd_redun_scheme.get_cached_prefix(cached)
        degraded_ssd_count = ssd_m + ssd_k + ssd_l - failure_info['failure_count']
        total_read_bw_for_ssds += ssd_read_bw * degraded_ssd_count
        normal_ssds_count += degraded_ssd_count
        latencies[prefix + 'latencies'][normal_latency] += (ssd_m + ssd_k + ssd_l)
    
    # if dram bandwidth is bottleneck, we need to reduce the read bw as dram bandwidth
    local_read_degradation_ratio = dram_bandwidth / total_read_bw_for_ssds
    #print ("local_read", local_read_degradation_ratio)
    if (local_read_degradation_ratio > 1):
        local_read_degradation_ratio = 1

    total_read_bw_for_ssds = total_read_bw_for_ssds * local_read_degradation_ratio

    # calculate bottlneck read performance to calculate degraded and rebuilding speed
    bottleneck_read_bw = min(total_read_bw_for_ssds, hardware_max_flow)
    # common module's flow is shared by all network nodes, so we need to divide it by the number of network nodes
    bottleneck_read_bw_per_ssd = bottleneck_read_bw / normal_ssds_count
    #print (bottleneck_read_bw, total_read_bw_for_ssds, common_module_max_flow, hardware_max_flow)
  #  print (bottleneck_read_bw, total_read_bw_for_ssds)
    ssd_total_count = ssd_redun_scheme.get_total_ssds()
    availability_ratio['cached_availability'] = 1
    availability_ratio['availability'] = 1
    data_loss_ssds = 0
    changed_latencies = 0
    for group_index, failure_info in failure_info_per_ssd_group.items():
        # failed ssd shall be read from the other ssds
        
        cached = ssd_redun_scheme.is_ssd_group_index_cached(group_index)
        ssd_read_bw = ssd_redun_scheme.get_read_bw(cached)
        ssd_write_bw = ssd_redun_scheme.get_write_bw(cached)
        if (end_module_degraded):
            ssd_read_bw = ssd_read_bw / 2
            ssd_write_bw = ssd_write_bw / 2
        network_m = ssd_redun_scheme.get_network_m(cached)
        network_k = ssd_redun_scheme.get_network_k(cached)
        ssd_read_latency = ssd_redun_scheme.get_read_latency(cached)
        local_ssd_read_bw = ssd_read_bw * local_read_degradation_ratio
        prefix = ssd_redun_scheme.get_cached_prefix(cached)
        tiered_ssds = ssd_redun_scheme.get_tiered_ssds(cached)
        ssd_m = ssd_redun_scheme.get_m(cached)
        ssd_k = ssd_redun_scheme.get_k(cached)
        ssd_l = ssd_redun_scheme.get_l(cached)
        inter_replicas = ssd_redun_scheme.get_inter_replicas(cached)
        intra_replicas = ssd_redun_scheme.get_intra_replicas(cached)
        inter_replication = (inter_replicas > 0)
        intra_replication = (intra_replicas > 0)
        normal_latency = ssd_read_latency + network_latency * critical_path_length + tcp_stack_latency

        # before catastrophic failure for the simulated nodes
        if judge_state_from_failure_info(failure_info, ssd_redun_scheme, disconnected, cached) == SSD_state_intra_rebuilding:
            local_failure_count = failure_info['failure_count']
            degraded_ssds = ssd_m + ssd_k + ssd_l - local_failure_count
            rebuild_speed_up = 1
            if (ssd_l > 0):
                # rebuilding to reserve space
                rebuild_speed_up = degraded_ssds
            degraded_bw = calculate_bottleneck_speed(df, ssd_m, local_failure_count, [local_ssd_read_bw], options, intra_replication)
            rebuilding_bw = calculate_bottleneck_speed(df, ssd_m, local_failure_count, [local_ssd_read_bw, ssd_write_bw * rebuild_speed_up], options, intra_replication)
            tables[prefix + 'intra_rebuilding_bw'][local_failure_count] = rebuilding_bw
            # degraded read interferes with other ssds, so we need to reduce the read bw
            total_read_bw_for_ssds = total_read_bw_for_ssds - degraded_bw * degraded_ssds
            # failed ssd can be reconstructed from other ssds, so we need to increase that bw
            total_read_bw_for_ssds = total_read_bw_for_ssds + degraded_bw * local_failure_count
            #print (total_read_bw_for_ssds, degraded_bw * degraded_ssds)
            intra_degraded_latency = normal_latency + utils.get_encoding_latency_usec(df, ssd_m, local_failure_count, intra_replication)
            changed_latencies += utils.latency_changed(latencies[prefix + 'latencies'], normal_latency, intra_degraded_latency, local_failure_count)

        # catastrophic failure for the simulated nodes
        elif judge_state_from_failure_info(failure_info, ssd_redun_scheme, disconnected, cached) == SSD_state_inter_rebuilding:
            # failed ssd shall be read from the other network
            # we can assume that another network nodes also have similar status with the simulated nodes,
            # so we use bottlenck_read_bw_per_ssd as other degraded bw for the other network nodes
            network_failure_count = failure_info['network_failure_count']
            rebuilding_bw = calculate_bottleneck_speed(df, network_m, network_failure_count + 1, [bottleneck_read_bw_per_ssd, ssd_write_bw], options, inter_replication)
            tables[prefix + 'inter_rebuilding_bw'][network_failure_count] = rebuilding_bw
            inter_degraded_latency = normal_latency + utils.get_encoding_latency_usec(df, network_m, network_failure_count + 1, inter_replication)
            changed_latencies += utils.latency_changed(latencies[prefix + 'latencies'], normal_latency, inter_degraded_latency, network_failure_count + 1)

        # data loss because of catastrophic failure is more than network_k
        elif judge_state_from_failure_info(failure_info, ssd_redun_scheme, disconnected, cached) == SSD_state_data_loss:
            rebuilding_bw = calculate_bottleneck_speed(df, network_m, 0, [bottleneck_read_bw_per_ssd, ssd_write_bw], options)
            tables[prefix + 'backup_rebuild_speed'] = rebuilding_bw
            #print (disconnected, bottleneck_read_bw_per_ssd, rebuilding_bw)
            # all read is removed from data loss failure, but bottleneck read bw is not changed (it is used for other ssds)
            ssd_n = ssd_m + ssd_k + ssd_l

            # declustered parity will not failed all portion of stripe, so we need to calculate the portion of failed ssds
            data_loss_portion = 1 - combinations_count(ssd_n - failure_info['failure_count'], ssd_m) / combinations_count(ssd_n, ssd_m)

            degraded_ssd_count = ssd_m + ssd_k + ssd_l - failure_info['failure_count']
            # degraded read interferes with other ssds, so we need to reduce the read bw
            total_read_bw_for_ssds = total_read_bw_for_ssds - local_ssd_read_bw * degraded_ssd_count
            availability_ratio[prefix + 'availability'] -= (ssd_m + ssd_k + ssd_l) / tiered_ssds * data_loss_portion
            latencies[prefix + 'latencies'][normal_latency] -= (ssd_m + ssd_k + ssd_l)
            data_loss_ssds += ssd_m + ssd_k + ssd_l

        # catastrophic failure for the other nodes. reduce the read bw for the simulated nodes
        if is_other_nodes_catastrophic_failure_and_recoverable(failure_info, ssd_k, network_k, disconnected):
            # failed ssd shall be read from this network to rebuild
            degraded_bw = calculate_bottleneck_speed(df, network_m, failure_info['network_failure_count'], [bottleneck_read_bw_per_ssd, ssd_write_bw], options, inter_replication)
            # its degraded ssd count is the same with the simulated nodes
            degraded_ssd_count = ssd_m + ssd_k + ssd_l - failure_info['failure_count']
            bottleneck_read_bw = bottleneck_read_bw - degraded_bw * degraded_ssd_count
    #print (bottleneck_read_bw, total_read_bw_for_ssds, max_read_performance_without_any_failure, common_module_max_flow, network_m + network_k)
    max_read_performance = min(bottleneck_read_bw, total_read_bw_for_ssds)
    max_read_performance = min(max_read_performance, common_module_max_flow / options['network_nodes'])
    tables['eff_availability_ratio'] = max_read_performance / max_read_performance_without_any_failure
    tables['availability_ratio'] = availability_ratio
    sum = 0
    for key in latencies.keys():
        for latency in latencies[key]:
            sum += latencies[key][latency]

    assert sum + data_loss_ssds == ssd_total_count
    credit_avail_ratio = 1
    #if (sum != 0):
    #     credit_avail_ratio = min(1 - changed_latencies / sum, credit_avail_ratio)
    operational_utilization = 0.5
    if (tables['eff_availability_ratio'] < operational_utilization):
        credit_avail_ratio = min(tables['eff_availability_ratio'] / operational_utilization, credit_avail_ratio)
    avail_ratio = availability_ratio['availability'] * availability_ratio['cached_availability']
    if (write_through):
        avail_ratio = availability_ratio['availability']
    tables['credit_availability_ratio'] = min(credit_avail_ratio, avail_ratio)
    #print (tables['credit_availability_ratio'], changed_latencies, tables['eff_availability_ratio'])
    tables['latencies'] = latencies
    flows_and_speed_table[key2] = tables

def generate_first_failure_events(hardware_graph, node_to_module_map, ssd_total_count, ssd_redun_scheme):
    events = []
    for node in list(hardware_graph.G.nodes()):
        push_failed_event(events, node, 0, node_to_module_map, hardware_graph, ssd_redun_scheme)
    # Generate enclosure failure and repair events
    for enclosure in list(hardware_graph.enclosures):
        push_failed_event(events, enclosure, 0, node_to_module_map, hardware_graph, ssd_redun_scheme)
    for ssd_index in range(0, ssd_total_count):
        push_failed_event(events, ssd.get_ssd_name(ssd_index), 0, node_to_module_map, hardware_graph, ssd_redun_scheme)
    return events

def calculate_bottleneck_speed(df, m, k, other_bws, options, replication = False):
    erasure_coding_latency = 0.00000001
    if (k > 0 and not replication):
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


def update_failure_info(event_type, event_node, failure_info_per_ssd_group, failed_nodes_and_enclosures, ssd_redun_scheme):
    if (ssd.is_event_node_ssd(event_node)):
        ssd_index = ssd.get_ssd_index(event_node)
        ssd_group_index = ssd_redun_scheme.get_ssd_group_index(ssd_index)
        if (event_type == 'fail'):
            failure_info_per_ssd_group[ssd_group_index]['failure_count'] += 1    
        elif (event_type == 'repair'):
            failure_info_per_ssd_group[ssd_group_index]['failure_count'] -= 1
    else:
        if (event_type == 'fail'):
            failed_nodes_and_enclosures[event_node] = True
        elif (event_type == 'repair'):
            failed_nodes_and_enclosures[event_node] = False

def get_SSD_changed_state(failure_info, ssd_redun_scheme, disconnected, cached):
    return judge_state_from_failure_info(failure_info, ssd_redun_scheme, disconnected, cached)

# only SSDs info shall be output and can be modified
def update_all_ssd_states(failure_info_per_ssd_group, SSDs, ssd_redun_scheme, disconnected):
    for ssd_index in range(0, ssd_redun_scheme.total_ssds):
        if SSDs[ssd_index]['failed'] == False:
            continue
        ssd_group_index = ssd_redun_scheme.get_ssd_group_index(ssd_index)
        cached = ssd_redun_scheme.is_ssd_index_cached(ssd_index)
        failure_info = failure_info_per_ssd_group[ssd_group_index]
        changed_state = get_SSD_changed_state(failure_info, ssd_redun_scheme, disconnected, cached)
        SSDs[ssd_index]['state'] = changed_state

def update_ssd_state(ssd_name, failure_info_per_ssd_group, SSDs, capacity, event_type, prep_time_for_rebuilding, ssd_redun_scheme, disconnected):
    ssd_index = ssd.get_ssd_index(ssd_name)
    if (event_type == 'fail'):
        assert SSDs[ssd_index]['failed'] == False
        SSDs[ssd_index]['failed'] = True
        SSDs[ssd_index]['remaining_capacity_to_rebuild'] = capacity
        SSDs[ssd_index]['rebuild_speed'] = 0
        SSDs[ssd_index]['remaining_prep_time_for_rebuilding'] = random.expovariate(1 / prep_time_for_rebuilding)
        
    elif (event_type == 'repair'):
        assert SSDs[ssd_index]['failed'] == True
        SSDs[ssd_index]['failed'] = False
    
    cached = ssd_redun_scheme.is_ssd_index_cached(ssd_index)
    m = ssd_redun_scheme.get_m(cached)
    k = ssd_redun_scheme.get_k(cached)
    l = ssd_redun_scheme.get_l(cached)
    n = m + k + l
    group_index = ssd_redun_scheme.get_ssd_group_index(ssd_index)
    
    failure_info = failure_info_per_ssd_group[group_index]
        
    changed_state = get_SSD_changed_state(failure_info, ssd_redun_scheme, disconnected, cached)
    start_ssd_index = ssd_redun_scheme.get_start_ssd_index(group_index)
    for i in range(start_ssd_index, start_ssd_index + n):
        if SSDs[i]['failed'] == False:
            continue
        assert changed_state != SSD_state_normal
        SSDs[i]['state'] = changed_state    

def combinations_count(n, k):
    if (n < k):
        return 0
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

def generate_network_failure_table(network_n, availability_without_network_parity, availability_without_network_parity_for_cached_ssds, network_availability_table):
    probability = 0
    single_availability = availability_without_network_parity
    single_availability_for_cached_ssds = availability_without_network_parity_for_cached_ssds
    network_availability_table['availability'] = {}
    network_availability_table['cached_availability'] = {}
    for failed_count in range(0, network_n):
        probability += combinations_count(network_n - 1, failed_count) * ((1 - single_availability) ** failed_count) * ((single_availability) ** (network_n - 1 - failed_count))
        network_availability_table['availability'][failed_count] = probability
    for failed_count in range(0, network_n):
        probability += combinations_count(network_n - 1, failed_count) * ((1 - single_availability_for_cached_ssds) ** failed_count) * ((single_availability_for_cached_ssds) ** (network_n - 1 - failed_count))
        network_availability_table['cached_availability'][failed_count] = probability
    #print (network_availability_table)

def update_network_state(failure_info_per_ssd_group, ssd_redun_scheme, network_availability_table):
    changed = False
    total_group_count = ssd_redun_scheme.get_total_group_count()
    for group_index in range(0, total_group_count):    
        random_value = random.uniform(0, 1)
        failure_info = failure_info_per_ssd_group[group_index]
        cached = ssd_redun_scheme.is_ssd_group_index_cached(group_index)
        prefix = ssd_redun_scheme.get_cached_prefix(cached)
        network_n = ssd_redun_scheme.get_network_m(cached) + ssd_redun_scheme.get_network_k(cached)
        for failed_count in range(0, network_n):
            if (random_value <= network_availability_table[prefix + 'availability'][failed_count]):
                if failed_count != failure_info['network_failure_count']:
                    failure_info['network_failure_count'] = failed_count
                    changed = True
                break
    return changed

def push_failed_event(failed_events, event_node, current_time, node_to_module_map, hardware_graph, ssd_redun_scheme):
    if (ssd.SSD_module_name in event_node):
        ssd_index = ssd.get_ssd_index(event_node)
        cached = ssd_redun_scheme.is_ssd_index_cached(ssd_index)
        mttf = ssd_redun_scheme.get_mttf(cached)
    else:
        module = node_to_module_map[event_node]
        mttf = hardware_graph.mttfs[module]
    failure_time = current_time + random.expovariate(1 / mttf)
    heapq.heappush(failed_events, (failure_time, 'fail', event_node, current_time))

def push_repair_event(repair_events, event_node, current_time, node_to_module_map, hardware_graph):
    if (ssd.SSD_module_name in event_node):
        heapq.heappush(repair_events, (current_time + big_number, 'repair', event_node, current_time))
        # update_repair_event_for_SSDs shall be called right after this function
    else:
        module = node_to_module_map[event_node]
        repair_time = current_time + hardware_graph.mtrs[module]
        heapq.heappush(repair_events, (repair_time, 'repair', event_node, current_time))

def get_dram_bandwidth(options):
    dram_bandwidth = utils.KMG_to_bytes(options["dram_bandwidth"])
    if (options['nic_to_ssd_direct'] == False):
        dram_bandwidth /= 2
    return dram_bandwidth


def update_repair_event_for_SSDs(repair_events, current_time, SSDs, flows_and_speed_entry, failure_info_per_ssd_group, ssd_redun_scheme):
    updated_repair_events = []
    while(1):
        if (len(repair_events) == 0): break
        popped_event = heapq.heappop(repair_events)
        _, repair_event, repair_node, prev_time = popped_event
        assert repair_event == 'repair'
        if (not ssd.SSD_module_name in repair_node):
            updated_repair_events.append(popped_event)
            continue
        repaired_ssd_index = ssd.get_ssd_index(repair_node)
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
            if (remaining_capacity_to_rebuild <= -100):
                print (repair_node, current_ssd_info, remaining_consumed_time, rebuild_speed, current_ssd_info['remaining_capacity_to_rebuild'])
            #assert remaining_capacity_to_rebuild >= -10

            current_ssd_info['remaining_capacity_to_rebuild'] = remaining_capacity_to_rebuild    
            #print (repair_node, current_time, repair_time, updated_repair_time, case_flag)
        # change rebuild speed to the current state
        rebuild_speed = 0
        ssd_group_index = ssd_redun_scheme.get_ssd_group_index(repaired_ssd_index)
        cached = ssd_redun_scheme.is_ssd_group_index_cached(ssd_group_index)
        prefix = ssd_redun_scheme.get_cached_prefix(cached)
        ssd_failure_count = failure_info_per_ssd_group[ssd_group_index]['failure_count']
        network_failure_count = failure_info_per_ssd_group[ssd_group_index]['network_failure_count']

        #print (ssd_index, repair_node, current_ssd_info, flows_and_speed_entry['intra_rebuilding_bw'], failure_info_per_ssd_group)
        if (current_ssd_info['state'] == SSD_state_intra_rebuilding): # it can be reconstructed
            try:
                rebuild_speed = flows_and_speed_entry[prefix + 'intra_rebuilding_bw'][ssd_failure_count]
            except:
                print (repair_node, current_ssd_info, ssd_failure_count, flows_and_speed_entry, failure_info_per_ssd_group)
                assert False
        elif (current_ssd_info['state'] == SSD_state_inter_rebuilding):
            try:
                rebuild_speed = flows_and_speed_entry[prefix + 'inter_rebuilding_bw'][network_failure_count]
            except:
                print (repair_node, current_ssd_info, network_failure_count, flows_and_speed_entry, failure_info_per_ssd_group)
                assert False
        elif (current_ssd_info['state'] == SSD_state_data_loss):
            rebuild_speed = flows_and_speed_entry[prefix + 'backup_rebuild_speed']
        else:
            assert False
        current_ssd_info['rebuild_speed'] = rebuild_speed
        assert (current_ssd_info['remaining_prep_time_for_rebuilding'] >= 0)
        if (rebuild_speed == 0):
            # it may be disconnected, so, set rebuild_speed as small value now
            rebuild_speed = 1 / big_number
        rebuild_speed_per_hour = rebuild_speed * 3600
        if (current_ssd_info['remaining_capacity_to_rebuild'] < 0):
            current_ssd_info['remaining_capacity_to_rebuild'] = 0
        updated_repair_event_time = current_ssd_info['remaining_prep_time_for_rebuilding'] + current_ssd_info['remaining_capacity_to_rebuild'] / rebuild_speed_per_hour
        if (updated_repair_event_time < 0):
            print (repair_node, current_ssd_info, rebuild_speed, current_ssd_info['remaining_prep_time_for_rebuilding'], current_ssd_info['remaining_capacity_to_rebuild'])
            assert False
        updated_repair_events.append((current_time + updated_repair_event_time, 'repair', repair_node, current_time))

    for event in updated_repair_events:
        heapq.heappush(repair_events, event)

def initialize_simulation(hardware_graph, ssd_read_bw, total_ssd_count, options):
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
    ssd_max_read_performance = min(ssd_max_read_performance, common_module_max_flow / options['network_nodes'])
    dram_bandwidth = get_dram_bandwidth(options)
    ssd_max_read_performance = min(ssd_max_read_performance, dram_bandwidth)
    
    return node_to_module_map, enclosure_to_node_map, ssd_max_read_performance

def get_key1(failed_nodes_and_enclosures):
    return frozenset(failed_nodes_and_enclosures.items())

def get_key2(failed_nodes_and_enclosures, failure_info_per_ssd_group, ssd_redun_scheme):
    frozen_set_list = []
    for group_index, failure_info in failure_info_per_ssd_group.items():
        cached = ssd_redun_scheme.is_ssd_group_index_cached(group_index)
        frozen_set_list.append(frozenset([cached, frozenset(failure_info.items())]))
    return frozenset([get_key1(failed_nodes_and_enclosures), frozenset(frozen_set_list)])

def get_latencies_from_options(options, params_and_results):
    ssd_read_latency = 0
    is_qlc = params_and_results['qlc']
    if (is_qlc):
        ssd_read_latency = utils.convert_to_microseconds(options["qlc_read_latency"])
    else:
        ssd_read_latency = utils.convert_to_microseconds(options["tlc_read_latency"])
    tcp_stack_latency = utils.convert_to_microseconds(options["tcp_stack_latency"])
    return ssd_read_latency, tcp_stack_latency

def calculate_module_cost(node, node_to_module_map, costs, ssd_redun_scheme, cached_ssd_cost, uncached_ssd_cost, options):
    if (not ssd.SSD_module_name in node):
        module = node_to_module_map[node]
        return costs[module] * get_coefficient_for_cost(module, options)
    else:
        ssd_index = ssd.get_ssd_index(node)
        cached = ssd_redun_scheme.is_ssd_index_cached(ssd_index)
        if (cached):
            return cached_ssd_cost
        return uncached_ssd_cost * get_coefficient_for_cost(ssd.SSD_module_name, options)

# We set coefficient for cost manually
def get_coefficient_for_cost(module, options):
    if (module == ssd.SSD_module_name):
        return options['network_nodes']
    if (module == 'io_module'):
        return options['network_nodes']
    if (module == "host_module"):
        return options['network_nodes']
    if (module == "backend_module"):
        return options['network_nodes']
    if (module == "NVMeEnclosure"):
        return options['network_nodes']
    return 1


def get_initial_cost(hardware_graph, node_to_module_map, ssd_total_count, ssd_redun_scheme, cached_ssd_cost, uncached_ssd_cost, costs, options):
    initial_cost = 0
    for node in list(hardware_graph.G.nodes()):
        module = node_to_module_map[node]
        initial_cost += costs[module] * get_coefficient_for_cost(module, options)
    # Generate enclosure failure and repair events
    for enclosure in list(hardware_graph.enclosures):
        module = node_to_module_map[enclosure]
        initial_cost += costs[module] * get_coefficient_for_cost(module, options)
    for ssd_index in range(0, ssd_total_count):
        cached = ssd_redun_scheme.is_ssd_index_cached(ssd_index)
        if (cached):
            initial_cost += cached_ssd_cost * get_coefficient_for_cost(ssd.SSD_module_name, options)
        else:
            initial_cost += uncached_ssd_cost * get_coefficient_for_cost(ssd.SSD_module_name, options)
    return initial_cost

def monte_carlo_simulation(params_and_results, graph_structure_origin, num_simulations, options, costs):
    procs = 40 #os.cpu_count()
    
    batch_size = (num_simulations + procs - 1) // procs
    jobs = []
    queue = multiprocessing.Queue(maxsize=100000000)
    for i in range(0, procs):
        process = multiprocessing.Process(target=simulation_per_core, 
                                          args=(i, params_and_results, graph_structure_origin, batch_size, options, costs, queue))
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
    total_cached_up_time = 0
    total_effective_up_time = 0
    total_credit_up_time = 0
    effective_availabilities_dict = defaultdict(int)
    latencies_dict = defaultdict(int)
    cached_latencies_dict = defaultdict(int)
    total_latencies_dict = defaultdict(int)

    cached_read_ratio = params_and_results['cached_read_ratio']
    initial_cost_sum = 0
    total_cost_sum = 0
    total_time_for_rebuilding = 0
    total_count_for_rebuilding = 0
    cached_mttf_output = 0
    mttf_output = 0
    total_credit_ratio = 0
    total_cached_ssd_repair_cost = 0
    total_uncached_ssd_repair_cost = 0

    for up_time, cached_up_time, credit_up_time, simulation_time, effective_up_time, effective_availabilities, latencies, cached_latencies, initial_cost, total_cost, time_for_rebuilding, count_for_rebuilding, cached_mttf, mttf, cached_ssd_repair_cost, uncached_ssd_repair_cost in results_from_proc:
        total_up_time += up_time
        total_cached_up_time += cached_up_time
        total_credit_up_time += credit_up_time
        total_time += simulation_time
        total_effective_up_time += effective_up_time
        for key in effective_availabilities.keys():
            effective_availabilities_dict[key] += effective_availabilities[key] 
        for key in latencies.keys():
            latencies_dict[key] += latencies[key] 
            total_latencies_dict[key] += latencies[key] * (1 - cached_read_ratio)
        for key in cached_latencies.keys():
            cached_latencies_dict[key] += cached_latencies[key]
            total_latencies_dict[key] += cached_latencies[key] * cached_read_ratio
        initial_cost_sum += initial_cost
        total_cost_sum += total_cost
        total_time_for_rebuilding += time_for_rebuilding
        total_count_for_rebuilding += count_for_rebuilding
        cached_mttf_output = cached_mttf
        mttf_output = mttf
        total_credit_ratio += nine_to_credit(utils.get_nines(credit_up_time / simulation_time))
        total_cached_ssd_repair_cost += cached_ssd_repair_cost
        total_uncached_ssd_repair_cost += uncached_ssd_repair_cost

    avg_time_for_rebuilding = total_time_for_rebuilding / total_count_for_rebuilding
    avg_initial_cost = initial_cost_sum / procs
    avg_total_cost = total_cost_sum / procs
    total_credit_ratio /= procs
   
    #_, _, p99, p99_9, p99_99 = utils.get_percentile_value(effective_availabilities_dict, False)
    avg_latency, median, p99, p99_9, p99_99 = utils.get_percentile_value(latencies_dict, True)
    if (len(cached_latencies_dict) == 0):
        cached_avg_latency = 0
        cached_median = 0
        cached_p99 = 0
        cached_p99_9 = 0
        cached_p99_99 = 0
    else:
        cached_avg_latency, cached_median, cached_p99, cached_p99_9, cached_p99_99 = utils.get_percentile_value(cached_latencies_dict, True)
    total_avg_latency, total_median, total_p99, total_p99_9, total_p99_99 = utils.get_percentile_value(total_latencies_dict, True)
    params_and_results['simulation_year'] = options["simulation_years"]
    params_and_results['up_time'] = total_up_time
    params_and_results['cached_up_time'] = total_cached_up_time
    params_and_results['simulation_time'] = total_time
    params_and_results['effective_up_time'] = total_effective_up_time
    params_and_results['credit_up_time'] = total_credit_up_time
    params_and_results['uncached_availability'] = total_up_time / total_time
    params_and_results['cached_availability'] = total_cached_up_time / total_time
    params_and_results['availability'] = params_and_results['uncached_availability'] * params_and_results['cached_availability']
    params_and_results['avail_nines'] = utils.get_nines(params_and_results['availability'])
    #down_time_hours_per_year = 365 * 24 * (1 - params_and_results['availability'])
    #total_down_time_cost = (total_time - total_up_time) * options['down_time_cost_per_hour']
    params_and_results['total_credit_ratio'] = total_credit_ratio
    repair_cost_per_year = (avg_total_cost - avg_initial_cost) / options["simulation_years"] / batch_size
    cached_ssd_repair_cost_per_year = total_cached_ssd_repair_cost / options["simulation_years"] / batch_size
    uncached_ssd_repair_cost_per_year = total_uncached_ssd_repair_cost / options["simulation_years"] / batch_size
    down_cost_per_year = total_credit_ratio * (avg_initial_cost / 10 + repair_cost_per_year)
    operation_cost_per_year = repair_cost_per_year + down_cost_per_year
    if (params_and_results['write_through']):
        params_and_results['availability'] = params_and_results['uncached_availability']
    params_and_results['avail_nines'] = utils.get_nines(params_and_results['availability'])
    params_and_results['effective_availability'] = total_effective_up_time / total_time
    params_and_results['eff_avail_nines'] = utils.get_nines(params_and_results['effective_availability'])
    params_and_results['credit_availability'] = total_credit_up_time / total_time
    params_and_results['credit_avail_nines'] = utils.get_nines(params_and_results['credit_availability'])
    params_and_results['avg_latency'] = avg_latency
    params_and_results['median'] = median
    params_and_results['p99'] = p99
    params_and_results['p99_9'] = p99_9
    params_and_results['p99_99'] = p99_99
    params_and_results['cached_avg_latency'] = cached_avg_latency
    params_and_results['cached_median'] = cached_median
    params_and_results['cached_p99'] = cached_p99
    params_and_results['cached_p99_9'] = cached_p99_9
    params_and_results['cached_p99_99'] = cached_p99_99
    params_and_results['total_avg_latency'] = total_avg_latency
    params_and_results['total_median'] = total_median
    params_and_results['total_p99'] = total_p99
    params_and_results['total_p99_9'] = total_p99_9
    params_and_results['total_p99_99'] = total_p99_99
    params_and_results['operation_cost_per_year'] = operation_cost_per_year
    params_and_results['repair_cost_per_year'] = repair_cost_per_year
    params_and_results['down_cost_per_year'] = down_cost_per_year
    params_and_results['cached_ssd_repair_cost_per_year'] = cached_ssd_repair_cost_per_year
    params_and_results['uncached_ssd_repair_cost_per_year'] = uncached_ssd_repair_cost_per_year
    params_and_results['initial_cost'] = avg_initial_cost
    params_and_results['repair_cost_for_10_years'] = repair_cost_per_year * 10
    params_and_results['down_cost_for_10_years'] = down_cost_per_year * 10
    params_and_results['cached_ssd_repair_cost_for_10_years'] = cached_ssd_repair_cost_per_year * 10
    params_and_results['uncached_ssd_repair_cost_for_10_years'] = uncached_ssd_repair_cost_per_year * 10
    params_and_results['total_cost_for_10_years'] = avg_initial_cost + operation_cost_per_year * 10
    params_and_results['effective_capacity'] = \
        params_and_results['capacity'] * (params_and_results['total_ssds'] - params_and_results['cached_ssds']) * params_and_results['m'] / (params_and_results['m'] + params_and_results['k'] + params_and_results['l']) * \
        params_and_results['network_m'] / (params_and_results['network_m'] + params_and_results['network_k'] + params_and_results['network_l'])
    params_and_results['initial_cost_per_gb'] = params_and_results['initial_cost'] / params_and_results['effective_capacity'] * 1_000_000_000 / options['network_nodes']
    params_and_results['cost_per_gb'] = params_and_results['total_cost_for_10_years'] / params_and_results['effective_capacity'] * 1_000_000_000 / options['network_nodes']
        
    params_and_results['avg_time_for_rebuilding'] = avg_time_for_rebuilding
    params_and_results['cached_mttf'] = cached_mttf_output
    params_and_results['mttf'] = mttf_output

    print (params_and_results)
    if (params_and_results['network_k'] == 0):
        # open "last_avail.txt" file and write the availability
        with open(options["avail_file"], "w") as f:
            f.write(str(params_and_results['uncached_availability']) + "\n")
            f.write(str(params_and_results['cached_availability']) + "\n")        

def nine_to_credit(nine):
    if (nine >= 6):
        return 0
    elif (nine >= 5 and nine < 6):
        return 0.03
    elif (nine >= 4 and nine < 5):
        return 0.05
    elif (nine >= 3.301 and nine < 4):
        return 0.1
    elif (nine >= 3 and nine < 3.301):
        return 0.25
    return 1

def simulation_per_core(simulation_idx, params_and_results, graph_structure_origin, batch_size, options, costs, queue):
    m = params_and_results['m']
    k = params_and_results['k']
    l = params_and_results['l']
    total_ssds = params_and_results['total_ssds']
    cached_m = params_and_results['cached_m']
    cached_k = params_and_results['cached_k']
    cached_l = params_and_results['cached_l']
    cached_network_m = params_and_results['cached_network_m']
    cached_network_k = params_and_results['cached_network_k']
    cached_network_l = params_and_results['cached_network_l']
    cached_ssds = params_and_results['cached_ssds']
    dwpd = params_and_results['dwpd']
    cached_dwpd_limit = params_and_results['cached_dwpd_limit']
    dwpd_limit = params_and_results['dwpd_limit']
    cached_write_ratio = params_and_results['cached_write_ratio']
    guaranteed_years = params_and_results['guaranteed_years']
    read_bw = params_and_results['ssd_read_bw']
    write_bw = params_and_results['ssd_write_bw']
    cached_read_bw = params_and_results['cached_ssd_read_bw']
    cached_write_bw = params_and_results['cached_ssd_write_bw']
    cached_read_latency = utils.convert_to_microseconds(params_and_results['cached_ssd_read_latency'])
    capacity = params_and_results['capacity']
    use_tbwpd = params_and_results['use_tbwpd']
    tbwpd = params_and_results['tbwpd']
    network_m = params_and_results['network_m']
    network_k = params_and_results['network_k']
    network_l = params_and_results['network_l']
    df = params_and_results['df']
    write_through = params_and_results['write_through']
    inter_replicas = params_and_results['inter_replicas']
    intra_replicas = params_and_results['intra_replicas']
    op_ratio = params_and_results['op_ratio']
    waf_ratio = params_and_results['waf_ratio']
    
    cached_ssd_cost = options['tlc_cost_per_gb'] / 1_000_000_000 * capacity
    uncached_ssd_cost = options['tlc_cost_per_gb'] / 1_000_000_000 * capacity
    if (params_and_results['qlc_cache'] == True):
        cached_ssd_cost = options['qlc_cost_per_gb'] / 1_000_000_000 * capacity
    if params_and_results['qlc'] == True:
        uncached_ssd_cost = options['qlc_cost_per_gb'] / 1_000_000_000 * capacity
    ssd_read_latency, tcp_stack_latency = get_latencies_from_options(options, params_and_results)
    network_latency = utils.convert_to_microseconds(options["network_latency"])
    # In static analysis, standby_ssd is not considered
    mttf = guaranteed_years * 365 * 24 * dwpd_limit / dwpd
    if (use_tbwpd):
        mttf = guaranteed_years * 365 * 24 * (dwpd_limit * capacity / 1_000_000_000_000) / tbwpd
    network_availability_table = {}
    cached_mttf = guaranteed_years * 365 * 24 * cached_dwpd_limit / dwpd
    
    # convert cached_dwpd as op ratio
    if (waf_ratio > 0):
        cached_dwpd_limit = cached_dwpd_limit * 6.2 / waf_ratio
    else:
        cached_dwpd_limit = cached_dwpd_limit * utils.get_waf_from_op(0.07) / utils.get_waf_from_op(op_ratio)
    print ("op_ratio : ", op_ratio, " waf : ", utils.get_waf_from_op(op_ratio))
    # for cached ssds
    if (cached_m > 0):
        # total tbwpd is calculated by the sum of the tbwpd of cached and uncached ssds for fair comparison
        total_tbwpd = capacity * total_ssds * dwpd / 1_000_000_000_000
        if (use_tbwpd):
            total_tbwpd = tbwpd
        #effective_total_tbwpd = total_tbwpd * (cached_ssds / total_ssds * effective_capacity_for_cached + (total_ssds - cached_ssds) / total_ssds * effective_capacity_for_uncached)
        if (write_through):
            cached_tbwpd = total_tbwpd / cached_ssds
            uncached_tbwpd = total_tbwpd / (total_ssds - cached_ssds)
        elif (cached_write_ratio > 0):
            #cached_tbwpd = total_tbwpd * cached_write_ratio / cached_ssds
            cached_tbwpd = total_tbwpd / cached_ssds
            uncached_tbwpd = total_tbwpd * (1 - cached_write_ratio) / (total_ssds - cached_ssds)
        cached_mttf = guaranteed_years * 365 * 24 * (cached_dwpd_limit * capacity / 1_000_000_000_000) / cached_tbwpd
        mttf = guaranteed_years * 365 * 24 * (dwpd_limit * capacity / 1_000_000_000_000) / uncached_tbwpd
            #print (cached_mttf, mttf, cached_tbwpd, uncached_tbwpd, total_tbwpd)
    

    ssd_redun_scheme = ssd.SSDRedundancyScheme(write_bw, read_bw, ssd_read_latency, mttf, cached_write_ratio, cached_write_bw, cached_read_bw, cached_read_latency, cached_mttf, m, k, l, cached_m, cached_k, cached_l, network_m, network_k, network_l, cached_network_m, cached_network_k, cached_network_l, cached_ssds, total_ssds, inter_replicas, intra_replicas)
    
    # only 20% of the bandwidth is used for writing

    hardware_graph = copy.deepcopy(graph_structure_origin)
    node_to_module_map, enclosure_to_node_map, max_read_performance_without_any_failure = initialize_simulation(hardware_graph, read_bw, total_ssds, options)
    flows_and_speed_table = {}
    disconnected_table = {}
    failed_hardware_graph_table = {}
    effective_availabilities = defaultdict(int)
    latencies = defaultdict(int)
    cached_latencies = defaultdict(int)
    module_failure_counts = defaultdict(int)

    total_up_time = 0
    total_cached_up_time = 0
    total_credit_up_time = 0
    total_time = 0
    total_effective_up_time = 0
    completed = 0

    total_cached_ssd_repair_cost = 0
    total_uncached_ssd_repair_cost = 0

    total_time_for_rebuilding_ssd0 = 0
    count_for_rebuilding_ssd0 = 0
    last_timestamp_for_rebuilding_ssd0 = 0
    
    # open avail file and read the availability
    availability_without_network_parity = 0
    availability_without_network_parity_for_cached_ssds = 0
    try:
        with open(options["avail_file"], "r") as f:
            availability_without_network_parity = float(f.readline())
            try:
                availability_without_network_parity_for_cached_ssds = float(f.readline())
            except:
                assert (cached_network_m == 0)
    except:
        assert (network_k == 0)
    
    generate_network_failure_table(network_m + network_k, availability_without_network_parity, availability_without_network_parity_for_cached_ssds, network_availability_table)

    # to reduce remaining time from simulation, we just batch_size as 1 and multiply the simulation time by batch_size
    
    SSDs = {}
    for index in range(0, total_ssds):
        SSDs[index] = {}
        SSDs[index]['failed'] = False
        SSDs[index]['remaining_prep_time_for_rebuilding'] = 0
        SSDs[index]['remaining_capacity_to_rebuild'] = 0
        SSDs[index]['rebuild_speed'] = 0

    up_time = 0
    cached_up_time = 0
    effective_up_time = 0
    credit_up_time = 0
    timestamp = 0
    failed_events = []
    repair_events = []
    
    failed_nodes_and_enclosures = OrderedDict()
    failure_info_per_ssd_group = OrderedDict()
    for node in list(hardware_graph.G.nodes()):
        failed_nodes_and_enclosures[node] = False
    for group_index in range(0, ssd_redun_scheme.get_total_group_count()):
        failure_info_per_ssd_group[group_index] = {}
        failure_info_per_ssd_group[group_index]['failure_count'] = 0
        failure_info_per_ssd_group[group_index]['network_failure_count'] = 0
        
    # Generate failure and repair events
    failed_events = generate_first_failure_events(hardware_graph, node_to_module_map, total_ssds, ssd_redun_scheme)
    initial_cost = get_initial_cost(hardware_graph, node_to_module_map, total_ssds, ssd_redun_scheme, cached_ssd_cost, uncached_ssd_cost, costs, options)
    total_cost = initial_cost
    
    prev_time = 0
    # maximum flow for initial state
    
    last_disconnected = {}
    last_disconnected['local_module'] = False
    last_disconnected['common_module'] = False

    key1 = get_key1(failed_nodes_and_enclosures)
    key2 = get_key2(failed_nodes_and_enclosures, failure_info_per_ssd_group, ssd_redun_scheme)
    calculate_hardware_graph(hardware_graph, failed_nodes_and_enclosures, enclosure_to_node_map, options, failed_hardware_graph_table, disconnected_table, key1)
    calculate_flows_and_speed(df, failed_hardware_graph_table[key1], failure_info_per_ssd_group, ssd_redun_scheme, tcp_stack_latency, network_latency, options, flows_and_speed_table, max_read_performance_without_any_failure, write_through, disconnected_table[key1], key2)
    simulation_hours = options["simulation_years"] * 365 * 24
    percentage = percentage_increasing = 1
    # Process events
    while(1):    
        event_time, event_type, event_node, _ = pop_event(failed_events, repair_events)
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
        availability_ratio = flows_and_speed_entry['availability_ratio']['availability']
        cached_availability_ratio = flows_and_speed_entry['availability_ratio']['cached_availability']
        latency_entries = flows_and_speed_entry['latencies']
        effective_availability_ratio = flows_and_speed_entry['eff_availability_ratio']
        credit_availability_ratio = flows_and_speed_entry['credit_availability_ratio']
        effective_up_time += time_diff * effective_availability_ratio
        up_time += time_diff * availability_ratio
        cached_up_time += time_diff * cached_availability_ratio
        credit_up_time += time_diff * credit_availability_ratio
        timestamp += time_diff
        effective_availabilities[effective_availability_ratio] += time_diff
        for key, each_latency in latency_entries.items():
            if (key == 'latencies'):
                for latency in each_latency:
                    latencies[latency] += each_latency[latency] * time_diff
            elif (key == 'cached_latencies'):
                for latency in each_latency:
                    cached_latencies[latency] += each_latency[latency] * time_diff
        if (break_flag == True):
            break

        update_failure_info(event_type, event_node, failure_info_per_ssd_group, failed_nodes_and_enclosures, ssd_redun_scheme)
        network_state_changed = update_network_state(failure_info_per_ssd_group, ssd_redun_scheme, network_availability_table)

        key1 = get_key1(failed_nodes_and_enclosures)

        calculate_hardware_graph(hardware_graph, failed_nodes_and_enclosures, enclosure_to_node_map, options, failed_hardware_graph_table, disconnected_table, key1)
        disconnected = disconnected_table[key1]

        key2 = get_key2(failed_nodes_and_enclosures, failure_info_per_ssd_group, ssd_redun_scheme)
        calculate_flows_and_speed(df, failed_hardware_graph_table[key1], failure_info_per_ssd_group, ssd_redun_scheme, tcp_stack_latency, network_latency, options, flows_and_speed_table, max_read_performance_without_any_failure, write_through, disconnected, key2)
        if (ssd.SSD_module_name in event_node):
            update_ssd_state(event_node, failure_info_per_ssd_group, SSDs, capacity, event_type, options["prep_time_for_rebuilding"], ssd_redun_scheme, disconnected)
        if (frozenset(last_disconnected.items()) != frozenset(disconnected.items()) or network_state_changed):
            update_all_ssd_states(failure_info_per_ssd_group, SSDs, ssd_redun_scheme, disconnected)
            last_disconnected = disconnected
    
        #if (not leaf_node_module in event_node):
        if event_type == 'fail':
            if (ssd.SSD_module_name in event_node and ssd.get_ssd_index(event_node) == 0):
                last_timestamp_for_rebuilding_ssd0 = event_time
            elif (not ssd.SSD_module_name in event_node):
                module_failure_counts[node_to_module_map[event_node]] += 1
            push_repair_event(repair_events, event_node, event_time, node_to_module_map, hardware_graph)
        if event_type == 'repair':
            if (ssd.SSD_module_name in event_node and ssd.get_ssd_index(event_node) == 0):
                total_time_for_rebuilding_ssd0 += event_time - last_timestamp_for_rebuilding_ssd0
                count_for_rebuilding_ssd0 += 1
            module_cost = calculate_module_cost(event_node, node_to_module_map, costs, ssd_redun_scheme, cached_ssd_cost, uncached_ssd_cost, options)
            if (ssd.SSD_module_name in event_node):
                if (ssd_redun_scheme.is_ssd_index_cached(ssd.get_ssd_index(event_node))):
                    total_cached_ssd_repair_cost += module_cost
                else:
                    total_uncached_ssd_repair_cost += module_cost
            total_cost += module_cost
            push_failed_event(failed_events, event_node, event_time, node_to_module_map, hardware_graph, ssd_redun_scheme)

        flows_and_speed_entry = flows_and_speed_table[key2]
        update_repair_event_for_SSDs(repair_events, event_time, SSDs, flows_and_speed_entry, failure_info_per_ssd_group, ssd_redun_scheme)
      
        prev_time = event_time
        if (simulation_idx == 0 and (event_time * 100) // (simulation_hours * batch_size) > percentage):
            print ("completed ", percentage ,"%")
            percentage += percentage_increasing

    # Calculate the availability
    total_up_time += up_time
    total_cached_up_time += cached_up_time
    total_time += timestamp
    total_effective_up_time += effective_up_time
    total_credit_up_time += credit_up_time
    completed += 1
    assert (timestamp >= simulation_hours * batch_size - 1 and timestamp <= simulation_hours * batch_size + 1)

    queue.put((total_up_time, total_cached_up_time, total_credit_up_time, total_time, total_effective_up_time, effective_availabilities, latencies, cached_latencies, initial_cost, total_cost, total_time_for_rebuilding_ssd0, count_for_rebuilding_ssd0, cached_mttf, mttf, total_cached_ssd_repair_cost, total_uncached_ssd_repair_cost))
    
    return ""
    