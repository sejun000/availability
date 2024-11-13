import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from collections import OrderedDict
import heapq
import random
import networkx as nx

SSD_module_name = "SSD"
network_module_name = "network"
SSD_Enclosure = "NVMeEnclosure"

backup_reconstruction_speed = 200_000_000

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

def get_network_group_name(network_index):
    return f"{network_module_name}_group_{network_index}"

def get_ssd_group_index_from_network_group_name(network_group_name):
    return int(network_group_name[len(f"{network_module_name}_group"):])

def get_ssd_group_name(ssd_group_index):
    return f"{SSD_module_name}_group_{ssd_group_index}"

def get_ssd_name(ssd_index):
    return f"{SSD_module_name}{ssd_index}"

def get_ssd_index(ssd_name):
    return int(ssd_name[len(SSD_module_name):])

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
    disconnected_table[key1] = not connected
    failed_hardware_graph_table[key1] = hardware_graph_copy

def is_catastrophic_failure(failure_info, ssd_k, disconnected):
    return failure_info['failure_count'] > ssd_k or disconnected

def judge_state_from_failure_info(failure_info, ssd_k, network_k, disconnected):
    if failure_info['failure_count'] > 0 and (not is_catastrophic_failure(failure_info, ssd_k, disconnected)):
        return SSD_state_intra_rebuilding
    elif (is_catastrophic_failure(failure_info, ssd_k, disconnected) and failure_info['network_failure_count'] <= network_k - 1):
        return SSD_state_inter_rebuilding
    elif (failure_info['network_failure_count'] > 0 and (not is_catastrophic_failure(failure_info, ssd_k, disconnected)) and failure_info['network_failure_count'] <= network_k):
        return SSD_state_inter_degraded
    elif (failure_info['network_failure_count'] > network_k or (is_catastrophic_failure(failure_info, ssd_k, disconnected) and failure_info['network_failure_count'] > network_k - 1)):
        return SSD_state_data_loss
# input : failed_hardware_graph_table, ssd groups' failure
# output : flows_and_speed_table
def calculate_flows_and_speed(df, hardware_graph_copy, failure_info_per_ssd_group, ssd_m, ssd_k, ssd_total_count, network_m, network_k, ssd_read_bw, ssd_write_bw, options, flows_and_speed_table, key2):
    # local rebuild
    if (key2 in flows_and_speed_table):
        return
    total_read_bw_for_ssds = 0
    tables = {}
    tables['intra_rebuilding_bw'] = {}
    tables['inter_rebuilding_bw'] = {}
    tables['max_read_performance'] = {}
    availability_ratio = 1

    hardware_graph_copy.add_virtual_nodes(options["start_module"], options["end_module"])
    hardware_max_flow = nx.maximum_flow_value(hardware_graph_copy.G, 'virtual_source', 'virtual_sink')
    hardware_graph_copy.remove_virtual_nodes()
    hardware_graph_copy.add_virtual_nodes(options["start_module"], options["lowest_common_module"])
    common_module_max_flow = nx.maximum_flow_value(hardware_graph_copy.G, 'virtual_source', 'virtual_sink')
    hardware_graph_copy.remove_virtual_nodes()

    normal_ssds_count = 0
    total_ssds_count = 0
    for failure_info in failure_info_per_ssd_group.values():
        degraded_ssd_count = ssd_m - failure_info['failure_count']
        total_read_bw_for_ssds += ssd_read_bw * degraded_ssd_count
        normal_ssds_count += degraded_ssd_count
        total_ssds_count += ssd_m + ssd_k
    # calculate bottlneck read performance to calculate degraded and rebuilding speed
    bottleneck_read_bw = min(total_read_bw_for_ssds, hardware_max_flow)
    # common module's flow is shared by all network nodes, so we need to divide it by the number of network nodes
    bottleneck_read_bw = min(bottleneck_read_bw, common_module_max_flow / (network_m + network_k))
    bottleneck_read_bw_per_ssd = bottleneck_read_bw / normal_ssds_count
  #  print (bottleneck_read_bw, total_read_bw_for_ssds, common_module_max_flow, hardware_max_flow)
    
  #  print (bottleneck_read_bw, total_read_bw_for_ssds)
    disconnected = (bottleneck_read_bw == 0)
    for failure_info in failure_info_per_ssd_group.values():
        # failed ssd shall be read from the other ssds
        if judge_state_from_failure_info(failure_info, ssd_k, network_k, disconnected) == SSD_state_intra_rebuilding:
            degraded_bw = calculate_bottleneck_speed(df, ssd_m, failure_info['failure_count'], [ssd_read_bw], options)
            rebuilding_bw = calculate_bottleneck_speed(df, ssd_m, failure_info['failure_count'], [ssd_read_bw, ssd_write_bw], options)
            tables['intra_rebuilding_bw'][failure_info['failure_count']] = rebuilding_bw
            degraded_ssd_count = ssd_m - failure_info['failure_count']
            total_read_bw_for_ssds = total_read_bw_for_ssds - (ssd_read_bw - degraded_bw) * degraded_ssd_count

        # catastrophic failure for the simulated nodes
        elif judge_state_from_failure_info(failure_info, ssd_k, network_k, disconnected) == SSD_state_inter_rebuilding:
            # failed ssd shall be read from the other network
            degraded_bw = calculate_bottleneck_speed(df, network_m, failure_info['network_failure_count'] + 1, [bottleneck_read_bw_per_ssd], options)
            rebuilding_bw = calculate_bottleneck_speed(df, network_m, failure_info['network_failure_count'] + 1, [bottleneck_read_bw_per_ssd, ssd_write_bw], options)
            tables['inter_rebuilding_bw'][failure_info['network_failure_count'] + 1] = rebuilding_bw
            # read from catastrophic ssd is same as the bottleneck read bw

        # catastrophic failure for the other nodes
        elif judge_state_from_failure_info(failure_info, ssd_k, network_k, disconnected) == SSD_state_inter_degraded:
             # failed ssd shall be read from this network to rebuild
            degraded_bw = calculate_bottleneck_speed(df, network_m, failure_info['network_failure_count'], [bottleneck_read_bw_per_ssd], options)
            rebuilding_bw = calculate_bottleneck_speed(df, network_m, failure_info['network_failure_count'], [bottleneck_read_bw_per_ssd, ssd_write_bw], options)
            tables['inter_rebuilding_bw'][failure_info['network_failure_count']] = rebuilding_bw
            degraded_ssd_count = ssd_m - failure_info['failure_count']
            bottleneck_read_bw = bottleneck_read_bw - (bottleneck_read_bw_per_ssd - degraded_bw) * degraded_ssd_count

        # data loss because of catastrophic failure is more than network_k
        elif judge_state_from_failure_info(failure_info, ssd_k, network_k, disconnected) == SSD_state_data_loss:
            tables['inter_rebuilding_bw'][failure_info['network_failure_count']] = backup_reconstruction_speed
            degraded_ssd_count = ssd_m - failure_info['failure_count']
            # all read is removed from data loss failure 
            bottleneck_read_bw = bottleneck_read_bw - degraded_ssd_count * bottleneck_read_bw_per_ssd
            availability_ratio -= (ssd_m + ssd_k) / ssd_total_count
    
    max_read_performance = min(bottleneck_read_bw, total_read_bw_for_ssds)
    tables['max_read_performance'] = max_read_performance
    tables['availabiltiy_ratio'] = availability_ratio
    flows_and_speed_table[key2] = tables

def generate_first_failure_events(hardware_graph, node_to_module_map, ssd_total_count, n, ssd_mttf, network_mttf_table):
    events = []
    for node in list(hardware_graph.G.nodes()):
        push_failed_event(events, node, 0, node_to_module_map, hardware_graph, ssd_mttf, network_mttf_table)
    # Generate enclosure failure and repair events
    for enclosure in list(hardware_graph.enclosures):
        push_failed_event(events, enclosure, 0, node_to_module_map, hardware_graph, ssd_mttf, network_mttf_table)
    for ssd_index in range(0, ssd_total_count):
        push_failed_event(events, get_ssd_name(ssd_index), 0, node_to_module_map, hardware_graph, ssd_mttf, network_mttf_table)
    #for network_index in range(0, ssd_total_count / n):
    #    push_failed_event(events, get_network_group_name(network_index), 0, node_to_module_map, hardware_graph, ssd_mttf, network_mttf_table)
    return events

def calculate_bottleneck_speed(df, m, k, other_bws, options):
    erasure_coding_latency = 0.00001
    if (k > 0):
        erasure_coding_latency = df[(df['n'] == m) & (df['k'] == k)]['Encoding Time'].values[0] / 1000 / 1000
    erasure_coding_speed = 256_000 / erasure_coding_latency
    min_speed = erasure_coding_speed
    for other_bw in other_bws:
        if (other_bw * options["rebuild_bw_ratio"] < min_speed):
            min_speed = other_bw
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
    elif (network_module_name in event_node):
        ssd_group_index = get_ssd_group_index_from_network_group_name(event_node)
        ssd_group_name = get_ssd_group_name(ssd_group_index)
        if (event_type == 'fail'):
            failure_info_per_ssd_group[ssd_group_name]['network_failure_count'] += 1
        elif (event_type == 'repair'):
            failure_info_per_ssd_group[ssd_group_name]['network_failure_count'] -= 1
    else:
        if (event_type == 'fail'):
            failed_nodes_and_enclosures[event_node] = True
        elif (event_type == 'repair'):
            failed_nodes_and_enclosures[event_node] = False


def update_ssd_state(ssd_name, failure_info_per_ssd_group, SSDs, capacity, event_type, prep_time_for_rebuilding, m, k, network_k, disconnected):
    ssd_index = get_ssd_index(ssd_name)
    if (event_type == 'fail'):
        SSDs[ssd_index]['failed'] = True
        SSDs[ssd_index]['remaining_capacity_to_rebuild'] = capacity
        SSDs[ssd_index]['rebuild_speed'] = 0
        SSDs[ssd_index]['remaining_prep_time_for_rebuilding'] = prep_time_for_rebuilding
        
    elif (event_type == 'repair'):
        SSDs[ssd_index]['failed'] = False

    n = m + k
    group_index = ssd_index // n
    failure_info = failure_info_per_ssd_group[get_ssd_group_name(group_index)]
        
    if judge_state_from_failure_info(failure_info, k, network_k, disconnected) == SSD_state_data_loss:
        changed_state = SSD_state_data_loss       
    elif judge_state_from_failure_info(failure_info, k, network_k, disconnected) == SSD_state_inter_degraded:
        changed_state = SSD_state_inter_rebuilding
    elif judge_state_from_failure_info(failure_info, k, network_k, disconnected) == SSD_state_intra_rebuilding:
        changed_state = SSD_state_intra_rebuilding
    else:
        changed_state = SSD_state_normal

    for i in range(group_index * n, group_index * n + n):
        #print (k, failure_info['failure_count'], network_k)
        if SSDs[i]['failed'] == False:
            continue
        #print ("changed", state_updated_ssd_name, changed_state)
        assert changed_state != SSD_state_normal
        SSDs[i]['state'] = changed_state

def push_failed_event(failed_events, event_node, current_time, node_to_module_map, hardware_graph, ssd_mttf, network_mttf):
    if (network_module_name in event_node):
        assert False
        mttf = network_mttf
    elif (SSD_module_name in event_node):
        mttf = ssd_mttf
    else:
        module = node_to_module_map[event_node]
        mttf = hardware_graph.mttfs[module]
    failure_time = current_time + random.expovariate(1 / mttf)
    heapq.heappush(failed_events, (failure_time, 'fail', event_node, current_time))

def push_repair_event(repair_events, event_node, current_time, node_to_module_map, hardware_graph):
    if (SSD_module_name in event_node):
        big_number = 1_000_000_000_000_000
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
            assert remaining_capacity_to_rebuild >= 0
            current_ssd_info['remaining_capacity_to_rebuild'] = remaining_capacity_to_rebuild    
            #print (repair_node, current_time, repair_time, updated_repair_time, case_flag)
        # change rebuild speed to the current state
        rebuild_speed = 0
        ssd_group_index = repaired_ssd_index // n
        ssd_failure_count = failure_info_per_ssd_group[get_ssd_group_name(ssd_group_index)]['failure_count']
        network_failure_count = failure_info_per_ssd_group[get_ssd_group_name(ssd_group_index)]['network_failure_count']
        #print (ssd_index, repair_node, current_ssd_info, flows_and_speed_entry['intra_rebuilding_bw'], failure_info_per_ssd_group)
        if (current_ssd_info['state'] == SSD_state_intra_rebuilding): # it can be reconstructed
            rebuild_speed = flows_and_speed_entry['intra_rebuilding_bw'][ssd_failure_count]
        elif (current_ssd_info['state'] == SSD_state_inter_rebuilding):
            rebuild_speed = flows_and_speed_entry['inter_rebuilding_bw'][network_failure_count]
        elif (current_ssd_info['state'] == SSD_state_data_loss):
            rebuild_speed = backup_reconstruction_speed
        else:
            assert False
        current_ssd_info['rebuild_speed'] = rebuild_speed
        assert (current_ssd_info['remaining_prep_time_for_rebuilding'] >= 0)
        assert (rebuild_speed > 0)
        rebuild_speed_per_hour = rebuild_speed * 3600
        updated_repair_event_time = current_ssd_info['remaining_prep_time_for_rebuilding'] + current_ssd_info['remaining_capacity_to_rebuild'] / rebuild_speed_per_hour
        updated_repair_events.append((current_time + updated_repair_event_time, 'repair', repair_node, current_time))

    for event in updated_repair_events:
        heapq.heappush(repair_events, event)

def initialize_simulation(hardware_graph, ssd_read_bw, total_ssd_count, options, network_n):
    node_to_module_map = {}
    node_to_group_map = {}
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
    for group, (nodes, M) in hardware_graph.redundancy_groups.items():
        for node in nodes:
            node_to_group_map[node] = group
    
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
    
    return node_to_module_map, node_to_group_map, enclosure_to_node_map, ssd_max_read_performance

def get_key1(failed_nodes_and_enclosures):
    return frozenset(failed_nodes_and_enclosures.items())

def get_key2(failed_nodes_and_enclosures, failure_info_per_ssd_group):
    frozen_set_list = []
    for ssd_group_name, failure_info in failure_info_per_ssd_group.items():
        #print (frozenset([ssd_group_name, frozenset(failure_info.items())]))
        frozen_set_list.append(frozenset([ssd_group_name, frozenset(failure_info.items())]))
    return frozenset([get_key1(failed_nodes_and_enclosures), frozenset(frozen_set_list)])

def monte_carlo_simulation(simulation_idx, guaranteed_years, use_tbwpd, tbwpd, dwpd_limit, capacity, dwpd, results, m, k, total_ssd_count, network_m, network_k, df, write_bw, read_bw, graph_structure_origin, batch_size, options):
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
    node_to_module_map, node_to_group_map, enclosure_to_node_map, max_read_performance_without_any_failure = initialize_simulation(hardware_graph, read_bw, total_ssd_count, options, network_m + network_k)
    flows_and_speed_table = {}
    disconnected_table = {}
    failed_hardware_graph_table = {}

    total_up_time = 0
    total_time = 0
    total_effective_up_time = 0
    completed = 0
    last_event_time_stamp_for_first_ssd_group = 0
    last_state_for_first_ssd_group = SSD_state_normal
    first_ssd_group_mttfs = []
    first_ssd_group_mtrs = []

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
    for group, (nodes, M) in hardware_graph.redundancy_groups.items():
        for node in nodes:
            node_to_group_map[node] = group
        
    for _ in range(0, batch_size):
        SSDs = {}
        for index in range(0, total_ssd_count):
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
        for i in range(0, total_ssd_count // n):
            ssd_group_name = get_ssd_group_name(i)
            failure_info_per_ssd_group[ssd_group_name] = {}
            failure_info_per_ssd_group[ssd_group_name]['failure_count'] = 0
            failure_info_per_ssd_group[ssd_group_name]['network_failure_count'] = 0
           
        # Generate failure and repair events
        failed_events = generate_first_failure_events(hardware_graph, node_to_module_map, total_ssd_count, n, ssd_mttf, network_mttf_table)
        prev_time = 0
        # maximum flow for initial state
        key1 = get_key1(failed_nodes_and_enclosures)
        key2 = get_key2(failed_nodes_and_enclosures, failure_info_per_ssd_group)
        calculate_hardware_graph(hardware_graph, failed_nodes_and_enclosures, enclosure_to_node_map, options, failed_hardware_graph_table, disconnected_table, key1)
        calculate_flows_and_speed(df, failed_hardware_graph_table[key1], failure_info_per_ssd_group, m, k, total_ssd_count, network_m, network_k, read_bw, write_bw, options, flows_and_speed_table, key2)
        simulation_hours = options["simulation_years"] * 365 * 24

        # Process events
        while(1):    
            event_time, event_type, event_node, temp = pop_event(failed_events, repair_events)
            # Calculate the time difference from the previous event
            break_flag = False
            if (event_time > simulation_hours):
                event_time = simulation_hours
                break_flag = True
            time_diff = event_time - prev_time
            if (time_diff < 0):
                print (event_type, event_node, event_time, prev_time)
                assert False

            # Calculate the availability
            flows_and_speed_entry = flows_and_speed_table[key2]
            availability_ratio = flows_and_speed_entry['availabiltiy_ratio']
            effective_availability_ratio = flows_and_speed_entry['max_read_performance'] / max_read_performance_without_any_failure
            
            effective_up_time += time_diff * effective_availability_ratio
            up_time += time_diff * availability_ratio
            timestamp += time_diff
    
            if (break_flag == True):
                break
            
            if (last_state_for_first_ssd_group == SSD_state_normal and judge_state_from_failure_info(failure_info_per_ssd_group[get_ssd_group_name(0)], k, network_k, disconnected_table[key1]) == SSD_state_data_loss):
                first_ssd_group_mttfs.append(event_time - last_event_time_stamp_for_first_ssd_group)
                last_event_time_stamp_for_first_ssd_group = event_time
                last_state_for_first_ssd_group = SSD_state_data_loss
            elif (last_state_for_first_ssd_group == SSD_state_data_loss and judge_state_from_failure_info(failure_info_per_ssd_group[get_ssd_group_name(0)], k, network_k, disconnected_table[key1]) != SSD_state_data_loss):
                first_ssd_group_mtrs.append(event_time - last_event_time_stamp_for_first_ssd_group)
                last_event_time_stamp_for_first_ssd_group = event_time
                last_state_for_first_ssd_group = SSD_state_normal
            update_failure_info(event_type, event_node, failure_info_per_ssd_group, failed_nodes_and_enclosures, n)
            if (SSD_module_name in event_node):
                update_ssd_state(event_node, failure_info_per_ssd_group, SSDs, capacity, event_type, options["prep_time_for_rebuilding"], m, k, network_k, disconnected_table[key1])

            key1 = get_key1(failed_nodes_and_enclosures)
            calculate_hardware_graph(hardware_graph, failed_nodes_and_enclosures, enclosure_to_node_map, options, failed_hardware_graph_table, disconnected_table, key1)
            key2 = get_key2(failed_nodes_and_enclosures, failure_info_per_ssd_group)
            calculate_flows_and_speed(df, failed_hardware_graph_table[key1], failure_info_per_ssd_group, m, k, total_ssd_count, network_m, network_k, read_bw, write_bw, options, flows_and_speed_table, key2)

            #if (not leaf_node_module in event_node):
            if event_type == 'fail':
                push_repair_event(repair_events, event_node, event_time, node_to_module_map, hardware_graph)
            if event_type == 'repair':
                push_failed_event(failed_events, event_node, event_time, node_to_module_map, hardware_graph, ssd_mttf, network_mttf_table)

            flows_and_speed_entry = flows_and_speed_table[key2]
            #print (key2, flows_and_speed_entry)
            update_repair_event_for_SSDs(repair_events, event_time, SSDs, flows_and_speed_entry, failure_info_per_ssd_group, n)
            
            prev_time = event_time
            #print (prev_disconnected_ssds)
        # Calculate the availability
        total_up_time += up_time
        total_time += timestamp
        #print (timestamp, up_time, effective_availability_ratio, availability_ratio)
        total_effective_up_time += effective_up_time
        completed += 1
        #if (completed % 10000 == 0):
        if (simulation_idx == 0 and (completed * 100) % batch_size == 0):
            print ("completed "+ str(completed * 100 // batch_size) + "%")
        assert (timestamp >= simulation_hours - 1 and timestamp <= simulation_hours + 1)
    print (total_up_time, total_time, total_effective_up_time, total_up_time / total_time, total_effective_up_time / total_time)
    print (first_ssd_group_mttfs, first_ssd_group_mtrs) 
    #   queue.put((total_up_time, total_down_time, total_effective_up_time, total_an_ssd_up_time, max_flow_cdf))
    return ""
    