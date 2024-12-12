import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import encoding_time_data
from utils import KMG_to_bytes
from static_analysis import test_static_analyze_ssd_only
from graph_structure import GraphStructure
import simulation as sim
import json

import argparse

# https://blog.synology.com/tlc-vs-qlc-ssds-what-are-the-differences

def parse_input_from_json(file_path):
    """
    JSON 파일로부터 데이터를 파싱하여 edges, enclosures, availabilities, redundancies, mttfs, mtrs를 반환합니다.
    
    Parameters:
        file_path (str): JSON 파일의 경로.
        
    Returns:
        tuple: (edges, enclosures, availabilities, redundancies, mttfs, mtrs)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Edges: List of tuples (start, end, weight)
    edges = []
    for edge in data.get("edges", []):
        start = edge.get("start")
        end = edge.get("end")
        weight = edge.get("bandwidth")
        if start is not None and end is not None and weight is not None:
            edges.append((start, end, weight))
    
    # Enclosures: Dict of enclosure name to list of nodes
    enclosures = data.get("enclosures", {})
            
    # MTTFs: Dict of module to MTTF
    mttfs = data.get("mttf", {})
    
    # MTRs: Dict of module to MTR
    mtrs = data.get("mtr", {})
    
    # Redundancies: Dict of module to tuple (M, K)
    
    options = data.get("options", {})
    return edges, enclosures, mttfs, mtrs, options

def parse_arguments():
    parser = argparse.ArgumentParser(description='SSD only analysis\' parameters.')
    parser.add_argument('--total_ssds', type=int, default=48, help='Total number of SSDs')
    parser.add_argument('--m', type=int, default=15, help='Number of Data chunks')
    parser.add_argument('--k', type=int, default=1, help='Number of Parity chunks')
    parser.add_argument('--cached_ssds', type=int, default=0, help='Total number of SSDs in cache tier')
    parser.add_argument('--cached_m', type=int, default=0, help='Number of Data chunks in cache tier')
    parser.add_argument('--cached_k', type=int, default=0, help='Number of Parity chunks in cache tier')
    parser.add_argument('--cached_network_m', type=int, default=0, help='Number of Data chunks in network cache tier')
    parser.add_argument('--cached_network_k', type=int, default=0, help='Number of Parity chunks in network cache tier')
    parser.add_argument('--inter_replicas', type=int, default=0, help='Number of network copys')
    parser.add_argument('--intra_replicas', type=int, default=0, help='Number of local copys')
    parser.add_argument('--cached_write_ratio', type=float, default=0, help='Cached write ratio relative to total write')
    parser.add_argument('--cached_read_ratio', type=float, default=0.8, help='Cached read ratio relative to total write')
    parser.add_argument('--write_through', action='store_true', help='Flag to indicate if write through is used')
    parser.add_argument('--total_network_nodes', type=int, default=6, help='Total number of network nodes')
    parser.add_argument('--network_m', type=int, default=6, help='Number of Data chunks in network')
    parser.add_argument('--network_k', type=int, default=0, help='Number of Parity chunks in network')
    parser.add_argument('--capacity', type=int, default=64_000_000_000_000, help='Capacity of SSDs')
    parser.add_argument('--qlc', action='store_true', help='Flag to indicate if QLC SSDs are used. default is TLC')
    parser.add_argument('--simulation', action='store_true', help='Flag to indicate if simulation is being run')
    parser.add_argument('--dwpd', type=float, default=1, help='DWPD (Drive writes per day) of SSDs. Writes amount for cached tier if cached tier is used')
    parser.add_argument('--tbwpd', type=float, default=4.5, help='TB writes per day of SSDs. Writes amount for cached tier if cached tier is used')
    parser.add_argument('--use_tbwpd', action='store_true', help='Flag to indicate if TB writes per day is used instead of DWPD')
    parser.add_argument('--guarnanteed_years', type=int, default=5, help='Guaranteed years of SSDs')
    parser.add_argument('--graph_structure_file', type=str, default='graph.json', help='Graph structure file path')
    parser.add_argument('--output_file', type=str, default='results.txt', help='Output file path to save results')
    args = parser.parse_args()
    return args

args = parse_arguments()

total_ssds = args.total_ssds
m = args.m
k = args.k

cached_ssds = args.cached_ssds
cached_m = args.cached_m
cached_k = args.cached_k
cached_network_m = args.cached_network_m
cached_network_k = args.cached_network_k
if (cached_network_m == 0):
    cached_network_m = args.network_m
inter_replicas = args.inter_replicas
intra_replicas = args.intra_replicas

cached_write_ratio = args.cached_write_ratio
capacity = args.capacity
qlc = args.qlc
simulation = args.simulation
dwpd = args.dwpd
output_file = args.output_file
tbwpd = args.tbwpd
use_tbwpd = args.use_tbwpd
simulation = args.simulation
total_network_nodes = args.total_network_nodes

edges, enclosures, mttfs, mtrs, options = parse_input_from_json(args.graph_structure_file)
hardware_graph = GraphStructure(edges, enclosures, mttfs, mtrs)

qlc_write_bw = KMG_to_bytes(options['qlc_write_bw'])
qlc_read_bw = KMG_to_bytes(options['qlc_read_bw'])
qlc_dwpd = options['qlc_dwpd_limit']
tlc_write_bw = KMG_to_bytes(options['tlc_write_bw'])
tlc_read_bw = KMG_to_bytes(options['tlc_read_bw'])
tlc_dwpd = options['tlc_dwpd_limit']

write_bw = qlc_write_bw if qlc else tlc_write_bw
read_bw = qlc_read_bw if qlc else tlc_read_bw
cached_write_bw = write_bw * cached_write_ratio
guaranteed_years = args.guarnanteed_years
dwpd_limit = qlc_dwpd if qlc else tlc_dwpd
simulation = args.simulation

n = m + k

if (n > total_ssds):
    raise ValueError('The sum of m, kvshould not exceed total_ssds')
if ((total_ssds - cached_ssds) % (n) != 0):
    raise ValueError('total_ssds should be divisible by the sum of m, k')

if (args.write_through):
    if (cached_ssds == 0):
        raise ValueError('Write through should be used with cached tier')
    if (cached_write_ratio != 0):
        raise ValueError('Do not use cached_write_ratio with write through')

if (cached_ssds > 0):
    if ((cached_write_ratio == 0 or cached_write_ratio >= 1) and not args.write_through):
        raise ValueError('cached_write_ratio should be between 0 and 1')
    if (cached_m + cached_k > cached_ssds):
        raise ValueError('The sum of cached_m, cached_k should not exceed cached_ssds')
    if (inter_replicas > 0 and cached_network_m != cached_network_k):
        raise ValueError('cached_network_m should be equal to cached_network_k when inter_replicas > 0')
    if (intra_replicas > 0 and cached_m != cached_k):
        raise ValueError('cached_m should be equal to cached_k when intra_replicas > 0')
    if (cached_ssds % (cached_m + cached_k) != 0):
        raise ValueError('cached_ssds should be divisible by the sum of cached_m, cached_k')

network_m = args.network_m
network_k = args.network_k
network_n = network_m + network_k

params_and_results = {}
params_and_results['total_ssds'] = total_ssds
params_and_results['m'] = m
params_and_results['k'] = k
params_and_results['cached_ssds'] = cached_ssds
params_and_results['cached_m'] = cached_m
params_and_results['cached_k'] = cached_k
params_and_results['cached_network_m'] = cached_network_m
params_and_results['cached_network_k'] = cached_network_k
params_and_results['inter_replicas'] = inter_replicas
params_and_results['intra_replicas'] = intra_replicas
params_and_results['cached_write_ratio'] = cached_write_ratio
params_and_results['network_m'] = network_m
params_and_results['network_k'] = network_k
params_and_results['capacity'] = capacity
params_and_results['qlc'] = qlc
params_and_results['simulation'] = simulation
params_and_results['dwpd'] = dwpd
params_and_results['guaranteed_years'] = guaranteed_years
params_and_results['dwpd_limit'] = dwpd_limit
params_and_results['cached_dwpd_limit'] = tlc_dwpd
params_and_results['use_tbwpd'] = use_tbwpd
params_and_results['tbwpd'] = tbwpd
params_and_results['simulation'] = simulation
params_and_results['total_network_nodes'] = total_network_nodes
params_and_results['ssd_read_bw'] = read_bw
params_and_results['ssd_write_bw'] = write_bw
params_and_results['cached_ssd_read_bw'] = tlc_read_bw
params_and_results['cached_ssd_write_bw'] = tlc_write_bw
params_and_results['cached_ssd_read_latency'] = options['tlc_read_latency']
params_and_results['cached_read_ratio'] = args.cached_read_ratio
params_and_results['write_through'] = args.write_through

df = pd.DataFrame(encoding_time_data)

params_and_results['df'] = df


def output_params_and_results():
    global params_and_results, output_file
    del params_and_results["df"]
    with open(output_file, 'a') as f:
        for key in params_and_results:
            f.write(f'{key} | {params_and_results[key]} | ')
        f.write('\n')

if __name__ == "__main__":
    global batch_size
    if (simulation):
        batch_size = 20000
        sim.monte_carlo_simulation(params_and_results, hardware_graph, batch_size, options)
        print (edges, enclosures, mttfs, mtrs)
    else:
        test_static_analyze_ssd_only(guaranteed_years, use_tbwpd, tbwpd, dwpd_limit, capacity, dwpd, params_and_results, m, k, n, df, write_bw)
    output_params_and_results()