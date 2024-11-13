import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import encoding_time_data
from static_analysis import test_static_analyze_ssd_only
from graph_structure import GraphStructure
import simulation as sim

import argparse

# https://blog.synology.com/tlc-vs-qlc-ssds-what-are-the-differences

def parse_arguments():
    parser = argparse.ArgumentParser(description='SSD only analysis\' parameters.')
    parser.add_argument('--total_ssds', type=int, default=48, help='Total number of SSDs')
    parser.add_argument('--m', type=int, default=15, help='Number of Data chunks')
    parser.add_argument('--k', type=int, default=1, help='Number of Parity chunks')
    parser.add_argument('--total_network_nodes', type=int, default=16, help='Total number of network nodes')
    parser.add_argument('--network_m', type=int, default=16, help='Number of Data chunks in network')
    parser.add_argument('--network_k', type=int, default=0, help='Number of Parity chunks in network')
    parser.add_argument('--capacity', type=int, default=64_000_000_000_000, help='Capacity of SSDs')
    parser.add_argument('--qlc', action='store_true', help='Flag to indicate if QLC SSDs are used. default is TLC')
    parser.add_argument('--simulation', action='store_true', help='Flag to indicate if simulation is being run')
    parser.add_argument('--dwpd', type=float, default=1, help='DWPD (Drive writes per day) of SSDs')
    parser.add_argument('--tbwpd', type=float, default=4.5, help='TB writes per day of SSDs.')
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
capacity = args.capacity
qlc = args.qlc
simulation = args.simulation
dwpd = args.dwpd
output_file = args.output_file
tbwpd = args.tbwpd
use_tbwpd = args.use_tbwpd
simulation = args.simulation
total_network_nodes = args.total_network_nodes

tlc_dwpd = 1
qlc_dwpd = 0.26
tlc_write_bw = 7_064_000_000 # sequential write bandwidth of TLC SSDs
qlc_write_bw = 1_400_000_000
tlc_read_bw = 12_000_000_000
qlc_read_bw = 12_000_000_000

write_bw = qlc_write_bw if qlc else tlc_write_bw
read_bw = qlc_read_bw if qlc else tlc_read_bw
guaranteed_years = args.guarnanteed_years
dwpd_limit = qlc_dwpd if qlc else tlc_dwpd
simulation = args.simulation

results = {}
results['total_ssds'] = total_ssds
results['m'] = m
results['k'] = k
results['network_m'] = args.network_m
results['network_k'] = args.network_k
results['capacity'] = capacity
results['qlc'] = qlc
results['simulation'] = simulation
results['dwpd'] = dwpd
results['guaranteed_years'] = guaranteed_years
results['dwpd_limit'] = dwpd_limit
results['use_tbwpd'] = use_tbwpd
results['tbwpd'] = tbwpd
results['simulation'] = simulation
results['total_network_nodes'] = total_network_nodes

n = m + k
if (n > total_ssds):
    raise ValueError('The sum of m, kvshould not exceed total_ssds')
if (total_ssds % (n) != 0):
    raise ValueError('total_ssds should be divisible by the sum of m, k')

network_m = args.network_m
network_k = args.network_k
network_n = network_m + network_k


df = pd.DataFrame(encoding_time_data)

def output_results():
    global results, output_file
    with open(output_file, 'a') as f:
        for key in results:
            f.write(f'{key} | {results[key]} | ')
        f.write('\n')

if __name__ == "__main__":
    global batch_size
    if (simulation):
        edges, enclosures, redundancies, mttfs, mtrs, options = GraphStructure.parse_input_from_json(args.graph_structure_file)
        hardware_graph = GraphStructure(edges, enclosures, redundancies, mttfs, mtrs)
        batch_size = 40000
        sim.monte_carlo_simulation(guaranteed_years, use_tbwpd, tbwpd, dwpd_limit, capacity, dwpd, results, m, k, total_ssds, network_m, network_k, df, write_bw, read_bw, hardware_graph, batch_size, options)
        print (edges, enclosures, redundancies, mttfs, mtrs)
    else:
        test_static_analyze_ssd_only(guaranteed_years, use_tbwpd, tbwpd, dwpd_limit, capacity, dwpd, results, m, k, n, df, write_bw)
    output_results()