import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import encoding_time_data
from utils import KMG_to_bytes
from utils import parse_input_from_json
#from static_analysis import test_static_analyze_ssd_only
from graph_structure import GraphStructure
import simulation as sim

import argparse

# https://blog.synology.com/tlc-vs-qlc-ssds-what-are-the-differences

def parse_arguments():
    parser = argparse.ArgumentParser(description='SSD only analysis\' parameters.')
    parser.add_argument('--total_ssds', type=int, default=48, help='Total number of SSDs')
    parser.add_argument('--m', type=int, default=15, help='Number of Data chunks')
    parser.add_argument('--k', type=int, default=1, help='Number of Parity chunks')
    parser.add_argument('--l', type=int, default=0, help='Number of Remaining chunks, so l is equal to zero, clustered parity is used')
    parser.add_argument('--cached_ssds', type=int, default=0, help='Total number of SSDs in cache tier')
    parser.add_argument('--cached_m', type=int, default=0, help='Number of Data chunks in cache tier')
    parser.add_argument('--cached_k', type=int, default=0, help='Number of Parity chunks in cache tier')
    parser.add_argument('--cached_l', type=int, default=0, help='Number of Remaining chunks in cache tier')
    parser.add_argument('--cached_network_m', type=int, default=0, help='Number of Data chunks in network cache tier')
    parser.add_argument('--cached_network_k', type=int, default=0, help='Number of Parity chunks in network cache tier')
    parser.add_argument('--cached_network_l', type=int, default=0, help='Number of Remaining chunks in network cache tier')
    parser.add_argument('--inter_replicas', type=int, default=0, help='Number of network copys')
    parser.add_argument('--intra_replicas', type=int, default=0, help='Number of local copys')
    parser.add_argument('--cached_write_ratio', type=float, default=0, help='Cached write ratio relative to total write')
    parser.add_argument('--cached_read_ratio', type=float, default=0.8, help='Cached read ratio relative to total write')
    parser.add_argument('--write_through', action='store_true', help='Flag to indicate if write through is used')
    parser.add_argument('--total_network_nodes', type=int, default=6, help='Total number of network nodes')
    parser.add_argument('--network_m', type=int, default=6, help='Number of Data chunks in network')
    parser.add_argument('--network_k', type=int, default=0, help='Number of Parity chunks in network')
    parser.add_argument('--network_l', type=int, default=0, help='Number of Remaining chunks in network')
    parser.add_argument('--capacity', type=int, default=64_000_000_000_000, help='Capacity of SSDs')
    parser.add_argument('--qlc', action='store_true', help='Flag to indicate if QLC SSDs are used. default is TLC')
    parser.add_argument('--simulation', action='store_true', help='Flag to indicate if simulation is being run')
    parser.add_argument('--dwpd', type=float, default=1, help='DWPD (Drive writes per day) of SSDs. Writes amount for cached tier if cached tier is used')
    parser.add_argument('--tbwpd', type=float, default=4.5, help='TB writes per day of SSDs. Writes amount for cached tier if cached tier is used')
    parser.add_argument('--use_tbwpd', action='store_true', help='Flag to indicate if TB writes per day is used instead of DWPD')
    parser.add_argument('--guarnanteed_years', type=int, default=5, help='Guaranteed years of SSDs')
    parser.add_argument('--config_file', type=str, default='2tier.json', help='Graph structure file path')
    parser.add_argument('--output_file', type=str, default='results.txt', help='Output file path to save results')
    parser.add_argument('--qlc_cache', action='store_true', help='Flag to indicate if QLC SSDs are used in cache tier. default is TLC')
    args = parser.parse_args()
    return args

args = parse_arguments()

total_ssds = args.total_ssds
m = args.m
k = args.k
l = args.l

cached_ssds = args.cached_ssds
cached_l = args.cached_l
cached_m = args.cached_m
cached_k = args.cached_k
cached_network_l = args.cached_network_l
cached_network_m = args.cached_network_m
cached_network_k = args.cached_network_k
if (cached_network_m == 0):
    cached_network_m = args.network_m
inter_replicas = args.inter_replicas
intra_replicas = args.intra_replicas

cached_write_ratio = args.cached_write_ratio
cached_read_ratio = args.cached_read_ratio
capacity = args.capacity
qlc = args.qlc
simulation = args.simulation
dwpd = args.dwpd
output_file = args.output_file
tbwpd = args.tbwpd
use_tbwpd = args.use_tbwpd
simulation = args.simulation
total_network_nodes = args.total_network_nodes

edges, enclosures, mttfs, mtrs, costs, options = parse_input_from_json(args.config_file)
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

n = m + k + l

if (n > total_ssds):
    raise ValueError('The sum of m, kvshould not exceed total_ssds')
if ((total_ssds - cached_ssds) % (n) != 0):
    raise ValueError('total_ssds should be divisible by the sum of m, k, l')

if (args.write_through):
    if (cached_ssds == 0):
        raise ValueError('Write through should be used with cached tier')
    if (cached_write_ratio != 0):
        raise ValueError('Do not use cached_write_ratio with write through')

if (cached_ssds == 0 and cached_write_ratio != 0):
    cached_write_ratio = 0
    cached_read_ratio = 0

if (cached_ssds == 0 and cached_m + cached_k + cached_l > 0):
    raise ValueError('Do not use cached_m, cached_k, cached_l without cached_ssds')

if (cached_ssds > 0):
    if (intra_replicas > 1):
        cached_m = 1
        cached_k = intra_replicas - 1
        cached_l = 0
        print ("input cached_m and cached_k are ignored, and calculated as 1 and intra_replicas - 1")
    if (inter_replicas > 1):
        cached_network_m = 1
        cached_network_k = inter_replicas - 1
        cached_network_l = 0
        print ("input cached_network_m and cached_network_k are ignored, and calculated as 1 and inter_replicas - 1")
    if ((cached_write_ratio == 0 or cached_write_ratio >= 1) and not args.write_through):
        raise ValueError('cached_write_ratio should be between 0 and 1')
    if (cached_m + cached_k + cached_l > cached_ssds):
        raise ValueError('The sum of cached_m, cached_k should not exceed cached_ssds')
    if (cached_ssds % (cached_m + cached_k) != 0):
        raise ValueError('cached_ssds should be divisible by the sum of cached_m, cached_k')
    if (intra_replicas == 1 or inter_replicas == 1):
        raise ValueError('replicas should be more than 1')
    

network_l = args.network_l
network_m = args.network_m
network_k = args.network_k
network_n = network_m + network_k

params_and_results = {}
params_and_results['total_ssds'] = total_ssds
params_and_results['m'] = m
params_and_results['k'] = k
params_and_results['l'] = args.l
params_and_results['cached_ssds'] = cached_ssds
params_and_results['cached_m'] = cached_m
params_and_results['cached_k'] = cached_k
params_and_results['cached_l'] = cached_l
params_and_results['cached_network_m'] = cached_network_m
params_and_results['cached_network_k'] = cached_network_k
params_and_results['cached_network_l'] = cached_network_l
params_and_results['inter_replicas'] = inter_replicas
params_and_results['intra_replicas'] = intra_replicas
params_and_results['cached_write_ratio'] = cached_write_ratio
params_and_results['network_m'] = network_m
params_and_results['network_k'] = network_k
params_and_results['network_l'] = network_l
params_and_results['capacity'] = capacity
params_and_results['qlc'] = qlc
params_and_results['simulation'] = simulation
params_and_results['dwpd'] = dwpd
params_and_results['guaranteed_years'] = guaranteed_years
params_and_results['dwpd_limit'] = dwpd_limit

params_and_results['use_tbwpd'] = use_tbwpd
params_and_results['tbwpd'] = tbwpd
params_and_results['simulation'] = simulation
params_and_results['total_network_nodes'] = total_network_nodes
params_and_results['ssd_read_bw'] = read_bw
params_and_results['ssd_write_bw'] = write_bw
if (args.qlc_cache == True):
    params_and_results['qlc_cache'] = True
    params_and_results['cached_dwpd_limit'] = qlc_dwpd
    params_and_results['cached_ssd_read_bw'] = qlc_read_bw
    params_and_results['cached_ssd_write_bw'] = qlc_write_bw
    params_and_results['cached_ssd_read_latency'] = options['qlc_read_latency']
else:
    params_and_results['qlc_cache'] = False
    params_and_results['cached_dwpd_limit'] = tlc_dwpd
    params_and_results['cached_ssd_read_bw'] = tlc_read_bw
    params_and_results['cached_ssd_write_bw'] = tlc_write_bw
    params_and_results['cached_ssd_read_latency'] = options['tlc_read_latency']

params_and_results['cached_read_ratio'] = cached_read_ratio
params_and_results['write_through'] = args.write_through
params_and_results['config_file'] = args.config_file

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
    if (simulation):
        num_simulations = 100000
        sim.monte_carlo_simulation(params_and_results, hardware_graph, num_simulations, options, costs)
        print (edges, enclosures, mttfs, mtrs)
    
        #test_static_analyze_ssd_only(guaranteed_years, use_tbwpd, tbwpd, dwpd_limit, capacity, dwpd, params_and_results, m, k, n, df, write_bw)
    output_params_and_results()