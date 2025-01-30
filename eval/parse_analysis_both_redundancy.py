import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='SSD only analysis\' parameters.')
    parser.add_argument('--input_file', type=str, default='results.txt', help='Input file path to read results')
    args = parser.parse_args()
    return args

def parse_file(input_file):
    print (f'Parsing file: {input_file}')
    records = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip():
                continue
            
            tokens = [token.strip() for token in line.strip('| ').split('|')]
            record = {}
            i = 0
            while i < len(tokens):
                print (f'Processing token: {tokens[i]}')
                if not tokens[i]:
                    i += 1
                    continue
                key = tokens[i]
                value = tokens[i + 1] if i + 1 < len(tokens) else None
                try:
                    if '.' in value:
                        value = float(value)
                        if("_cost" in key):
                            record[key + "_log_scale"] = math.log10(value)
                    else:
                        if (key == 'network_k'):
                            if (int(value) == 0):
                                record['rebuild_type'] = 'intra_only'
                            elif not 'rebuild_type' in record:
                                record['rebuild_type'] = 'both'
                        elif (key == 'k'):
                            if (int(value) == 0):
                                record['rebuild_type'] = 'inter_only'
                            elif not 'rebuild_type' in record:
                                record['rebuild_type'] = 'both'
                        value = int(value)
                except ValueError:
                    if value == 'True':
                        value = True
                    elif value == 'False':
                        value = False
                record[key] = value
                i += 2
            records.append(record)

        df = pd.DataFrame(records)
    return df


args = parse_arguments()
input_file = args.input_file
df = parse_file(input_file)

filtered_df = df[
    (df['config_file'] == '2tier.json')
]

filtered_df['x'] = filtered_df['m'] / (filtered_df['m'] + filtered_df['k']) * filtered_df['network_m'] / (filtered_df['network_m'] + filtered_df['network_k']) 
filtered_df['total_cost_per_effective_gb_log_scale'] = np.log(filtered_df['total_cost_for_10_years'] / (filtered_df['x'] * 64_000 * 48))
# dwpd 값에 따른 색상 매핑
"""
color_map = {
    0.1: 'blue',
    0.5: 'green',
    1.0: 'orange',
    2.0: 'red'
}
"""

"""
color_map = {
    'both': 'blue',
    'intra_only': 'green',
    'inter_only': 'orange',
}
"""
color_vector = [
    'blue',
    'green',
    'orange',
    'red'
]




# 'm + k'를 그룹화 기준으로 추가 열 생성
filtered_df['n'] = filtered_df['m'] + filtered_df['k']
filtered_df['network_n'] = filtered_df['network_m'] + filtered_df['network_k']
# if network 'k' is 0, then it is intra_only
# if 'k' is 0, then it is inter_only
# if both 'k' and 'network_k' are non-zero, then it is both


#y_axis_value = 'avail_nines'
#y_axis_value = 'avail_nines'
#y_axis_value = 'eff_avail_nines'
#y_axis_value = 'total_cost_for_10_years'
#y_axis_value = 'down_cost_for_10_years'
#y_axis_value = 'repair_cost_for_10_years'
#y_axis_value = 'initial_cost'
# y_axis_value = 'operation_cost_for_10_years'
y_axis_value = 'avg_time_for_rebuilding'

print ("intra_only")

intra_df = filtered_df[
    (filtered_df['rebuild_type'] == 'intra_only')
]
output_df = intra_df[['x', y_axis_value, 'm', 'k', 'network_m', 'network_k']]

print("x       ", y_axis_value)
for _, row in output_df.iterrows():
    print(f"{row['x']:.3f}     {row[y_axis_value]:.2f}   {row['m']}   {row['k']}  {row['network_m']}   {row['network_k']}")


print ("inter_only")
inter_df = filtered_df[
    (filtered_df['rebuild_type'] == 'inter_only')
]

output_df = inter_df[['x', y_axis_value, 'm', 'k', 'network_m', 'network_k']]
print("x       ", y_axis_value)
for _, row in output_df.iterrows():
    print(f"{row['x']:.3f}     {row[y_axis_value]:.2f}   {row['m']}   {row['k']}  {row['network_m']}   {row['network_k']}")

print ("both")
both_df = filtered_df[
    (filtered_df['rebuild_type'] == 'both')
]

output_df = both_df[['x', y_axis_value, 'm', 'k', 'network_m', 'network_k']]
print("x       ", y_axis_value)
for _, row in output_df.iterrows():
    print(f"{row['x']:.3f}     {row[y_axis_value]:.2f}   {row['m']}   {row['k']}  {row['network_m']}   {row['network_k']}")