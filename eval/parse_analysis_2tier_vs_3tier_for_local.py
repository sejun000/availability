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
                # 값의 타입 변환 시도
                try:
                    # 숫자인 경우 변환
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
                    # 불리언 값 처리
                    if value == 'True':
                        value = True
                    elif value == 'False':
                        value = False
                    # 숫자가 아니면 그대로 문자열로 유지
                record[key] = value
                i += 2
            records.append(record)

        # DataFrame 생성
        df = pd.DataFrame(records)
    return df


args = parse_arguments()
input_file = args.input_file
df = parse_file(input_file)
df['n'] = df['m'] + df['k']
filtered_df = df[
    (df['n'] == 48)
]

filtered_df['x'] = filtered_df['m'] / (filtered_df['m'] + filtered_df['k']) * filtered_df['network_m'] / (filtered_df['network_m'] + filtered_df['network_k']) 
filtered_df['total_cost_per_effective_gb_log_scale'] = np.log(filtered_df['total_cost_for_10_years'] / (filtered_df['x'] * 64_000 * 48))

color_vector = [
    'blue',
    'green',
    'orange',
    'red'
]

filtered_df['network_n'] = filtered_df['network_m'] + filtered_df['network_k']
# if network 'k' is 0, then it is intra_only
# if 'k' is 0, then it is inter_only
# if both 'k' and 'network_k' are non-zero, then it is both


#y_axis_value = 'avail_nines'
#y_axis_value = 'eff_avail_nines'
#y_axis_value = 'total_cost_for_10_years'
#y_axis_value = 'down_cost_for_10_years'
#y_axis_value = 'repair_cost_for_10_years'
#y_axis_value = 'initial_cost'
# y_axis_value = 'operation_cost_for_10_years'

# 데이터프레임을 피벗 형태로 변환
# 표 출력



#table = filtered_df.pivot(index='k', columns='config_file', values='avail_nines')
#table = filtered_df.pivot(index='k', columns='config_file', values='eff_avail_nines')
#table = filtered_df.pivot(index='k', columns='config_file', values='eff_avail_nines')
#table = filtered_df.pivot(index='k', columns='config_file', values='total_cost_for_10_years')
#table = filtered_df.pivot(index='k', columns='config_file', values='total_credit_ratio')
#table = filtered_df.pivot(index='k', columns='config_file', values='down_cost_for_10_years')
table = filtered_df.pivot(index='k', columns='config_file', values='cost_per_gb')
#table = filtered_df.pivot(index='k', columns='config_file', values='repair_cost_for_10_years')
#table = filtered_df.pivot(index='k', columns='config_file', values='initial_cost')
#table = filtered_df.pivot(index='k', columns='config_file', values='operation_cost_for_10_years')
#table = filtered_df.pivot(index='k', columns='config_file', values='avg_time_for_rebuilding')

# 표 출력
print("k/n", " ".join(map(str, table.columns)))
for k, row in table.iterrows():
    print(k, " ".join(f"{value:.9f}" if not pd.isna(value) else "" for value in row))