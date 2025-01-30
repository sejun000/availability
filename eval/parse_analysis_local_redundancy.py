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
                # check if the token is empty
                if not tokens[i]:
                    # print ('Empty token')
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

filtered_df = df[
    (df['config_file'] == '2tier.json')
]

filtered_df['x'] = filtered_df['m'] / (filtered_df['m'] + filtered_df['k']) * filtered_df['network_m'] / (filtered_df['network_m'] + filtered_df['network_k']) * (filtered_df['total_ssds'] - filtered_df['cached_ssds']) / (filtered_df['total_ssds'])
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

# 데이터프레임을 피벗 형태로 변환
#table = filtered_df.pivot(index='k', columns='n', values='avail_nines')
#table = filtered_df.pivot(index='k', columns='n', values='avg_time_for_rebuilding')
#table = filtered_df.pivot(index='k', columns='n', values='eff_avail_nines')
#table = filtered_df.pivot(index='k', columns='n', values='total_cost_for_10_years')
#table = filtered_df.pivot(index='k', columns='n', values='down_cost_for_10_years')
#table = filtered_df.pivot(index='k', columns='n', values='repair_cost_for_10_years')
table = filtered_df.pivot(index='k', columns='n', values='initial_cost')
#table = filtered_df.pivot(index='k', columns='n', values='operation_cost_for_10_years')

# 표 출력
print("k/n", " ".join(map(str, table.columns)))
for k, row in table.iterrows():
    print(k, " ".join(f"{value:.9f}" if not pd.isna(value) else "" for value in row))
