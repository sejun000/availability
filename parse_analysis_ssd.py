import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import math

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
            # check if the line is empty
            if not line.strip():
                # print ('Empty line')
                continue
             # '|' 기호로 분리하고 양쪽 공백 제거
            tokens = [token.strip() for token in line.strip('| ').split('|')]
            # 키-값 쌍을 딕셔너리로 변환
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
                        if (key == 'availability'):
                            if (value == 1.0):
                                record['avail_nines'] = 10
                            else:
                                avail_nine = -math.log10(1 - value)
                                record['avail_nines'] = avail_nine
                        elif (key == 'effective_availability'):
                            if (value == 1.0):
                                record['eff_avail_nines'] = 10
                            else:
                                eff_avail_nine = -math.log10(1 - value)
                                record['eff_avail_nines'] = eff_avail_nine
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

#filtered_df = df[(df['capacity'] == 64_000_000_000_000) & (df['qlc'] == False)]
#filtered_df = df[(df['dwpd'] == 1.0) & (df['qlc'] == False) & ((df['network_m'] == 6) and (df['network_k'] == 0))]
filtered_df = df[
    (df['dwpd'] == 1.0) &
    (df['qlc'] == False) &
    ((df['network_m'] == 6) & (df['network_k'] == 0) | (df['network_k'] > 0))
]

#filtered_df = df[(df['dwpd'] == 1.0) & (df['qlc'] == False) & (df['capacity'] == 64_000_000_000_000)]
filtered_df['x'] = filtered_df['m'] / (filtered_df['m'] + filtered_df['k']) * filtered_df['network_m'] / (filtered_df['network_m'] + filtered_df['network_k'])

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
    8_000_000_000_000: 'blue',
    16_000_000_000_000: 'green',
    32_000_000_000_000: 'orange',
    64_000_000_000_000: 'red'
}
"""
color_map = {
    'both': 'blue',
    'intra_only': 'green',
    'inter_only': 'orange',
}



# 색상 리스트 생성
## colors = filtered_df['dwpd'].map(color_map).fillna('gray')  # 정의되지 않은 dwpd 값은 회색으로 표시
colors = filtered_df['rebuild_type'].map(color_map).fillna('gray')  # 정의되지 않은 capacity 값은 회색으로 표시

# 그래프 크기 설정
plt.figure(figsize=(8, 6))

# 분산 그래프 그리기

for dwpd_value in filtered_df['rebuild_type'].unique():
    # 각 dwpd 값에 해당하는 데이터 선택
    df_subset = filtered_df[filtered_df['rebuild_type'] == dwpd_value]
    plt.scatter(df_subset['x'], df_subset['avail_nines'], color=color_map.get(dwpd_value, 'gray'), label=f'rebuild_type={dwpd_value}')
    for idx, row in df_subset.iterrows():
       plt.annotate(f"({row['m']},{row['k']},{row['network_m']},{row['network_k']})", (row['x'], row['avail_nines']), textcoords="offset points", xytext=(0,10), ha='center')

# 축 레이블 및 제목 설정
plt.xlabel('Capcity Ratio')
plt.ylabel('Nines')
plt.title('Nines vs capacity ratio')

# 범례 추가
plt.legend(title='Host Writes')

# 그리드 추가
plt.grid(True)

print (filtered_df)
# 그래프 표시
plt.show()


"""
filtered_df = df[(df['capacity'] == 64_000_000_000_000) & (df['qlc'] == False) & df['dwpd'] == 1.0]
filtered_df['x'] = filtered_df['m'] + filtered_df['k']
#filtered_df = df[(df['dwpd'] == 1.0) & (df['qlc'] == False)]

# 그래프 크기 설정
plt.figure(figsize=(10, 6))

# 각 k 값에 대해 그래프 그리기
for idx, k_value in enumerate(sorted(filtered_df['k'].unique())):
    subset = filtered_df[filtered_df['k'] == k_value]
    plt.plot(subset['x'], subset['rebuild_speed'] / 1_000_000,
             marker='o', label=f'k = {k_value}')

# 축 레이블 및 제목 설정
plt.xlabel('n')
plt.ylabel('Rebuild Speed (MegaBytes/s)')
plt.title('Rebuild Speed vs n for different k values')

# 그리드 추가
plt.grid(True)

# 범례 표시
plt.legend()
# 그래프 표시
plt.show()
"""