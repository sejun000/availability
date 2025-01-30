import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# 결과 파일 파싱
def parse_results(file_path):
    data = {
        'Tier': [],
        'Configuration': [],
        'Availability': [],
        'Effective_Availability': []
    }
    pattern = re.compile(r"(\S+) \| (.*?) \| Total Up Time: .*? \| Availability: ([0-9.]+) \| Effective Availability: ([0-9.]+) \|")
    
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                tier = match.group(1)
                configuration = match.group(2)
                availability = float(match.group(3))
                effective_availability = float(match.group(4))
                
                # 가용성 값이 1에 매우 가까운 경우 작은 양을 추가
                if availability >= 1.0:
                    availability = 1.0 - 1e-10
                if effective_availability >= 1.0:
                    effective_availability = 1.0 - 1e-10
                
                data['Tier'].append(tier)
                data['Configuration'].append(configuration)
                data['Availability'].append(-np.log10(1 - availability))
                data['Effective_Availability'].append(-np.log10(1 - effective_availability))
    
    return pd.DataFrame(data)

# 결과 플로팅
def plot_results(df):
    fig, ax = plt.subplots(figsize=(14, 7))

    tiers = df['Tier'].unique()
    configurations = df['Configuration'].unique()
    
    x = np.arange(len(tiers))  # the label locations
    width = 0.2  # the width of the bars

    # 데이터 그룹화 및 평균 계산
    data = {}
    for config in configurations:
        data[config] = {
            'Availability': [],
            'Effective_Availability': []
        }
        for tier in tiers:
            config_data = df[(df['Tier'] == tier) & (df['Configuration'] == config)]
            data[config]['Availability'].append(config_data['Availability'].mean())
            data[config]['Effective_Availability'].append(config_data['Effective_Availability'].mean())
    
    # 막대그래프 그리기
    for i, config in enumerate(configurations):
        ax.bar(x + i*width - width, data[config]['Availability'], width, label=f'{config} Availability (nines)')
        ax.bar(x + i*width, data[config]['Effective_Availability'], width, label=f'{config} Effective Availability (nines)')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Tier')
    ax.set_ylabel('Nines')
    ax.set_title('Availability and Effective Availability by Tier and Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, rotation=45, ha="right")
    ax.legend()

    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse and plot simulation results.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the results file')

    args = parser.parse_args()
    df = parse_results(args.file_path)
    plot_results(df)
