import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

# 결과 파일 파싱
def parse_results(file_path):
    data = {
        'SSD_capacity': [],
        'availability': [],
        'effective_availability': []
    }
    pattern = re.compile(r"Total Up Time: .*? \| Availability: ([0-9.]+) \| Effective Availability: ([0-9.]+) \| .* SSD Capacity: ([0-9_]+)")
    
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                availability = float(match.group(1))
                effective_availability = float(match.group(2))
                ssd_capacity = int(match.group(3).replace('_', ''))
                data['SSD_capacity'].append(ssd_capacity)
                print (availability)
                print (effective_availability)
                data['availability'].append(-np.log10(1 - availability))
                data['effective_availability'].append(-np.log10(1 - effective_availability))
    
    return pd.DataFrame(data)

# 결과 플로팅
def plot_results(df):
    plt.figure(figsize=(12, 6))

    plt.plot(df['SSD_capacity'], df['availability'], label='Availability (nines)', marker='o')
    plt.plot(df['SSD_capacity'], df['effective_availability'], label='Effective Availability (nines)', marker='o')

    plt.xscale('log')
    plt.xlabel('SSD Capacity (Bytes)')
    plt.ylabel('Nines')
    plt.title('Availability vs SSD Capacity')
    plt.legend()
    plt.grid(True)
    plt.xticks(df['SSD_capacity'], [f'{x // 1_000_000_000_000}T' for x in df['SSD_capacity']])
    plt.ylim(0, None)  # y축 범위를 0부터 시작
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse and plot simulation results.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the results file')

    args = parser.parse_args()
    df = parse_results(args.file_path)
    plot_results(df)
