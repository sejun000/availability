#!/usr/bin/env python3
import argparse
import datetime
from datetime import timedelta
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def parse_timestamp(header_line):
    """
    헤더 라인 예:
    "========== Sat 08 Mar 2025 08:30:23 PM KST =========="
    앞뒤의 '='를 제거하고, 마지막 시간대 토큰(KST)은 무시한 후,
    "Sat 08 Mar 2025 08:30:23 PM" 형식으로 파싱.
    """
    ts_str = header_line.strip("= \n")
    parts = ts_str.split()
    if len(parts) >= 7:
        ts_str = " ".join(parts[:-1])  # 마지막 토큰 제거
    return datetime.datetime.strptime(ts_str, "%a %d %b %Y %I:%M:%S %p")

def load_trace(filename):
    """
    파일을 읽어 (timestamp, nand_write_count, write_count) 튜플의 리스트를 반환합니다.
    각 블록은 3줄(헤더, nand_write_count, write_count)로 구성되어 있다고 가정합니다.
    """
    blocks = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("=========="):
            try:
                timestamp = parse_timestamp(line)
                nand_line = lines[i+1].strip()
                write_line = lines[i+2].strip()
                nand_val = int(nand_line.split(':')[1].strip())
                write_val = int(write_line.split(':')[1].strip())
                blocks.append((timestamp, nand_val, write_val))
                i += 3
            except Exception as e:
                print("Error parsing block at line", i, e)
                i += 1
        else:
            i += 1
    return blocks

def compute_waf(blocks):
    """
    인접한 블록 쌍에 대해 WAF 값을 계산합니다.
    WAF = (nand_write_count(t+1) - nand_write_count(t)) / (write_count(t+1) - write_count(t))
    각 구간의 (시작시간, 끝시간, WAF)를 튜플로 반환합니다.
    """
    intervals = []
    for i in range(len(blocks) - 1):
        t1, nand1, write1 = blocks[i]
        t2, nand2, write2 = blocks[i+1]
        delta_nand = nand2 - nand1
        delta_write = write2 - write1
        if delta_write == 0:
            waf = float('inf')
        else:
            waf = delta_nand / delta_write
        intervals.append((t1, t2, waf))
    return intervals

def plot_waf(intervals):
    """
    시간에 따른 WAF 그래프를 그립니다.
    x축: 구간의 중간 시각, y축: WAF 값.
    """
    x = []
    y = []
    for t1, t2, waf in intervals:
        midpoint = t1 + (t2 - t1)/2
        x.append(midpoint)
        y.append(waf)
    plt.figure(figsize=(10,5))
    plt.plot(x, y, marker='o')
    plt.xlabel("Time")
    plt.ylabel("WAF")
    plt.title("WAF over time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Calculate WAF from trace file. "
                    "Use --date (YYYY/MM/DD) and --time (HH:MM) to filter data for a specific minute. "
                    "If not provided, the entire trace is plotted."
    )
    parser.add_argument("--date", help="Date in format YYYY/MM/DD")
    parser.add_argument("--time", help="Time in format HH:MM")
    parser.add_argument("--file", required=True, help="Path to the trace file")
    args = parser.parse_args()

    blocks = load_trace(args.file)
    if not blocks:
        print("No trace blocks found in the file.")
        return

    # 입력이 모두 제공된 경우: 지정된 날짜와 시간(분 단위) 내의 데이터만 필터링
    if args.date and args.time:
        try:
            filter_start = datetime.datetime.strptime(f"{args.date} {args.time}", "%Y/%m/%d %H:%M")
        except Exception as e:
            print("Invalid date/time format:", e)
            return
        # filter: filter_start <= timestamp < filter_start + 1분
        filter_end = filter_start + timedelta(minutes=1)
        filtered_blocks = [b for b in blocks if filter_start <= b[0] < filter_end]
        if len(filtered_blocks) < 2:
            print("Not enough data points in the given time window.")
            return
        intervals = compute_waf(filtered_blocks)
        print("Calculated WAF values for the given time window:")
        for t1, t2, waf in intervals:
            print(f"{t1.strftime('%Y-%m-%d %H:%M:%S')} -> {t2.strftime('%Y-%m-%d %H:%M:%S')}: WAF = {waf:.4f}")
    else:
        # 입력이 없으면 전체 trace의 데이터를 사용하여 그래프를 그림
        intervals = compute_waf(blocks)
        if not intervals:
            print("Not enough data points to compute WAF.")
            return
        print("Plotting WAF over the entire trace...")
        plot_waf(intervals)

if __name__ == "__main__":
    main()
