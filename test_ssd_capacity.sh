#!/bin/bash

output_file="results_$(date '+%Y%m%d_%H%M%S').txt"
capacities=(1_000_000_000_000 2_000_000_000_000 4_000_000_000_000 8_000_000_000_000 16_000_000_000_000 32_000_000_000_000 64_000_000_000_000)

# 파일 초기화
echo "" > $output_file

# 시뮬레이션 실행
for capacity in "${capacities[@]}"
do
    python3 monte_carlo_simulation.py --SSD_capacity $capacity -o $output_file
done
python3 plot_capacity.py --file_path $output_file
