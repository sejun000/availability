#!/bin/bash

output_file="results_all_$(date '+%Y%m%d_%H%M%S').txt"
#excel_files=("2tier.xlsx" "3tier.xlsx")
excel_files=("3tier.xlsx")
network_only_values=(False True)
network_M=4
labels=("local_active + network_active" "network_only")
#network_parities=(0 1 2 3)
network_parities=(3)
# 파일 초기화
echo "" > $output_file

# 시뮬레이션 실행
for excel_file in "${excel_files[@]}"; do
    for i in "${!network_only_values[@]}"; do
        network_only=${network_only_values[$i]}
        label=${labels[$i]}
        for network_K in ${network_parities[@]}; do
 	    if [ "$network_only" = "True" ]; then
                echo "Running simulation with file: $excel_file, rebuild, network M : $network_M, network K : $network_K, network_only: $network_only"
       	        python3 monte_carlo_simulation.py --file_path "$excel_file" --network_M "$network_M" --network_K "$network_K" --network_only -o $output_file
            else
                echo "Running simulation with file: $excel_file, rebuild, network M : $network_M, network K : $network_K, network_only: $network_only"
       	        python3 monte_carlo_simulation.py --file_path "$excel_file" --network_M "$network_M" --network_K "$network_K" -o $output_file
            fi
	done
    done
done
