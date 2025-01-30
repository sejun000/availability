#!/bin/bash

output_file="results_all_$(date '+%Y%m%d_%H%M%S').txt"
excel_files=("2tier.xlsx" "3tier.xlsx")
rebuild_redundancy_values=(False True True)
network_only_values=(False True False)
labels=("local_active + network_backup" "network_only" "local_active + network_active")

# 파일 초기화
echo "" > $output_file

# 시뮬레이션 실행
for excel_file in "${excel_files[@]}"; do
    for i in "${!rebuild_redundancy_values[@]}"; do
        rebuild_redundancy=${rebuild_redundancy_values[$i]}
        network_only=${network_only_values[$i]}
        label=${labels[$i]}
        echo "Running simulation with file: $excel_file, rebuild_redundancy: $rebuild_redundancy, network_only: $network_only"
        
        if [ "$rebuild_redundancy" = "True" ] && [ "$network_only" = "True" ]; then
            python3 monte_carlo_simulation.py --file_path "$excel_file" --network_redundancy 5 --network_only -o $output_file
        elif [ "$rebuild_redundancy" = "True" ]; then
            python3 monte_carlo_simulation.py --file_path "$excel_file" --network_redundancy 5 -o $output_file
        else
            python3 monte_carlo_simulation.py --file_path "$excel_file" -o $output_file
        fi
    done
done
