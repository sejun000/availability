#!/bin/bash

output_file="results_all_$(date '+%Y%m%d_%H%M%S').txt"
execute_file="core.py"
excel_files=("3tier.xlsx" "2tier.xlsx")
#excel_files=("3tier_both_fast.xlsx")
#excel_files=("3tier_both_fast.xlsx" "2tier_both_fast.xlsx")
additional_param=("--lowest_common_level_module backend_module" "--lowest_common_level_module switch")
#network_only_values=(False True)
network_only_values=(False)
total_nodes=(32 32 32 32 32 32 32 32 32)
network_Ms=(4 3 2 8 7 6 16 15 14)
#network_Ms=(32 31 30 32 31 30 32 31 30)
#network_ms=(4 3 2 8 7 6 16 15 14)
network_ms=(0 0 0 0 0 0 0 0 0)
network_Ks=(0 1 2 0 1 2 0 1 2)
SSD_Ms=(8 16 32 7 15 31 6 14 30)
SSD_Ks=(0 0 0 1 1 1 2 2 2)
SSD_write_percentage=0
#labels=("local_active + network_active" "network_only")
labels=("local_active + network_active")
#network_parities=(0 1 2 3)

#capacity=(1_000_000_000_000 4_000_000_000_000 16_000_000_000_000 64_000_000_000_000)
capacity=(64_000_000_000_000)
# 파일 초기화
echo "" > $output_file

# 시뮬레이션 실행
for ii in "${!additional_param[@]}"; do
    excel_file=${excel_files[$ii]}
    additional_param_value=${additional_param[$ii]}
    for i in "${!network_only_values[@]}"; do
        network_only=${network_only_values[$i]}
        label=${labels[$i]}
        for cap in ${capacity[@]}; do
            for j in "${!SSD_Ms[@]}"; do
                for k in ${!network_Ms[@]}; do
                    network_M=${network_Ms[$k]}
                    network_K=${network_Ks[$k]}
		    network_m=${network_ms[$k]}
                    total_node=${total_nodes[$k]}
                    SSD_M=${SSD_Ms[$j]}
                    SSD_K=${SSD_Ks[$j]}
                    if [ $k -eq 0 ]; then
                        echo "Running simulation with file: $excel_file, rebuild, network M : $network_M, network K : $network_K, network_only: $network_only, capacity: $cap, SSD_M: $SSD_M, SSD_K: $SSD_K, additional_param: $additional_param_value"
                        python3 ${execute_file} --file_path "$excel_file" --network_M "$network_M" --network_K "$network_K" --network_m "$network_m" --total_nodes $total_node -o $output_file --SSD_capacity $cap --SSD_M $SSD_M --SSD_K $SSD_K --SSD_write_percentage $SSD_write_percentage --generate_fault_injection $additional_param_value
                    fi
                    if [ "$network_only" = "True" ]; then
                        echo "Running simulation with file: $excel_file, rebuild, network M : $network_M, network K : $network_K, network_only: $network_only, capacity: $cap, SSD_M: $SSD_M, SSD_K: $SSD_K, additional_param: $additional_param_value"
                        python3 ${execute_file} --file_path "$excel_file" --network_M "$network_M" --network_K "$network_K" --network_m "$network_m" --total_nodes $total_node  --network_only -o $output_file --SSD_capacity $cap --SSD_M $SSD_M --SSD_K $SSD_K --SSD_write_percentage $SSD_write_percentage $additional_param_value
                    else
                        echo "Running simulation with file: $excel_file, rebuild, network M : $network_M, network K : $network_K, network_only: $network_only, capacity: $cap, SSD_M: $SSD_M, SSD_K: $SSD_K, additional_param: $additional_param_value"
                        python3 ${execute_file} --file_path "$excel_file" --network_M "$network_M" --network_K "$network_K" --network_m "$network_m" --total_nodes $total_node -o $output_file --SSD_capacity $cap --SSD_M $SSD_M --SSD_K $SSD_K --SSD_write_percentage $SSD_write_percentage $additional_param_value
                    fi
                done
            done
        done
    done
done
