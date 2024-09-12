#!/bin/bash

output_file="results_$(date '+%Y%m%d_%H%M%S').txt"
#capacities=(1_000_000_000_000 2_000_000_000_000 4_000_000_000_000 8_000_000_000_000 16_000_000_000_000 32_000_000_000_000 64_000_000_000_000)
capacities=(125_000_000_000 250_000_000_000 500_000_000_000 1_000_000_000_000 2_000_000_000_000 4_000_000_000_000 8_000_000_000_000)
execute_file="core.py"

# 파일 초기화
echo "" > $output_file

additional_args_for_injection=" --total_nodes 32 --network_M 16 --network_K 0 --SSD_M 6 --SSD_K 2 --file_path SSD.xlsx --generate_fault_injection --start_node_module switch --leaf_node_module nand --local_level_module port,ncpu,nand"
additional_args_for_test=" --total_nodes 32 --network_M 14 --network_K 2 --SSD_M 6 --SSD_K 2 --file_path SSD.xlsx --start_node_module switch --leaf_node_module nand --local_level_module port,ncpu,nand"

# 시뮬레이션 실행
for capacity in "${capacities[@]}"
do
    #echo python3 ${execute_file} --SSD_capacity $capacity -o $output_file ${additional_args_for_injection}
    python3 ${execute_file} --SSD_capacity $capacity -o $output_file ${additional_args_for_injection}
    python3 ${execute_file} --SSD_capacity $capacity -o $output_file ${additional_args_for_test}
done
