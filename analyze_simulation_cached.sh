#!/bin/bash
output_file="analysis_ssds_$(date '+%Y%m%d_%H%M%S').txt"
#stripe_size=(8 16 24 48) # m+k
stripe_div=(1 2) # m+k
parities=(0 1 2 3) # --k
parities_reserved=(0 2)

total_ssds=32
cache_ssds=(0)
#cache_ssds=(0 8)
cache_parities=(0 4)
network_datas=(4 8 12)
network_parities=(0 2 3)
tier_files=("2tier.json" "3tier.json")

dwpds=(0.1 1)
tbwpds=(4.5)
write_ratio=0.8
capacity=(64_000_000_000_000)
type=("qlc")
#type=("qlc")
additional_params="--simulation"
# iterate analyze_ssd_only.py over all possible combinations of parameters
echo "" > $output_file

for caches in "${cache_ssds[@]}"; do
    uncached_ssds=$(($total_ssds - $caches))
    run_once=false
    for s_div in "${stripe_div[@]}"; do
        s=$(($uncached_ssds / $s_div))
        for p in "${parities[@]}"; do
            for l in "${parities_reserved[@]}"; do
                m=$(($s - $p - $l))
                if [ "$m" -lt 1 ]; then
                    continue
                fi
                # if p == 0, we set m to 1
                if [ "$p" = 0 ]; then
                    if [ "$run_once" = true ]; then
                        continue
                    fi
                    m=1
                    run_once=true
                fi
                if [[ "$m" -eq 1 && "$l" -ne 0 ]]; then
                    continue
                fi
                for tier_file in "${tier_files[@]}"; do
                    for t in "${type[@]}"; do
                        current_type=""
                        if [ "$t" = "qlc" ]; then
                            current_type="--qlc"
                        fi
                        for d in "${dwpds[@]}"; do
                            for tbw in "${tbwpds[@]}"; do
                                for c in "${capacity[@]}"; do
                                    for cache_p in "${cache_parities[@]}"; do
                                        for network_data in "${network_datas[@]}"; do
                                            for network_parity in "${network_parities[@]}"; do # it shall be inner loop
                                                #additional_params="--network_m $network_data --network_k $network_parity --cached_network_m $network_data --cached_network_k $network_parity --simulation"
                                                additional_params="--network_m $network_data --network_k $network_parity --simulation --config_file $tier_file"
                                                cache_m=$(($caches - $cache_p))
                                                if [[ "$cache_m" -lt 0 ]]; then
                                                    continue
                                                fi
                                                if [[ "$cache_m" -gt 0 && "$cache_p" -eq 0 ]]; then
                                                    continue
                                                fi
                                                if [ "$use_tbwpd" = true ]; then
                                                    use_tbwpd_flag="--use_tbwpd"
                                                fi
                                                # change color of echo
                                                echo -e "\e[1;32m"
                                                echo "Running simulation python3 new_core.py $current_type --output_file $output_file --m $m --k $p --l $l --capacity $c $current_type --cached_ssds $caches --cached_m $cache_m --cached_k $cache_p --dwpd $d --cached_write_ratio $write_ratio $additional_params"
                                                echo -e "\e[0m"
                                                timeout 10m python3 new_core.py $current_type --total_ssds $total_ssds --output_file $output_file --m $m --k $p --l $l --capacity $c $current_type --cached_ssds $caches --cached_m $cache_m --cached_k $cache_p --dwpd $d --cached_write_ratio $write_ratio $additional_params
                                                # write through option
                                                if [[ $cached_m -gt 0 ]]; then
                                                    additional_params="--write_through --simulation"
                                                    timeout 10m python3 new_core.py $current_type  --total_ssds $total_ssds --output_file $output_file --m $m --k $p --l $l --capacity $c $current_type --cached_ssds $caches --cached_m $cache_m --cached_k $cache_p --dwpd $d --write_through $additional_params                                
                                                fi
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
