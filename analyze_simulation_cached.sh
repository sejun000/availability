#!/bin/bash
output_file="analysis_ssds_$(date '+%Y%m%d_%H%M%S').txt"
#stripe_size=(8 16 24 48) # m+k
stripe_div=(1 2 3) # m+k
parities=(1 2 3 4 5) # --k

total_ssds=48
cache_ssds=(6 12)
cache_parities=(2 3 4)
dwpds=(0.1 1)
tbwpds=(4.5)
write_ratio=0.8
capacity=(64_000_000_000_000)
type=("qlc")
#type=("qlc")
additional_params="--simulation"
run_once=false
# iterate analyze_ssd_only.py over all possible combinations of parameters
echo "" > $output_file

for caches in "${cache_ssds[@]}"; do
    uncached_ssds=$(($total_ssds - $caches))
    for s_div in "${stripe_div[@]}"; do
        s=$(($uncached_ssds / $s_div))
        for p in "${parities[@]}"; do
            m=$(($s - $p))
            # if p == 0, we set m to 1
            if [ "$p" = 0 ]; then
                if [ "$run_once" = true ]; then
                    continue
                fi
                m=1
                run_once=true
            fi
            for t in "${type[@]}"; do
                current_type=""
                if [ "$t" = "qlc" ]; then
                    current_type="--qlc"
                fi
                for d in "${dwpds[@]}"; do
                    for tbw in "${tbwpds[@]}"; do
                        for c in "${capacity[@]}"; do
                            for cache_p in "${cache_parities[@]}"; do
                                cache_m=$(($caches - $cache_p))
                                    
                                if [ "$use_tbwpd" = true ]; then
                                    use_tbwpd_flag="--use_tbwpd"
                                fi
                                # change color of echo
                                echo -e "\e[1;32m"
                                echo "Running simulation python3 new_core.py $current_type --output_file $output_file --m $m --k $p --capacity $c $current_type --cached_ssds $cache_ssds --cached_m $cache_m --cached_k $cache_p --dwpd $d --cached_write_ratio $write_ratio $additional_params"
                                echo -e "\e[0m"
                                timeout 10m python3 new_core.py $current_type --output_file $output_file --m $m --k $p --capacity $c $current_type --cached_ssds $cache_ssds --cached_m $cache_m --cached_k $cache_p --dwpd $d --cached_write_ratio $write_ratio $additional_params
                                # write through option
                                timeout 10m python3 new_core.py $current_type --output_file $output_file --m $m --k $p --capacity $c $current_type --cached_ssds $cache_ssds --cached_m $cache_m --cached_k $cache_p --dwpd $d --write_through $additional_params                                
                            done
                        done
                    done
                done
            done
        done
    done
done
