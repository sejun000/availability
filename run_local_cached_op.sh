output_file="analysis_local_redundancy_group_$(date '+%Y%m%d_%H%M%S').txt"
parities=(3) # --k
rgroups=(1)
#cached_ssds=(2 4 8 12)
cached_ssds=(8)
op_ratios=(0.498 0.243 0.07)
waf_ratios=(1.7 2.2 6.7)
#op_ratios=(0.47 0.31 0.18 0.07)
#cache_hit_ratios=(0.852 0.865 0.87 0.874 0.878)
#cache_hit_ratios=(0.6578 0.6695 0.6825) # cache hit
cache_hit_ratios=(0.597 0.612 0.629) # write size
#cache_hit_ratios=(0.4379 0.4427 0.4536)
#cache_hit_ratios=(0.8139 0.8398 0.878 0.8901)
capacity=64_000_000_000_000
#dwpds=(0.1 0.33 1 3)
dwpds=(0.1 0.33 1 3)
total_ssds=48
tier_files=("3tier.json")
#tier_files=("2tier.json" "3tier.json")
for t in "${tier_files[@]}"; do
    for cs in "${cached_ssds[@]}"; do
        op_index=0
        for op_ratio in "${op_ratios[@]}"; do
            cached_hit_ratio=${cache_hit_ratios[$op_index]}
            waf_ratio=${waf_ratios[$op_index]}
            #cached_hit_ratio=${cache_hit_ratios[0]}
            op_index=$(($op_index + 1))
            for rgroup in "${rgroups[@]}"; do
                for dwpd in "${dwpds[@]}"; do
                    for p in "${parities[@]}"; do
                        cold_ssds=$(($total_ssds - $cs))
                        m=$(($cold_ssds / $rgroup - $p))
                        if [ "$m" -lt 1 ]; then
                            continue
                        fi
                        for c in "${capacity[@]}"; do
                            echo -e "\e[1;32m"
                            echo "Running simulation with stripe size: $s, datas: $m, parities: $p, capacity: $c, tier_file: $t"
                            echo "Command Lines : python3 new_core.py --output_file $output_file --m $m --k $p --intra_replicas 2 --cached_ssds $cs --cached_write_ratio $cached_hit_ratio --capacity $c --config_file $t --simulation --total_ssds $total_ssds --dwpd $dwpd --waf_ratio $waf_ratio --op_ratio $op_ratio  --qlc"
                            echo -e "\e[0m"
                            python3 new_core.py --output_file $output_file --m $m --k $p --intra_replicas 2 --cached_ssds $cs --cached_write_ratio $cached_hit_ratio --capacity $c --config_file $t --simulation --total_ssds $total_ssds --dwpd $dwpd --waf_ratio $waf_ratio --op_ratio $op_ratio --qlc
                        done
                    done
                done
            done
        done
    done
done
