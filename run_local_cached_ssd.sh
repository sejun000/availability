#!/bin/bash
output_file="analysis_cached_redundancy_group_$(date '+%Y%m%d_%H%M%S').txt"
parities=(3) # --k
#parities=(3)
rgroups=(1)
#cached_ssds=(4 8 12)
cached_ssds=(0 2 4 6 8 10 12)
python="pypy3"
#cached_ssds=(0)
#cache_hit_ratios=(0.4437 0.6661 0.7169)
#cache_hit_ratios=(0 0.56 0.47 0.38 0.33)
cache_hit_ratios=(0 0.4151 0.438 0.4609 0.5289 0.592 0.6147)
capacity=64_000_000_000_000
dwpds=(0.02 0.066 0.2 0.66 2)
total_ssds=48
#tier_files=("3tier.json")
tier_files=("2tier.json")
#tier_files=("2tier.json" "3tier.json")
for t in "${tier_files[@]}"; do
    cs_index=0
    for cs in "${cached_ssds[@]}"; do
        cached_hit_ratio=${cache_hit_ratios[$cs_index]}
        cs_index=$(($cs_index + 1))
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
                        echo "Command Lines : $python new_core.py --output_file $output_file --m $m --k $p --intra_replicas 2 --cached_ssds $cs --cached_write_ratio $cached_hit_ratio --capacity $c --config_file $t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc"
                        echo -e "\e[0m"
                        $python new_core.py --output_file $output_file --m $m --k $p --intra_replicas 2 --cached_ssds $cs --cached_write_ratio $cached_hit_ratio --capacity $c --config_file $t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc
                    done
                done
            done
        done
    done
done
