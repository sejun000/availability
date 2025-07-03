#!/bin/bash
script_dir="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
output_file="result_cache_$(date '+%Y%m%d_%H%M%S').csv"
parities=(4) # --k
rgroups=(1)
cached_ssds=(0 2 4 6 8 10 12)
python="pypy3"
cache_hit_ratios=(0 0.47 0.503 0.518 0.527 0.533 0.542)
capacity=128_000_000_000_000
dwpds=(0.02 0.066 0.2 0.66 2)
total_ssds=48

tier_files=("2tier.json")

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
                        $python $script_dir/new_core.py --output_file $output_file --m $m --k $p --intra_replicas 2 --cached_ssds $cs --cached_write_ratio $cached_hit_ratio --capacity $c --config_file $script_dir/$t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc
                    done
                done
            done
        done
    done
done
