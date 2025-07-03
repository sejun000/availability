#!/bin/bash
script_dir="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"

output_file="result_capacity_$(date '+%Y%m%d_%H%M%S').csv"
data=48 # m + k + l
parities=(1 2 3 4 5) # --k
l=4 # --l
capacities=(16_000_000_000_000 32_000_000_000_000 64_000_000_000_000 128_000_000_000_000 256_000_000_000_000)
dwpd=0.1
total_ssds=48
tier_files=("2tier.json")
python="pypy3"
#tier_files=("2tier.json" "3tier.json")
for t in "${tier_files[@]}"; do
    for p in "${parities[@]}"; do
        for c in "${capacities[@]}"; do
            m=$(($data - $p - $l))
            $python $script_dir/new_core.py --output_file $output_file --m $m --k $p --l $l --capacity $c --config_file $script_dir/$t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc
        done
    done
done
