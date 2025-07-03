#!/bin/bash
script_dir="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"

output_file="result_tiering_dual_single_$(date '+%Y%m%d_%H%M%S').csv"
data=48 # m + k + l
parities=(1 2 3 4 5 6) # --k
l=4 # --l
capacity=(128_000_000_000_000)
dwpd=0.36
total_ssds=48
tier_files=("2tier.json" "2tier-singleport.json" "3tier.json" "3tier-singleport.json")
io_module_mttrs=(4 4 4 4)
python="pypy3"


#tier_files=("2tier.json" "3tier.json")
i=0
for t in "${tier_files[@]}"; do
    io_module_mttr=${io_module_mttrs[$i]}
    i=$(($i + 1))
    single_port=""
    if [[ "$t" == *"singleport"* ]]; then
        echo "Running for single port tier: $t"
        single_port="--single_port_ssd"
    else
        echo "Running for dual port tier: $t"
    fi
    for p in "${parities[@]}"; do
        m=$(($data - $p - $l))
        $python $script_dir/new_core.py --output_file $output_file --m $m --k $p --l $l --capacity $capacity --config_file $script_dir/$t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc $single_port --io_module_mttr $io_module_mttr
    done
done

