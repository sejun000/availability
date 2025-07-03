#!/bin/bash
script_dir="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"

output_file="result_dual_single_$(date '+%Y%m%d_%H%M%S').csv"
data=48 # m + k + l
parities=(1 2 3 4 5 6) # --k
l=0 # --l
capacity=(128_000_000_000_000)
dwpd=0.36
total_ssds=48
tier_files=("2tier.json" "2tier-singleport.json")
python="pypy3"


#tier_files=("2tier.json" "3tier.json")
for t in "${tier_files[@]}"; do
    single_port=""
    if [[ "$t" == *"singleport"* ]]; then
        echo "Running for single port tier: $t"
        single_port="--single_port_ssd"
    else
        echo "Running for dual port tier: $t"
    fi
    for p in "${parities[@]}"; do
        m=$(($data - $p - $l))
        $python $script_dir/new_core.py --output_file $output_file --m $m --k $p --l $l --capacity $capacity --config_file $script_dir/$t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc $single_port
    done
done

for t in "${tier_files[@]}"; do
    single_port=""
    if [[ "$t" == *"singleport"* ]]; then
        echo "Running for single port tier: $t"
        single_port="--single_port_ssd"
    else
        echo "Running for dual port tier: $t"
    fi
    m=$(($data - "0"))
    $python $script_dir/new_core.py --no_result --output_file $output_file --m 1 --k 0 --l 0 --network_m $m --network_k 0 --capacity $capacity --config_file $script_dir/$t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc $single_port --nprocs 20
    for p in "${parities[@]}"; do
        m=$(($data - $p))
        $python $script_dir/new_core.py --output_file $output_file --m 1 --k 0 --l 0 --network_m $m --network_k $p --capacity $capacity --config_file $script_dir/$t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc $single_port --nprocs 20
    done
done


: <<'COMMENT'
data=48 # m + k + l
parities=(1 2 3) # --k
network_parities=(2 3)

for t in "${tier_files[@]}"; do
    for p in "${parities[@]}"; do
        m=$(($data - $p - $l))
        single_port=""
        if [[ "$t" == *"singleport"* ]]; then
            echo "Running for single port tier: $t"
            single_port="--single_port_ssd"
        else
            echo "Running for dual port tier: $t"
        fi
        network_m=$(($data - "0" - $l))
        $python $script_dir/new_core.py --no_result --output_file $output_file --m $m --k $p --l $l --network_m $network_m --network_k 0 --capacity $capacity --config_file $script_dir/$t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc $single_port --nprocs 20
        for network_p in "${network_parities[@]}"; do
            network_m=$(($data - $network_p))
            $python $script_dir/new_core.py --output_file $output_file --m $m --k $p --l $l --network_m $network_m --network_k $network_p --capacity $capacity --config_file $script_dir/$t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc $single_port --nprocs 20
        done
    done
done
COMMENT