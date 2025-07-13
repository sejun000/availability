#!/bin/bash
script_dir="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"

output_file="result_intra_inter_$(date '+%Y%m%d_%H%M%S').csv"
#data=48 # m + k + l
stripes=(24 48)
parities=(1 2 3 4 5 6) # --k
l=0 # --l
capacity=(64_000_000_000_000)
dwpd=0.2
total_ssds=48
python="pypy3"
target_ratio=(0.5 1 1)

tier_files=("2tier.json" "2tier.json" "2tier.json")

#active_actives=("--active_active")
active_actives=("" "" "--active_active")
box_mttfs=(1440 416666666 416666666) # 1 year, 1 month, 1 day, 1 hour
io_module_mttrs=(0.25 0.25 0.25) # 4 hours

for active_active in "${active_actives[@]}"; do
    target_ratio=${target_ratio[$i]}
    t="${tier_files[$i]}"
    box_mttf=${box_mttfs[$i]}
    io_module_mttr=${io_module_mttrs[$i]}
    i=$(($i + 1))
  
    single_port=""
    data=48 # m + k + l
    parities=(1 2 3 4 5 6) # --k
    if [[ "$t" == *"singleport"* ]]; then
        echo "Running for single port tier: $t"
        single_port="--single_port_ssd"
    fi
    for stripe in "${stripes[@]}"; do
        for p in "${parities[@]}"; do
            m=$(($stripe - $p - $l))
            $python $script_dir/new_core.py --output_file $output_file --m $m --k $p --l $l --capacity $capacity --config_file $script_dir/$t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc $single_port --target_perf_ratio $target_ratio $active_active --box_mttf $box_mttf --io_module_mttr $io_module_mttr
        done
    done

# inter erasure coding
    single_port=""
    if [[ "$t" == *"singleport"* ]]; then
        echo "Running for single port tier: $t"
        single_port="--single_port_ssd"
    fi
    m=$(($data - "0"))
    $python $script_dir/new_core.py --no_result --output_file $output_file --m 1 --k 0 --l 0 --network_m $m --network_k 0 --capacity $capacity --config_file $script_dir/$t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc $single_port --target_perf_ratio $target_ratio $active_active  --box_mttf $box_mttf --io_module_mttr $io_module_mttr
    for stripe in "${stripes[@]}"; do
        for p in "${parities[@]}"; do
            m=$(($stripe - $p))
            $python $script_dir/new_core.py --output_file $output_file --m 1 --k 0 --l 0 --network_m $m --network_k $p --capacity $capacity --config_file $script_dir/$t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc $single_port --target_perf_ratio $target_ratio $active_active  --box_mttf $box_mttf --io_module_mttr $io_module_mttr
        done
    done

    # multi level erasure coding
    data=48 # m + k + l
    parities=(1 2 3) # --k
    network_parities=(1 2 3)

    for p in "${parities[@]}"; do
        m=$(($data - $p - $l))
        single_port=""
        if [[ "$t" == *"singleport"* ]]; then
            echo "Running for single port tier: $t"
            single_port="--single_port_ssd"
        fi
        for stripe in "${stripes[@]}"; do
            network_m=$(($stripe - "0" - $l))
            $python $script_dir/new_core.py --no_result --output_file $output_file --m $m --k $p --l $l --network_m $network_m --network_k 0 --capacity $capacity --config_file $script_dir/$t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc $single_port --target_perf_ratio $target_ratio $active_active  --box_mttf $box_mttf --io_module_mttr $io_module_mttr
            for network_p in "${network_parities[@]}"; do
                network_m=$(($stripe - $network_p))
                $python $script_dir/new_core.py --output_file $output_file --m $m --k $p --l $l --network_m $network_m --network_k $network_p --capacity $capacity --config_file $script_dir/$t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc $single_port --target_perf_ratio $target_ratio $active_active  --box_mttf $box_mttf --io_module_mttr $io_module_mttr
            done
        done
    done

done