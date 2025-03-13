output_file="analysis_network_redundancy_group_$(date '+%Y%m%d_%H%M%S').txt"
data=(8 16 24 48) # m+k
parities=(0 1 2 3 4) # --k
capacity=128_000_000_000_000
dwpd=0.1
total_ssds=48
tier_files=("2tier.json")
for t in "${tier_files[@]}"; do
    for s in "${data[@]}"; do
        for p in "${parities[@]}"; do
            m=$(($s - $p))
            if [ "$m" -lt 1 ]; then
                continue
            fi
            for c in "${capacity[@]}"; do
                echo -e "\e[1;32m"
                echo "Running simulation with stripe size: $s, datas: $m, parities: $p, capacity: $c, tier_file: $t"
                echo -e "\e[0m"
                python3 new_core.py --output_file $output_file --m 1 --k 0 --network_m $m --network_k $p --capacity $c --config_file $t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc
            done
        done
    done
done