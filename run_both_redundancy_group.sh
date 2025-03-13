output_file="analysis_both_redundancy_group_$(date '+%Y%m%d_%H%M%S').txt"
data=(8 16 24 48) # m+k
parities=(1 2 3) # --k
network_data=(8 16 24 48) # m+k
network_parity=(0 1 2) # m+k
capacity=128_000_000_000_000
dwpd=0.1
total_ssds=48
tier_files=("2tier.json")
for t in "${tier_files[@]}"; do
    for s in "${data[@]}"; do
        for p in "${parities[@]}"; do
            for s_network in "${network_data[@]}"; do
                for p_network in "${network_parity[@]}"; do
                    m=$(($s - $p))
                    network_m=$(($s_network - $p_network))
                    if [ "$m" -lt 1 ]; then
                        continue
                    fi
                    for c in "${capacity[@]}"; do
                        echo -e "\e[1;32m"
                        echo "Running simulation with stripe size: $s, datas: $m, parities: $p, capacity: $c, tier_file: $t"
                        echo -e "\e[0m"
                        if [[ "network_m" -eq 0 ]]; then
                            python3 new_core.py --output_file temp.txt --m $m --k $p --network_m $network_m --network_k $p_network --capacity $c --config_file $t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc
                        else
                            python3 new_core.py --output_file $output_file --m $m --k $p --network_m $network_m --network_k $p_network --capacity $c --config_file $t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc
                        fi
                    done
                done
            done
        done
    done
done