output_file="analysis_network_redundancy_group_$(date '+%Y%m%d_%H%M%S').txt"
data=(8 16 24 48) # m+k
parities=(0 1 2 3 4) # --k
replicas=(0 2 3 4)
rebuild_bw_ratios=(0.1 0.2 0.3 0.4 0.5)
capacity=128_000_000_000_000
dwpd=0.1
total_ssds=48
tier_files=("2tier.json")

#for t in "${tier_files[@]}"; do
#    for replica in "${replicas[@]}"; do
#        for c in "${capacity[@]}"; do
#            echo -e "\e[1;32m"
#            echo "Running simulation with stripe size: $s, datas: $m, parities: $p, capacity: $c, tier_file: $t"
#            echo "Command Lines : python3 new_core.py --output_file $output_file --m 1 --k 0 --inter_replicas $replica --capacity $c --config_file $t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc"
#            echo -e "\e[0m"
#            python3 new_core.py --output_file $output_file --m 1 --k 0 --inter_replicas $replica --capacity $c --config_file $t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc
#        done
#    done
#done
for t in "${tier_files[@]}"; do
    for s in "${data[@]}"; do
        for ratio in "${rebuild_bw_ratios[@]}";do
            for p in "${parities[@]}"; do
                m=$(($s - $p))
                if [ "$m" -lt 1 ]; then
                    continue
                fi
                echo -e "\e[1;32m"
                echo "Running simulation with stripe size: $s, datas: $m, parities: $p, capacity: $c, tier_file: $t"
                echo  "Command Lines : python3 new_core.py --output_file $output_file --m 1 --k 0 --network_m $m --network_k $p --capacity $c --config_file $t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc"
                echo -e "\e[0m"
                python3 new_core.py --output_file $output_file --m 1 --k 0 --network_m $m --network_k $p --capacity $capacity --config_file $t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc --rebuild_bw_ratio $ratio
            done
        done
    done
done

