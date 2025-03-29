output_file="analysis_local_redundancy_group_$(date '+%Y%m%d_%H%M%S').txt"
data=(8 16 24 48) # m+k
#data=(32) # m+k
parities=(1 2 3 4) # --k
capacity=64_000_000_000_000
dwpd=0.1
total_ssds=48
tier_files=("2tier.json")
python="pypy3"
#tier_files=("2tier.json" "3tier.json")
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
                echo "Command Lines : $python new_core.py --output_file $output_file --m $m --k $p --capacity $c --config_file $t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc"
                echo -e "\e[0m"
                $python new_core.py --output_file $output_file --m $m --k $p --capacity $c --config_file $t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc
            done
        done
    done
done