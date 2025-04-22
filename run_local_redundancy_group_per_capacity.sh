output_file="analysis_capacity_$(date '+%Y%m%d_%H%M%S').txt"
#data=(8 16 32) # m+k
#parities=(1 2 3 4) # --k
data=(48) # m+k
parities=(1 2 3 4) # --k
capacity=(8_000_000_000_000 16_000_000_000_000 32_000_000_000_000 64_000_000_000_000 128_000_000_000_000 256_000_000_000_000)
dwpd=0.2
total_ssds=48
tier_files=("2tier.json")
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
                echo -e "\e[0m"
                pypy3 new_core.py --output_file $output_file --m $m --k $p --capacity $c --config_file $t --simulation --total_ssds $total_ssds --dwpd $dwpd --qlc
            done
        done
    done
done