#!/bin/bash
output_file="analysis_ssds_$(date '+%Y%m%d_%H%M%S').txt"
stripe_size=(6 8 12 16 24 48) # m+k+standby
parities=(0 1 2 3 4 5 6) # --k
#dwpds=(0.1 0.5 1 2 3)
dwpds=(1)
tbwpds=(1.5 3.3 4.5)
capacity=(8_000_000_000_000 16_000_000_000_000 32_000_000_000_000 64_000_000_000_000)
type=("tlc" "qlc")
standby_ssd=0
use_tbwpd=true

# iterate analyze_ssd_only.py over all possible combinations of parameters
echo "" > $output_file
for t in "${type[@]}"; do
    current_type=""
    if [ "$t" = "qlc" ]; then
        current_type="--qlc"
    fi
    for s in "${stripe_size[@]}"; do
        for p in "${parities[@]}"; do
            for d in "${dwpds[@]}"; do
                for tbw in "${tbwpds[@]}"; do
                    # we set standby ssd to 0
                    m=$(($s - $standby_ssd - $p))
                    for c in "${capacity[@]}"; do
                        if [ "$use_tbwpd" = true ]; then
                            use_tbwpd_flag="--use_tbwpd"
                        fi
                        # change color of echo
                        echo -e "\e[1;32m"
                        echo "Running simulation with type: $current_type, stripe size: $s, datas: $m, parities: $p, standby_ssd: $standby_ssd, capacity: $c, dwpd: $d, tbwpd: $tbw, use_tbwpd: $use_tbwpd"
                        echo -e "\e[0m"
                        python3 analyze_ssd_only.py $current_type --output_file $output_file --m $m --k $p --standby_ssd $standby_ssd --capacity $c $current_type --dwpd $d --tbwpd $tbw --use_tbwpd
                    done
                done
            done
        done
    done
done
