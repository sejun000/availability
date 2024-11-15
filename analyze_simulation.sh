#!/bin/bash
output_file="analysis_ssds_$(date '+%Y%m%d_%H%M%S').txt"
#stripe_size=(8 16 24 48) # m+k
#stripe_size=(24 48) # m+k
#parities=(0 1 2 3 4 5) # --k
stripe_size=(1)
parities=(0)
network_stripes=(6 8 12) # network_m+k
network_parities=(0 1 2) # --network_k
dwpds=(1)
tbwpds=(4.5)
#tbwpds=(1.5 3.3 4.5)
#capacity=(8_000_000_000_000 16_000_000_000_000 32_000_000_000_000 64_000_000_000_000)
capacity=(64_000_000_000_000)
#type=("tlc" "qlc")
type=("qlc")
run_once=false
additional_params="--simulation"

# iterate analyze_ssd_only.py over all possible combinations of parameters
echo "" > $output_file
for t in "${type[@]}"; do
    current_type=""
    if [ "$t" = "qlc" ]; then
        current_type="--qlc"
    fi
    for s in "${stripe_size[@]}"; do
        for p in "${parities[@]}"; do
            m=$(($s - $p))
            # if p == 0, we set m to 1
            if [ "$p" = 0 ]; then
                if [ "$run_once" = true ]; then
                    continue
                fi
                m=1
                run_once=true
            fi
            for d in "${dwpds[@]}"; do
                for tbw in "${tbwpds[@]}"; do
                    for c in "${capacity[@]}"; do
                        for network_s in "${network_stripes[@]}"; do
                            for network_p in "${network_parities[@]}"; do
                                network_m=$(($network_s - $network_p))
                                if [ "$use_tbwpd" = true ]; then
                                    use_tbwpd_flag="--use_tbwpd"
                                fi
                                # change color of echo
                                echo -e "\e[1;32m"
                                echo "Running simulation with type: $current_type, stripe size: $s, datas: $m, parities: $p, capacity: $c, dwpd: $d, tbwpd: $tbw, use_tbwpd: $use_tbwpd, network_m: $network_m, network_k: $network_p"
                                echo -e "\e[0m"
                                timeout 10m python3 new_core.py $current_type --output_file $output_file --m $m --k $p --capacity $c $current_type --dwpd $d --tbwpd $tbw  --network_m $network_m --network_k $network_p $additional_params
                            done
                        done
                    done
                done
            done
        done
    done
done
