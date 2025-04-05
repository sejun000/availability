#!/bin/bash
result_file_1="../analysis_local_redundancy_group_20250404_124056.txt"
result_file_2="../analysis_capacity_20250404_122306.txt"
result_file_3="../analysis_local_redundancy_group_20250401_081044.txt"
result_file_4="../analysis_all_redundancy_group_20250402_151454.txt"
result_file_5="../analysis_all_redundancy_group_20250403_002906.txt"

python3 ./plot_sub_graphs.py $result_file_1 \
--xlabel "Stripe Count (M + K)" --x_expr "m+k" \
--y1_label "Availability (Nines)" --y1_expr "avail_nines" \
--y2_label "Performance-Aware Availability (Nines)" \
--y1_title "(a) Traditional Availability Nines" \
--y2_title "(b) Performance-Aware Availability Nines" \
--y2_expr "credit_avail_nines" \
--z_expr "k" --legend "K" --y1_min 0 --y1_max 10 --y1_interval 1 \
--y2_min 0 --y2_max 10 --y2_interval 1 \
--output_file "avail_nines_local.pdf"

python3 ./plot_sub_graphs_2.py $result_file_1  \
--x1_label "Stripe Count (M + K)" --x1_expr "m+k" \
--x2_label "Party Count (K)" --x2_expr "k" \
--z_expr "k" --legend "K" \
--y1_label "Total Cost For 10 Years (M$)" --y1_expr "total_cost_for_10_years/1000000" \
--y2_label "Initial Cost,Repair Cost,Penalty Cost" \
--y1_title "(a) Total Cost versus Erasure Coding Stripe Count and Parity Count" \
--y2_title "(b) Distribution of Total Costs for Stripe Count 48" \
--filter2_expr "m+k==48" \
--y2_total_label "Total Cost For 10 Years (M$)" \
--y2_expr "initial_cost/1000000,repair_cost_for_10_years/1000000,down_cost_for_10_years/1000000" \
--y1_min 0 --y1_max 8 --y1_interval 1 \
--y2_min 0 --y2_max 8 --y2_interval 1 \
--legend_location1 "upper left" \
--legend_location2 "upper left" \
--output_file "total_cost_local.pdf"

python3 ./plot_grouped_bar.py $result_file_1 \
--xlabel "Stripe Count (M + K)" --x_expr "m+k" \
--ylabel "Cost Per Gigabyte ($/GB)" --y_expr "cost_per_gb" \
--z_expr "k" --legend "K" \
--output_file "cost_per_gb_local.pdf"

python3 ./plot_sub_graphs.py $result_file_2 \
--xlabel "Capacity (TeraBytes)" --x_expr "capacity/1000000000000" \
--filter_expr "m+k==48" \
--y1_label "Performance-Aware Availability (Nines)" --y1_expr "credit_avail_nines" \
--y2_label "Average Reconstruction Time (Hours)" \
--y1_title "(a) Performance-Aware Availability Nines" \
--y2_title "(b) Average Reconstruction Time" \
--y2_expr "avg_time_for_rebuilding" \
--z_expr "k" --legend "K" \
--output_file "avail_nines_capacity.pdf"

python3 ./plot_sub_graphs_2.py $result_file_2  \
--x1_label "Capacity (TeraBytes)" --x1_expr "capacity/1000000000000" \
--x2_label "Capacity (TeraBytes)" --x2_expr "capacity/1000000000000" \
--z_expr "k" --legend "K" \
--y1_label "Total Cost For 10 Years (M$)" --y1_expr "total_cost_for_10_years/1000000" \
--y2_label "Initial Cost(HW),Initial Cost(SSD),Repair Cost,Penalty Cost" \
--y1_title "(a) Total Cost versus Erasure Coding Stripe Size and Parity Count" \
--y2_title "(b) Distribution of Total Costs for Stripe Size 48" \
--filter2_expr "k==3" \
--y2_total_label "Total Cost For 10 Years (M$)" \
--y2_expr "(initial_cost-uncached_initial_cost)/1000000,uncached_initial_cost/1000000,repair_cost_for_10_years/1000000,down_cost_for_10_years/1000000" \
--y1_min 0 --y1_max 30 --y1_interval 3 \
--y2_min 0 --y2_max 20 --y2_interval 3 \
--legend_location1 "upper left" \
--legend_location2 "upper left" \
--output_file "total_cost_capacity.pdf"

python3 ./plot_grouped_bar.py $result_file_2 \
--xlabel "Capacity (TeraBytes)" --x_expr "capacity/1000000000000" \
--ylabel "Cost Per Gigabyte ($/GB)" --y_expr "cost_per_gb" \
--z_expr "k" --legend "K" \
--y_min 0 --y_max 0.5 --y_interval 0.05 \
--legend_location "upper left" \
--output_file "cost_per_gb_capacity.pdf"


python3 ./plot_sub_graphs.py $result_file_3 \
--x_expr "dwpd" --xlabel "Drive Writes Per Day (DWPD)" \
--filter_expr "k==3&dwpd!=3" \
--y1_label "Performance-Aware Availability (Nines)" --y1_expr "credit_avail_nines" \
--y2_label "Total Cost For 10 Years (M$)" --y2_expr "total_cost_for_10_years/1000000" \
--y1_title "(a) Performance-Aware Availability Nines" \
--y2_title "(b) Total Cost For 10 Years" \
--legend "TLCs" --z_expr "cached_ssds" \
--output_file "avail_nines_cache.pdf"

python3 ./plot_sub_graphs_3.py $result_file_3  \
--xlabel "TLCs" --x_expr "cached_ssds" \
--y_label "Initial Cost(HW),Initial Cost(TLC),Initial Cost(QLC),Repair Cost(HW),Repair Cost(TLC),Repair Cost(QLC),Penalty Cost" \
--y_expr "(initial_cost-uncached_initial_cost-cached_initial_cost)/1000000,cached_initial_cost/1000000,uncached_initial_cost/1000000,(repair_cost_for_10_years-cached_ssd_repair_cost_for_10_years/20-uncached_ssd_repair_cost_for_10_years/20)/1000000,cached_ssd_repair_cost_for_10_years/20/1000000,uncached_ssd_repair_cost_for_10_years/20/1000000,down_cost_for_10_years/1000000" \
--title1 "(a) DWPD=0.01" \
--title2 "(b) DWPD=0.1" \
--title3 "(c) DWPD=0.33" \
--title4 "(d) DWPD=1" \
--filter_expr1 "k==3&dwpd==0.01" \
--filter_expr2 "k==3&dwpd==0.1" \
--filter_expr3 "k==3&dwpd==0.33" \
--filter_expr4 "k==3&dwpd==1" \
--y_total_label "Total Cost For 10 Years (M$)" \
--y_min 0 --y_max 24 --y_interval 2 \
--legend_location "upper left" \
--output_file "total_cost_cache.pdf"


python3 ./plot_grouped_bar.py $result_file_3 \
--x_expr "dwpd" --xlabel "Drive Writes Per Day (DWPD)" \
--filter_expr "k==3&dwpd!=3" \
--ylabel "Cost Per Gigabyte ($/GB)" --y_expr "cost_per_gb" \
--legend "TLCs" --z_expr "cached_ssds" \
--y_min 0 --y_max 1.2 --y_interval 0.2 \
--legend_location "upper left" \
--output_file "cost_per_gb_cache.pdf"

python3 ./plot_scatter.py $result_file_4 \
--filtered_expr "(~((k==0)&(network_k==0)))&(((m+k)>=24)|((network_m+network_k)<=24))&(box_mttf==41666)" \
--legend "Rebuild Method" \
--y_label "Traditional Availability (Nines), Performance-Aware Availability (Nines), Cost Per Gigabyte ($/GB)" --y_expr "avail_nines,credit_avail_nines,cost_per_gb" \
--x_expr "m/(m+k)*(network_m)/(network_m+network_k) * 100" --x_label "Effective Capacity Ratio (%)" \
--y_min 0,0,0.15 --y_max 14,14,0.35 --y_interval 2,2,0.05 \
--output_file "inter_intra_cost_per_gb.pdf"

python3 ./plot_scatter.py $result_file_5 \
--filtered_expr "(~((k==0)&(network_k==0)))&(((m+k)>=24)|((network_m+network_k)<=24))&(rebuild_bw_ratio==0.5)" \
--legend "Rebuild Method" \
--y_label "Traditional Availability (Nines), Performance-Aware Availability (Nines), Cost Per Gigabyte ($/GB)" --y_expr "avail_nines,credit_avail_nines,cost_per_gb" \
--x_expr "m/(m+k)*(network_m)/(network_m+network_k) * 100" --x_label "Effective Capacity Ratio (%)" \
--y_min 0,0,0.15 --y_max 14,14,0.35 --y_interval 2,2,0.05 \
--output_file "inter_intra_cost_per_gb_per_rebuild_bw.pdf"

#&((networ_k>0&k>0&k<=2&network_k<=2)|((network_k==0)|(k==0)))
#&((networ_k>0&k>0&k<=2&network_k<=2)| network_k==0| k==0)