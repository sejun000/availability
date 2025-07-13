#!/bin/bash
result_file_2="result_capacity_20250709_161833.csv"
result_file_3="result_cache_20250712_152425.csv"
result_file_4="result_intra_inter_20250710_115718.csv"
#result_file_4="../analysis_all_redundancy_group_20250402_151454.txt"
#result_file_4="../analysis_all_redundancy_group_20250409_151328.txt"
#result_file_4="../analysis_all_redundancy_group_20250409_151328.txt"
#result_file_4="../analysis_all_redundancy_group_20250410_080405.txt"
#result_file_4="../analysis_all_redundancy_group_20250411_234854.txt"

result_file_5="result_tiering_dual_single_20250710_232134.csv"


: <<'COMMENT'

python3 ./plot_sub_graphs.py $result_file_1 \
--xlabel "Stripe Count (M + K)" --x_expr "m+k" \
--y1_label "Nines" --y1_expr "avail_nines" \
--y2_label "Nines" \
--y1_title "(a) Traditional Availability" \
--y2_title "(b) Performance-Aware Availability" \
--y2_expr "credit_avail_nines" \
--z_expr "k" --legend "K" --y1_min 0 --y1_max 10 --y1_interval 1 \
--y2_min 0 --y2_max 10 --y2_interval 1 \
--output_file "avail_nines_local.pdf"

python3 ./plot_sub_graphs_2.py $result_file_1  \
--x1_label "Stripe Count (M + K)" --x1_expr "m+k" \
--x2_label "Party Count (K)" --x2_expr "k" \
--z_expr "k" --legend "K" \
--y1_label "M$" --y1_expr "total_cost_for_10_years/1000000" \
--y2_label "Initial,Repair,Penalty" \
--y1_title "(a) Total Cost" \
--y2_title "(b) Distribution of Total Cost" \
--filter2_expr "m+k==48" \
--y2_total_label "M$" \
--y2_expr "initial_cost/1000000,repair_cost_for_10_years/1000000,down_cost_for_10_years/1000000" \
--y1_min 0 --y1_max 25 --y1_interval 5 \
--y2_min 0 --y2_max 25 --y2_interval 5 \
--legend_location1 "upper left" \
--legend_location2 "upper left" \
--output_file "total_cost_local.pdf"

python3 ./plot_grouped_bar.py $result_file_1 \
--xlabel "Stripe Count (M + K)" --x_expr "m+k" \
--ylabel "$/Gigabyte" --y_expr "cost_per_gb" \
--z_expr "k" --legend "K" \
--legend_location "upper left" \
--y_min 0 --y_max 0.8 --y_interval 0.1 \
--output_file "cost_per_gb_local.pdf"
COMMENT

python3 ./plot_sub_graphs_4.py \
--xlabel "Capacity (TB)" --x_expr "capacity/1000000000000" \
--filter_expr "m+k==48&k<=4" \
--y1_label "Nines" \
--y2_label "Nines" --y2_expr "credit_avail_nines" \
--y1_title "(a) Durability with Markov Model" \
--y2_title "(b) Performance-Aware Availability" \
--y1_expr "avail_nines" \
--y2_expr "credit_avail_nines" \
--z_expr "k" --legend "K" \
--y1_min 0 --y1_max 12 --y1_interval 2 \
--y2_min 0 --y2_max 12 --y2_interval 2 \
--output_file "avail_nines_capacity.pdf" \
$result_file_2

python3 ./plot_grouped_bar.py \
--filter_expr "m+k==48&k<=4" \
--xlabel "Capacity (TB)" --x_expr "capacity/1000000000000" \
--ylabel "Hours" --y_expr "avg_time_for_rebuilding" \
--z_expr "k" --legend "K" \
--y_min 0 --y_max 250 --y_interval 50 \
--legend_location "upper left" \
--output_file "rebuilding_time_capacity.pdf" \
$result_file_2

python3 ./plot_sub_graphs_2.py \
--x1_label "Capacity (TB)" --x1_expr "capacity/1000000000000" \
--x2_label "Capacity (TB)" --x2_expr "capacity/1000000000000" \
--z_expr "k" --legend "K" \
--y1_label "M$" --y1_expr "total_cost_for_10_years/1000000" \
--y2_label "Initial(HW),Initial(SSD),Repair,Penalty" \
--y1_title "(a) Unified Cost" \
--y2_title "(b) Distribution of Unified Cost" \
--filter1_expr "k<=4" \
--filter2_expr "k==2" \
--y2_total_label "M$" \
--y2_expr "(initial_cost-uncached_initial_cost)/1000000,uncached_initial_cost/1000000,repair_cost_for_10_years/1000000,down_cost_for_10_years/1000000" \
--y1_min 0 --y1_max 50 --y1_interval 5 \
--y2_min 0 --y2_max 50 --y2_interval 5 \
--legend_location1 "upper left" \
--legend_location2 "upper left" \
--output_file "total_cost_capacity.pdf" \
$result_file_2 

python3 ./plot_grouped_bar.py \
--xlabel "Capacity (TB)" --x_expr "capacity/1000000000000" \
--ylabel "$/GB" --y_expr "cost_per_gb" \
--z_expr "k" --legend "K" \
--y_min 0 --y_max 1.0 --y_interval 0.2 \
--legend_location "upper left" \
--output_file "cost_per_gb_capacity.pdf" \
$result_file_2


python3 ./plot_sub_graphs.py \
--x_expr "dwpd" --xlabel "Host write workload (\$DWPD_{host}\$)" \
--filter_expr "dwpd!=3&cached_ssds!=12&cached_ssds!=10&qlc" \
--y1_label "Nines" --y1_expr "credit_avail_nines" \
--y2_label "M$" --y2_expr "total_cost_for_10_years/1000000" \
--y1_title "(a) Performance-Aware Availability" \
--y2_title "(b) Unified Cost" \
--legend "TLCs" --z_expr "cached_ssds" \
--y1_min 0 --y1_max 10 --y1_interval 2 \
--y2_interval 10 \
--y2_max 80 \
--legend_type "s" \
--output_file "avail_nines_cache.pdf" \
$result_file_3

python3 ./plot_sub_graphs_3.py \
--xlabel "TLCs" --x_expr "cached_ssds" \
--y_label "Initial(HW),Initial(TLC),Initial(QLC),Repair(HW),Repair(TLC),Repair(QLC),Penalty Cost" \
--y_expr "(initial_cost-uncached_initial_cost-cached_initial_cost)/1000000,cached_initial_cost/1000000,uncached_initial_cost/1000000,(repair_cost_for_10_years-cached_ssd_repair_cost_for_10_years-uncached_ssd_repair_cost_for_10_years)/1000000,cached_ssd_repair_cost_for_10_years/1000000,uncached_ssd_repair_cost_for_10_years/1000000,down_cost_for_10_years/1000000" \
--title1 "(a) \$DWPD_{host}\$=0.02" \
--title2 "(b) \$DWPD_{host}\$=0.2" \
--title3 "(c) \$DWPD_{host}\$=0.66" \
--title4 "(d) \$DWPD_{host}\$=2" \
--filter_expr1 "dwpd==0.02&cached_ssds!=12&cached_ssds!=10&qlc" \
--filter_expr2 "dwpd==0.2&cached_ssds!=12&cached_ssds!=10&qlc" \
--filter_expr3 "dwpd==0.66&cached_ssds!=12&cached_ssds!=10&qlc" \
--filter_expr4 "dwpd==2&cached_ssds!=12&cached_ssds!=10&qlc" \
--y_total_label "M$" \
--y_min "0,0,0,0" --y_max "20,20,20,80" --y_interval "5,5,5,10" \
--legend_location "upper left" \
--z_col 4 \
--output_file "total_cost_cache.pdf" \
$result_file_3  



python3 ./plot_grouped_bar.py \
--x_expr "dwpd" --xlabel "Host write workload (\$DWPD_{host}\$)" \
--filter_expr "dwpd!=3&cached_ssds!=12&cached_ssds!=10&qlc" \
--ylabel "$/GB" --y_expr "cost_per_gb" \
--legend "TLCs" --z_expr "cached_ssds" \
--y_min 0 --y_max 2.0 --y_interval 0.5 \
--legend_location "upper left" \
--legend_type "s" \
--output_file "cost_per_gb_cache.pdf" \
$result_file_3



python3 ./plot_scatter.py \
--filter_expr "active_active&(target_perf_ratio==1)&((network_k > 0 & k==0)|(network_k ==0 &k>0)|(network_k>0 & k > 0 & k < 3))" \
--legend "EC Scheme" \
--y_label "Nines, Nines, M$, $/GB" --y_expr "avail_nines,credit_avail_nines,total_cost_for_10_years/1000000,cost_per_gb" \
--x_expr "m/(m+k)*(network_m)/(network_m+network_k) * 100" --x_label "Effective Capacity Ratio (%)" \
--y_min 0,0,0,0 --y_max 14,14,40,0.70 --y_interval 2,2,5,0.1 \
--titles "(a) Traditional Availability, (b) Performance-Aware Availability, (c) Unified Cost, (d) UPEC" \
--output_file "scatter_1.pdf" \
$result_file_4

python3 ./plot_scatter.py \
--filter_expr "~active_active&(target_perf_ratio==1)&((network_k > 0 & k==0)|(network_k ==0 &k>0)|(network_k>0 & k > 0 & k < 3))" \
--legend "EC Scheme" \
--y_label "Nines, Nines, M$, $/GB" --y_expr "avail_nines,credit_avail_nines,total_cost_for_10_years/1000000,cost_per_gb" \
--x_expr "m/(m+k)*(network_m)/(network_m+network_k) * 100" --x_label "Effective Capacity Ratio (%)" \
--y_min 0,0,0,0 --y_max 14,14,40,0.70 --y_interval 2,2,5,0.1 \
--titles "(a) Traditional Availability, (b) Performance-Aware Availability, (c) Unified Cost, (d) UPEC" \
--output_file "scatter_2.pdf" \
$result_file_4 


python3 ./plot_scatter.py \
--filter_expr "~active_active&(target_perf_ratio==0.5)&((network_k > 0 & k==0)|(network_k ==0 &k>0)|(network_k>0 & k > 0 & k < 3))" \
--legend "EC Scheme" \
--y_label "Nines, Nines, M$, $/GB" --y_expr "avail_nines,credit_avail_nines,total_cost_for_10_years/1000000,cost_per_gb" \
--x_expr "m/(m+k)*(network_m)/(network_m+network_k) * 100" --x_label "Effective Capacity Ratio (%)" \
--y_min 0,0,0,0 --y_max 14,14,40,0.70 --y_interval 2,2,5,0.1 \
--titles "(a) Traditional Availability, (b) Performance-Aware Availability, (c) Unified Cost, (d) UPEC" \
--output_file "scatter_3.pdf" \
$result_file_4 


python3 ./plot_grouped_bar.py \
--xlabel "Parity Count" --x_expr "k" \
--ylabel "Nines" --y_expr "credit_avail_nines" \
--z_expr "(('2_' if 'singleport' in config_file else '1_') + ('3tier' if '3tier' in config_file else '2tier'))" \
--legend_labels "D_2,D_3,S_2,S_3" \
--y_min 0 --y_max 14 --y_interval 2 \
--legend_location "upper left" \
--output_file "avail_nines_per_tiers.pdf" \
$result_file_5



python3 ./plot_sub_graphs_2.py \
--x1_label "Parity Count" --x1_expr "k" \
--x2_label "SSD Type + Tier" --x2_expr "(('S' if 'singleport' in config_file else 'D') + ('_3tier' if '3tier' in config_file else '_2tier'))" \
--z_expr "(('2_' if 'singleport' in config_file else '1_') + ('3tier' if '3tier' in config_file else '2tier'))" \
--y1_label "M$" --y1_expr "total_cost_for_10_years/1000000" \
--y2_label "Initial(HW),Initial(SSD),Repair,Penalty" \
--y1_title "(a) Unified Cost" \
--y2_title "(b) Distribution of Unified Cost" \
--filter1_expr "k<=4" \
--filter2_expr "k==3" \
--y2_total_label "M$" \
--y2_expr "(initial_cost-uncached_initial_cost)/1000000,uncached_initial_cost/1000000,(repair_cost_for_10_years)/1000000,down_cost_for_10_years/1000000" \
--y1_min 0 --y1_max 30 --y1_interval 5 \
--y2_min 0 --y2_max 30 --y2_interval 5 \
--legend_location1 "upper left" \
--legend_location2 "upper left" \
--output_file "total_cost_per_tiers.pdf" \
--legend_type1 "s" \
--legend_labels1 "D_2,D_3,S_2,S_3" \
$result_file_5 


python3 ./plot_grouped_bar.py \
--xlabel "Parity Count" --x_expr "k" \
--ylabel "$/GB" --y_expr "cost_per_gb" \
--z_expr "(('2_' if 'singleport' in config_file else '1_') + ('3tier' if '3tier' in config_file else '2tier'))" \
--legend_labels "D_2,D_3,S_2,S_3" \
--y_min 0 --y_max 1.2 --y_interval 0.2 \
--legend_location "upper left" \
--output_file "cost_per_gb_per_tiers.pdf" \
$result_file_5


#python3 ./line_chart.py line_chart.pdf
#python3 ./plot_scatter.py $result_file_4 \
#--filtered_expr "(~((k==0)&(network_k==0)))&((m+k>=20)|(network_m+network_k)>=24)&(box_mttf==416)" \
#--legend "Rebuild Method" \
#--y_label "Availability (Nines), Availability (Nines), M$, $/GB" --y_expr "avail_nines,credit_avail_nines,total_cost_for_10_years/1000000,cost_per_gb" \
#--x_expr "m/(m+k)*(network_m)/(network_m+network_k) * 100" --x_label "Effective Capacity Ratio (%)" \
#--y_min 0,0,0,0.15 --y_max 14,14,40,0.8 --y_interval 2,2,5,0.05 \
#--titles "(a) Traditional, (b) Performance-Aware, (c) Total Cost, (d) Cost Per Gigabyte" \
#--output_file "inter_intra_cost_per_gb.pdf"

#python3 ./plot_scatter.py $result_file_5 \
#--filtered_expr "(~((k==0)&(network_k==0)))&(((m+k)>=24)|((network_m+network_k)<=24))&(rebuild_bw_ratio==0.5)" \
#--legend "Rebuild Method" \
#--y_label "Traditional Availability (Nines), Performance-Aware Availability (Nines), Cost Per Gigabyte ($/GB)" --y_expr "avail_nines,credit_avail_nines,cost_per_gb" \
#--x_expr "m/(m+k)*(network_m)/(network_m+network_k) * 100" --x_label "Effective Capacity Ratio (%)" \
#--y_min 0,0,0.15 --y_max 14,14,0.35 --y_interval 2,2,0.05 \
#--output_file "inter_intra_cost_per_gb_per_rebuild_bw.pdf"

#&((networ_k>0&k>0&k<=2&network_k<=2)|((network_k==0)|(k==0)))
#&((networ_k>0&k>0&k<=2&network_k<=2)| network_k==0| k==0)