#!/bin/bash
set -e
result_file_1="../plot/result_capacity_20250628_065022.csv"

result_file_1_low_dwpd="../plot/result_capacity_20250702_161255.csv"

result_file_2="../plot/result_clustered_declustered_parity_20250624_225000.csv"

result_file_3="../plot/result_dual_single_20250628_211315.csv"

result_file_4="../plot/result_intra_inter_20250624_163809.csv"

result_file_5="../plot/result_cache_20250628_070005.csv"

result_file_6="../plot/result_tiering_dual_single_20250628_005955.csv"

result_file_7="../plot/result_stripe_20250702_124004.csv"


# 5.2 Capacity and Parity Count

python3 ./plot_grouped_bar.py \
--x_label "Capacity (TB)" --x_expr "capacity/1000000000000" \
--y_label "Nines" --y_expr "credit_avail_nines" \
--z_expr "k" --legend "K" \
--y_min 0 --y_max 14 --y_interval 2 \
--legend_location "upper left" \
--output_file "avail_nines_per_capacity.png" \
$result_file_1

python3 ./plot_grouped_bar.py \
--x_label "Capacity (TB)" --x_expr "capacity/1000000000000" \
--y_label "Hours" --y_expr "avg_time_for_rebuilding" \
--z_expr "k" --legend "K" \
--y_min 0 --y_max 600 --y_interval 100 \
--legend_location "upper left" \
--output_file "rebuilding_time_per_capacity.png" \
$result_file_1

python3 ./plot_grouped_bar.py \
--x_label "Capacity (TB)" --x_expr "capacity/1000000000000" \
--y_label "Total cost for 10 years(M$)" --y_expr "total_cost_for_10_years/1000000" \
--z_expr "k" --legend "K" \
--y_min 0 --y_max 100 --y_interval 20 \
--legend_location "upper left" \
--output_file "total_cost_per_capacity.png" \
$result_file_1 

python3 ./plot_stacked_bar.py \
--x_label "Capacity (TB)" --x_expr "capacity/1000000000000" \
--y_label "Total cost for 10 years(M$)" \
--filter_expr "k==3" \
--y_expr "(initial_cost-uncached_initial_cost)/1000000,uncached_initial_cost/1000000,(repair_cost_for_10_years-uncached_ssd_repair_cost_for_10_years)/1000000,uncached_ssd_repair_cost_for_10_years/1000000,down_cost_for_10_years/1000000" \
--legend "Initial(HW),Initial(SSD),Repair(HW),Repair(SSD),Penalty" \
--y_min 0 --y_max 60 --y_interval 10 \
--legend_location "upper left" \
--output_file "cost_distribution_per_capacity.png" \
$result_file_1 

python3 ./plot_grouped_bar.py \
--x_label "Capacity (TB)" --x_expr "capacity/1000000000000" \
--y_label "$/GB" --y_expr "cost_per_gb" \
--z_expr "k" --legend "K" \
--y_min 0 --y_max 1.4 --y_interval 0.2 \
--legend_location "upper left" \
--output_file "cost_per_gb_per_capacity.png" \
$result_file_1

python3 ./plot_grouped_bar.py \
--x_label "Capacity (TB)" --x_expr "capacity/1000000000000" \
--y_label "$/GB" --y_expr "cost_per_gb" \
--z_expr "k" --legend "K" \
--y_min 0 --y_max 0.6 --y_interval 0.2 \
--legend_location "upper left" \
--output_file "cost_per_gb_per_capacity_low_dwpd.png" \
$result_file_1_low_dwpd

# 5.4 Clustered and Declustered Parity Count

python3 ./plot_grouped_bar.py \
--x_label "Gap disk count (L)" --x_expr "l" \
--y_label "Nines" --y_expr "credit_avail_nines" \
--z_expr "k" --legend "K" \
--y_min 0 --y_max 14 --y_interval 2 \
--legend_location "upper left" \
--output_file "avail_nines_per_gap_disk_count.png" \
$result_file_2

python3 ./plot_grouped_bar.py \
--x_label "Gap disk count (L)" --x_expr "l" \
--y_label "Hours" --y_expr "avg_time_for_rebuilding" \
--z_expr "k" --legend "K" \
--y_min 0 --y_max 700 --y_interval 100 \
--legend_location "upper left" \
--output_file "rebuilding_time_per_gap_disk_count.png" \
$result_file_2

python3 ./plot_grouped_bar.py \
--x_label "Gap disk count (L)" --x_expr "l" \
--y_label "Total cost for 10 years(M$)" --y_expr "total_cost_for_10_years/1000000" \
--z_expr "k" --legend "K" \
--y_min 0 --y_max 60 --y_interval 10 \
--legend_location "upper left" \
--output_file "total_cost_per_gap_disk_count.png" \
$result_file_2 

python3 ./plot_grouped_bar.py \
--x_label "Gap disk count (L)" --x_expr "l" \
--y_label "$/GB" --y_expr "cost_per_gb" \
--z_expr "k" --legend "K" \
--y_min 0 --y_max 1.4 --y_interval 0.2 \
--legend_location "upper left" \
--output_file "cost_per_gb_per_gap_disk_count.png" \
$result_file_2 


# 5.4 Single and Dual Port SSDs

python3 ./plot_grouped_bar.py \
--filter_expr "(network_k==0&network_m==8)|(k==0)" \
--x_label "Parity Count" --x_expr "k+network_k" \
--y_label "Nines" --y_expr "credit_avail_nines" \
--z_expr "str(int(network_k>0)) + '_' + config_file" --legend_labels "Single(Intra),Dual(Intra),Single(Inter),Dual(Inter)" \
--y_min 0 --y_max 14 --y_interval 2 \
--legend_location "upper left" \
--output_file "avail_nines_per_port.png" \
$result_file_3


python3 ./plot_grouped_bar.py \
--filter_expr "(network_k==0&network_m==8)|(k==0)" \
--x_label "Parity Count" --x_expr "k+network_k" \
--y_label "Hours" --y_expr "avg_time_for_rebuilding" \
--z_expr "str(int(network_k>0)) + '_' + config_file" --legend_labels "Single(Intra),Dual(Intra),Single(Inter),Dual(Inter)" \
--y_min 0 --y_max 600 --y_interval 100 \
--legend_location "upper left" \
--output_file "rebuilding_time_per_port.png" \
$result_file_3


python3 ./plot_grouped_bar.py \
--filter_expr "(network_k==0&network_m==8)|(k==0)" \
--x_label "Parity Count" --x_expr "k+network_k" \
--y_label "Total cost for 10 years(M$)" --y_expr "total_cost_for_10_years/1000000" \
--z_expr "str(int(network_k>0)) + '_' + config_file" --legend_labels "Single(Intra),Dual(Intra),Single(Inter),Dual(Inter)" \
--y_min 0 --y_max 60 --y_interval 10 \
--legend_location "upper left" \
--output_file "total_cost_per_port.png" \
$result_file_3


python3 ./plot_grouped_bar.py \
--filter_expr "(network_k==0&network_m==8)|(k==0)" \
--x_label "Parity Count" --x_expr "k+network_k" \
--y_label "$/GB" --y_expr "cost_per_gb" \
--z_expr "str(int(network_k>0)) + '_' + config_file" --legend_labels "Single(Intra),Dual(Intra),Single(Inter),Dual(Inter)" \
--y_min 0 --y_max 1.4 --y_interval 0.2 \
--legend_location "upper left" \
--output_file "cost_per_gb_per_port.png" \
$result_file_3

# 5.4 Inter / Intra Tiered Parity Count

python3 ./plot_scatter.py \
--filter_expr "active_active&config_file.str.contains('2tier.json')&((k>0&network_k==0)|(k==0&network_k>0)|((network_k>0)&(k>=1)))" \
--legend "EC Policy" \
--y_label "Nines, Nines, M$, $/GB" --y_expr "avail_nines,credit_avail_nines,total_cost_for_10_years/1000000,cost_per_gb" \
--x_expr "m/(m+k)*(network_m)/(network_m+network_k) * 100" --x_label "Effective Capacity Ratio (%)" \
--y_min 0,0,0,0 --y_max 14,14,40,1.4 --y_interval 2,2,5,0.2 \
--titles "(a) Availability, (b) Effective Availability, (c) Total Cost, (d) Cost per GB" \
--output_file "scatter_2tier_perf_active_active.png" \
$result_file_4

python3 ./plot_scatter.py \
--filter_expr "(~active_active)&config_file.str.contains('2tier.json')&((k>0&network_k==0)|(k==0&network_k>0)|((network_k>0)&(k>=1)))" \
--legend "EC Policy" \
--y_label "Nines, Nines, M$, $/GB" --y_expr "avail_nines,credit_avail_nines,total_cost_for_10_years/1000000,cost_per_gb" \
--x_expr "m/(m+k)*(network_m)/(network_m+network_k) * 100" --x_label "Effective Capacity Ratio (%)" \
--y_min 0,0,0,0 --y_max 14,14,40,1.4 --y_interval 2,2,5,0.2 \
--titles "(a) Availability, (b) Effective Availability, (c) Total Cost, (d) Cost per GB" \
--output_file "scatter_2tier_perf_active_standby.png" \
$result_file_4

python3 ./plot_scatter.py \
--filter_expr "config_file.str.contains('2tier-singleport.json')&((k>0&network_k==0)|(k==0&network_k>0)|((network_k>0)&(k>=1)))" \
--legend "EC Policy" \
--y_label "Nines, Nines, M$, $/GB" --y_expr "avail_nines,credit_avail_nines,total_cost_for_10_years/1000000,cost_per_gb" \
--x_expr "m/(m+k)*(network_m)/(network_m+network_k) * 100" --x_label "Effective Capacity Ratio (%)" \
--y_min 0,0,0,0 --y_max 14,14,40,1.4 --y_interval 2,2,5,0.2 \
--titles "(a) Availability, (b) Effective Availability, (c) Total Cost, (d) Cost per GB" \
--output_file "scatter_2tier_perf_active_standby_single_port.png" \
$result_file_4


python3 ./plot_grouped_bar.py \
--x_label "Host DWPD" --x_expr "dwpd" \
--y_label "Nines" --y_expr "credit_avail_nines" \
--z_expr "cached_ssds" --legend "TLC" \
--y_min 0 --y_max 14 --y_interval 2 \
--legend_location "upper left" \
--legend_ncol 2 \
--output_file "avail_nines_per_cache_ssds.png" \
$result_file_5

python3 ./plot_grouped_bar.py \
--x_label "Host DWPD" --x_expr "dwpd" \
--y_label "Total cost for 10 years(M$)" --y_expr "total_cost_for_10_years/1000000" \
--z_expr "cached_ssds" --legend "TLC" \
--y_min 0 --y_max 100 --y_interval 20 \
--legend_location "upper left" \
--legend_ncol 2 \
--output_file "total_cost_per_cache_ssds.png" \
$result_file_5


python3 ./plot_grouped_bar.py \
--x_label "Host DWPD" --x_expr "dwpd" \
--y_label "$/GB" --y_expr "cost_per_gb" \
--z_expr "cached_ssds" --legend "TLC" \
--y_min 0 --y_max 2 --y_interval 0.2 \
--legend_location "upper left" \
--output_file "cost_per_gb_per_cache_ssds.png" \
$result_file_5



python3 ./plot_sub_graphs.py \
--xlabel "TLCs" --x_expr "cached_ssds" \
--y_label "Initial(HW),Initial(TLC),Initial(QLC),Repair(HW),Repair(TLC),Repair(QLC),Penalty Cost" \
--y_expr "(initial_cost-uncached_initial_cost-cached_initial_cost)/1000000,cached_initial_cost/1000000,uncached_initial_cost/1000000,(repair_cost_for_10_years-cached_ssd_repair_cost_for_10_years-uncached_ssd_repair_cost_for_10_years)/1000000,cached_ssd_repair_cost_for_10_years/1000000,uncached_ssd_repair_cost_for_10_years/1000000,down_cost_for_10_years/1000000" \
--title1 "(a) \$DWPD_{host}\$=0.02" \
--title2 "(b) \$DWPD_{host}\$=0.2" \
--title3 "(c) \$DWPD_{host}\$=0.66" \
--title4 "(d) \$DWPD_{host}\$=2" \
--filter_expr1 "k==4&dwpd==0.02&cached_ssds!=12&cached_ssds!=10" \
--filter_expr2 "k==4&dwpd==0.2&cached_ssds!=12&cached_ssds!=10" \
--filter_expr3 "k==4&dwpd==0.66&cached_ssds!=12&cached_ssds!=10" \
--filter_expr4 "k==4&dwpd==2&cached_ssds!=12&cached_ssds!=10" \
--y_total_label "M$" \
--y_min "0,0,0,0" --y_max "20,20,60,100" --y_interval "5,5,10,20" \
--legend_location "upper left" \
--z_col 4 \
--output_file "cost_distribution_per_cache_ssds.png" \
$result_file_5


python3 ./plot_grouped_bar.py \
--x_label "Parity Count" --x_expr "k" \
--y_label "Nines" --y_expr "credit_avail_nines" \
--z_expr "(('2_' if 'singleport' in config_file else '1_') + ('3tier' if '3tier' in config_file else '2tier'))" \
--legend_labels "2-tier dual,3-tier dual,2-tier single,3-tier single" \
--y_min 0 --y_max 14 --y_interval 2 \
--legend_location "upper left" \
--output_file "avail_nines_per_tiers.png" \
$result_file_6



python3 ./plot_grouped_bar.py \
--x_label "Parity Count" --x_expr "k" \
--y_label "Total cost for 10 years(M$)" --y_expr "total_cost_for_10_years/1000000" \
--z_expr "(('2_' if 'singleport' in config_file else '1_') + ('3tier' if '3tier' in config_file else '2tier'))" \
--legend_labels "2-tier dual,3-tier dual,2-tier single,3-tier single" \
--y_min 0 --y_max 100 --y_interval 20 \
--legend_location "upper left" \
--legend_ncol 2 \
--output_file "total_cost_per_tiers.png" \
$result_file_6

python3 ./plot_stacked_bar.py \
--x_expr "(('single' if 'singleport' in config_file else 'dual') + (' 3-tier' if '3tier' in config_file else ' 2-tier'))" \
--x_label "Combination" \
--y_label "Total cost for 10 years(M$)" \
--filter_expr "k==3" \
--y_expr "(initial_cost-uncached_initial_cost)/1000000,uncached_initial_cost/1000000,(repair_cost_for_10_years-uncached_ssd_repair_cost_for_10_years)/1000000,repair_cost_for_10_years/1000000,down_cost_for_10_years/1000000" \
--legend "Initial(HW),Initial(SSD),Repair(HW),Repair(SSD),Penalty" \
--y_min 0 --y_max 60 --y_interval 10 \
--legend_location "upper left" \
--output_file "cost_distribution_per_tiers.png" \
$result_file_6

python3 ./plot_grouped_bar.py \
--x_label "Parity Count" --x_expr "k" \
--y_label "$/GB" --y_expr "cost_per_gb" \
--z_expr "(('2_' if 'singleport' in config_file else '1_') + ('3tier' if '3tier' in config_file else '2tier'))" \
--legend_labels "2-tier dual,3-tier dual,2-tier single,3-tier single" \
--y_min 0 --y_max 2 --y_interval 0.2 \
--legend_location "upper left" \
--legend_ncol 2 \
--output_file "cost_per_gb_per_tiers.png" \
$result_file_6

python3 ./plot_grouped_bar.py \
--filter_expr "m!=1" \
--x_label "Stripe Length" --x_expr "m+k" \
--y_label "Nines" --y_expr "credit_avail_nines" \
--z_expr "k" --legend "K" \
--y_min 0 --y_max 14 --y_interval 2 \
--legend_location "upper left" \
--output_file "avail_nines_per_stripe.png" \
$result_file_7

python3 ./plot_grouped_bar.py \
--filter_expr "m!=1" \
--x_label "Stripe Length" --x_expr "m+k" \
--y_label "Total cost for 10 years(M$)" --y_expr "total_cost_for_10_years/1000000" \
--z_expr "k" --legend "K" \
--y_min 0 --y_max 100 --y_interval 20 \
--legend_location "upper left" \
--output_file "total_cost_per_stripe.png" \
$result_file_7 

python3 ./plot_grouped_bar.py \
--filter_expr "m!=1" \
--x_label "Stripe Length" --x_expr "m+k" \
--y_label "$/GB" --y_expr "cost_per_gb" \
--z_expr "k" --legend "K" \
--y_min 0 --y_max 1.4 --y_interval 0.2 \
--legend_location "upper left" \
--output_file "cost_per_gb_per_stripe.png" \
$result_file_7

python3 ./plot_grouped_bar.py \
--filter_expr "m==1&k==0" \
--x_label "Stripe Length" --x_expr "network_m+network_k" \
--y_label "Nines" --y_expr "credit_avail_nines" \
--z_expr "network_k" --legend "K" \
--y_min 0 --y_max 14 --y_interval 2 \
--legend_location "upper left" \
--output_file "avail_nines_per_stripe_inter.png" \
$result_file_7

python3 ./plot_grouped_bar.py \
--filter_expr "m==1&k==0" \
--x_label "Stripe Length" --x_expr "network_m+network_k" \
--y_label "Total cost for 10 years(M$)" --y_expr "total_cost_for_10_years/1000000" \
--z_expr "network_k" --legend "K" \
--y_min 0 --y_max 100 --y_interval 20 \
--legend_location "upper left" \
--output_file "total_cost_per_stripe_inter.png" \
$result_file_7 

python3 ./plot_grouped_bar.py \
--filter_expr "m==1&k==0" \
--x_label "Stripe Length" --x_expr "network_m+network_k" \
--y_label "$/GB" --y_expr "cost_per_gb" \
--z_expr "network_k" --legend "K" \
--y_min 0 --y_max 1.4 --y_interval 0.2 \
--legend_location "upper left" \
--output_file "cost_per_gb_per_stripe_inter.png" \
$result_file_7