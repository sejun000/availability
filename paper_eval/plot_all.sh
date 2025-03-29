#!/bin/bash
result_file_1="../analysis_local_redundancy_group_20250329_085801.txt"
result_file_2="../analysis_capacity_20250329_210738.txt"

python3 ./plot_sub_graphs.py $result_file_1 \
--xlabel "Stripe Size (M + K)" --x_expr "m+k" \
--y1_label "Availability (Nines)" --y1_expr "avail_nines" \
--y2_label "Performance-Aware Availability (Nines)" \
--y1_title "(a) Availability Nines" \
--y2_title "(b) Performance-Aware Availability Nines" \
--y2_expr "credit_avail_nines" \
--z_expr "k" --legend "K" --y1_min 0 --y1_max 10 --y1_interval 1 \
--y2_min 0 --y2_max 10 --y2_interval 1

python3 ./plot_sub_graphs_2.py $result_file_1  \
--x1_label "Stripe Size (M + K)" --x1_expr "m+k" \
--x2_label "Party Size (K)" --x2_expr "k" \
--z_expr "k" --legend "K" \
--y1_label "Total Cost For 10 Years (k$)" --y1_expr "total_cost_for_10_years/1000" \
--y2_label "Initial Cost,Repair Cost,Penalty Cost" \
--y1_title "(a) Total Cost versus Erasure Coding Stripe Size and Parity Count" \
--y2_title "(b) Distribution of Total Costs for Stripe Size 48" \
--filter2_expr "m+k==48" \
--y2_total_label "Total Cost For 10 Years (k$)" \
--y2_expr "initial_cost/1000,repair_cost_for_10_years/1000,down_cost_for_10_years/1000" \
--y1_min 0 --y1_max 6000 --y1_interval 1000 \
--y2_min 0 --y2_max 6000 --y2_interval 1000 \
--legend_location1 "upper left" \
--legend_location2 "upper left"

python3 ./plot_grouped_bar.py $result_file_1 \
--xlabel "Stripe Size (M + K)" --x_expr "m+k" \
--ylabel "Cost Per Gigabyte ($/GB)" --y_expr "cost_per_gb" \
--z_expr "k" --legend "K"

python3 ./plot_sub_graphs.py $result_file_2 \
--xlabel "Capacity (TeraBytes)" --x_expr "capacity/1000000000000" \
--filter_expr "m+k==48" \
--y1_label "Performance-Aware Availability (Nines)" --y1_expr "credit_avail_nines" \
--y2_label "Average Reconstruction Time (Hours)" \
--y2_expr "avg_time_for_rebuilding" \
--z_expr "k" --legend "K"

python3 ./plot_sub_graphs_2.py $result_file_2  \
--x1_label "Capacity (TeraBytes)" --x1_expr "capacity/1000000000000" \
--x2_label "Capacity (TeraBytes)" --x2_expr "capacity/1000000000000" \
--z_expr "k" --legend "K" \
--y1_label "Total Cost For 10 Years (k$)" --y1_expr "total_cost_for_10_years/1000" \
--y2_label "Initial Cost(HW),Initial Cost(SSD),Repair Cost,Penalty Cost" \
--filter2_expr "k==3" \
--y2_total_label "Total Cost For 10 Years (k$)" \
--y2_expr "(initial_cost-uncached_initial_cost)/1000,uncached_initial_cost/1000,repair_cost_for_10_years/1000,down_cost_for_10_years/1000" \
--y1_min 0 --y1_max 20000 --y1_interval 2000 \
--y2_min 0 --y2_max 20000 --y2_interval 2000 \
--legend_location1 "upper left" \
--legend_location2 "upper left"


python3 ./plot_grouped_bar.py $result_file_2 \
--xlabel "Capacity (TeraBytes)" --x_expr "capacity/1000000000000" \
--ylabel "Cost Per Gigabyte ($/GB)" --y_expr "cost_per_gb" \
--z_expr "k" --legend "K" \
--y_min 0 --y_max 0.5 --y_interval 0.05 \
--legend_location "upper left"