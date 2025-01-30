
# read flow_cdf.txt

read_file = open("flow_cdf.txt", "r")
bandwidths = []
time_intervals = []
for line in read_file:
    bandwidth = float(line.split()[0])
    time_interval = float(line.split()[1])
    bandwidths.append(bandwidth)
    time_intervals.append(time_interval)
total_time_interval = 0.0
total_bandwidth = 0.0
for i in range(0, len(time_intervals)):
    #print ("time_interval: ", time_intervals[i], "bandwidth: ", bandwidths[i])
    total_time_interval += time_intervals[i]
    total_bandwidth += bandwidths[i] * time_intervals[i]
# check 99.9%, 99.99%, 99.999% percentile
percentiles = [0.99, 0.999, 0.9999, 0.99999, 0.999999]
check_percentiles = {}
cumulative_time_interval = 0
cumulative_bandwidth = 0
for i in range(0, len(time_intervals)):
    j = 0
    for percentile in percentiles:
        if j in check_percentiles:
            j += 1
            continue
        if cumulative_time_interval + time_intervals[i] >= percentile * total_time_interval:
            time_diff = percentile * total_time_interval - cumulative_time_interval 
            bandwidth_diff = time_diff * bandwidths[i]
            percentile_bandwidth = (total_bandwidth - (cumulative_bandwidth + bandwidth_diff)) / (total_time_interval - (cumulative_time_interval + time_diff))
            print("percentile: ", percentile, "bandwidth: ", percentile_bandwidth, "time_interval: ", total_bandwidth, total_time_interval, time_diff, i)
            check_percentiles[j] = 1
        j += 1
    #print ("why", cumulative_time_interval, i, time_intervals[i], total_time_interval)
    cumulative_bandwidth += bandwidths[i] * time_intervals[i]
    cumulative_time_interval += time_intervals[i]
    
        
