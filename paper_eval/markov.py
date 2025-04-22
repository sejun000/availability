# Parameters
N = 48                                # total number of drives
DWPD = 0.26                           # drive writes per day
# Calculate QLC drive MTTF (in hours) from given formula
QLCMTTF_hours = 5 * 365 * 24 * DWPD / 0.2   # = 113880 hours ≈ 13 years
# Rebuild speed and capacities
rebuild_speed = 0.4  # GB/s (20% of 2 GB/s)
capacities_TB = [8, 16, 32, 64, 128, 256]

import math
for P in [1, 2, 3, 4]:
    print(f"Parity P={P}:")
    for cap in capacities_TB:
        # Compute MTTR in hours for this drive capacity
        capacity_GB = cap * 1000.0  # using 1 TB = 1000 GB for calculation
        MTTR_hours = capacity_GB / rebuild_speed / 3600.0
        print (f"  Capacity {cap} TB -> MTTR ≈ {MTTR_hours:.2f} hours")
        # MTTDL calculation (analytical formula)
        numerator = (QLCMTTF_hours) ** (P + 1)
        # denominator: N * (N-1) * ... * (N-P) * (MTTR^P)
        denom_drives = 1
        for i in range(P + 1):
            denom_drives *= (N - i)
        denominator = denom_drives * (MTTR_hours ** P)
        MTTDL_hours = numerator / denominator
        MTTDL_years = MTTDL_hours / (24 * 365)
        # Annual durability
        one_year_hours = 24 * 365
        durability = math.exp(- one_year_hours / MTTDL_hours)
        # Print results
        print(f"  Capacity {cap} TB -> MTTDL ≈ {MTTDL_years:.2e} years, ",
              f"Durability ≈ {durability*100:.6f}%")
    print()
