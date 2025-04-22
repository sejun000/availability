import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
rebuild_bandwidth_MBps = 400  # 2GB/s * 20%
capacities_TB = [8, 16, 32, 64, 128, 256]
N_disks = 48  # total number of disks in the system
parities = [1, 2, 3, 4]  # number of parity disks (i.e., k = N - data disks)

# Disk MTTF calculation (in hours)
MTTF_hours = 24 * 365 * 5 * 0.26 / 0.2  # 51900 hours

# Function to compute MTTR from capacity
def compute_mttr_hours(cap_tb):
    return (cap_tb * 1024 * 1024) / rebuild_bandwidth_MBps / 3600  # hours

# Function to compute MTTDL using CTMC approximation
def compute_mttdl(N, k, MTTF, MTTR):
    """
    N: total number of disks in a stripe
    k: number of parity disks (tolerates up to k failures)
    MTTF: mean time to failure per disk (hours)
    MTTR: mean time to repair per disk (hours)
    Returns: MTTDL in hours
    """
    failure_rate = 1 / MTTF
    repair_rate = 1 / MTTR

    num_states = k + 2  # 0 to k failures + absorbing state
    Q = np.zeros((num_states, num_states))

    for i in range(k + 1):
        fail_lambda = (N - i) * failure_rate
        repair_mu = i * repair_rate if i > 0 else 0

        if i < k + 1:
            Q[i, i + 1] = fail_lambda
        if i > 0:
            Q[i, i - 1] = repair_mu
        Q[i, i] = -np.sum(Q[i])  # row sum = 0

    # Compute expected time to absorption (MTTDL)
    Q_transient = Q[:k + 1, :k + 1]
    N_matrix = -inv(Q_transient)
    t = N_matrix @ np.ones((k + 1, 1))
    return t[0, 0]  # time to absorption from state 0

results = []

for cap in capacities_TB:
    MTTR = compute_mttr_hours(cap)
    for p in parities:
        mttdl_hours = compute_mttdl(N_disks, p, MTTF_hours, MTTR)
        durability = np.exp(-8760 / mttdl_hours)  # 1-year durability
        results.append({
            "Capacity (TB)": cap,
            "Parity Disks": p,
            "MTTDL (years)": mttdl_hours / (24 * 365),
            "1-Year Durability (%)": durability * 100
        })

df = pd.DataFrame(results)
print(
    df.to_string(
        formatters={
            "1-Year Durability (%)": "{:.20f}".format
        }
    )
)
#import ace_tools as tools; tools.display_dataframe_to_user(name="MTTDL and Durability Results", dataframe=df)
