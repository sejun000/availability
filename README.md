# Availability Tool User Guide (2025.03.28)

This document provides guidance on using the Availability Tool, including instructions for installation, execution, and command-line parameter configuration.

---

## Installation and Execution

### Prerequisites

Install required packages by running the following shell script:

```
./prerequisite.sh
```

### Running the Simulation

Execute the command as follows:

```
python3 ./new_core.py --simulation [additional command parameters]
```

---

## Command Parameter Descriptions

The following table details the available command parameters and their default values:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--total_ssds` | int | 48 | Total number of SSDs per enclosure |
| `--m` | int | 15 | Number of data chunks for intra erasure coding |
| `--k` | int | 1 | Number of parity chunks for intra erasure coding |
| `--l` | int | 0 | Number of spare chunks (0 for clustered parity, >0 for declustered parity) |
| `--cached_ssds` | int | 0 | Number of cache-tier SSDs per enclosure |
| `--inter_replicas` | int | 0 | Number of network replicas |
| `--intra_replicas` | int | 0 | Number of local replicas (only for cache SSDs) |
| `--cached_write_ratio` | float | 0 | Ratio of writes absorbed by cache SSD (remaining writes go to QLC), value between 0~1 |
| `--network_m` | int | 6 | Number of network data chunks |
| `--network_k` | int | 0 | Number of network parity chunks |
| `--network_l` | int | 0 | Number of network spare chunks |
| `--capacity` | int | 64000000000000 | SSD capacity (bytes) |
| `--qlc` | flag | False | Use QLC SSDs (default is TLC) |
| `--dwpd` | float | 1 | Drive Writes Per Day (DWPD) of the host |
| `--guaranteed_years` | int | 5 | SSD warranty period (years) |
| `--config_file` | str | '2tier.json' | Path to the graph configuration file |
| `--output_file` | str | 'results.txt' | Output file path (appended results) |
| `--nprocs` | int | 40 | Number of parallel processes for simulation |

---

## Examples

- **Intra erasure coding test (m=44, k=4) with host DWPD=0.01**

```
python3 new_core.py --output_file analysis_local_redundancy_group_20250328_071038.txt --m 44 --k 4 --capacity 128_000_000_000_000 --config_file 2tier.json --simulation --total_ssds 48 --dwpd 0.01 --qlc
```

- **Inter erasure coding tests with host DWPD=0.1**

Conduct initial test with network_m=8, network_k=0, then use results to test various parity configurations:

```
python3 new_core.py --output_file analysis_network_redundancy_group_20250328_071743.txt --m 1 --k 0 --network_m 8 --network_k 0 --capacity 128_000_000_000_000 --config_file 2tier.json --simulation --total_ssds 48 --dwpd 0.1 --qlc
# Continue with network_m=7,6 and network_k=1,2 accordingly
```

- **Multi-level erasure coding tests with host DWPD=0.1**

Similar to inter-level tests, modify only m and k parameters:

```
python3 new_core.py --output_file analysis_network_redundancy_group_20250328_071743.txt --m 7 --k 1 --network_m 8 --network_k 0 --capacity 128_000_000_000_000 --config_file 2tier.json --simulation --total_ssds 48 --dwpd 0.1 --qlc
# Continue adjusting network_m and network_k
```

- **Replication usage at inter-level (inter_replicas)**

Instead of inter-enclosure erasure coding, replication is applied:

```
python3 new_core.py --output_file analysis_network_redundancy_group_20250328_074227.txt --m 1 --k 0 --inter_replicas 0 --capacity 128_000_000_000_000 --config_file 2tier.json --simulation --total_ssds 48 --dwpd 0.1 --qlc
# Then test inter_replicas=2
```

- **Storage tiering at intra-level**

Requires intra_replicas, cached_ssds, cached_write_ratio settings:

```
python3 new_core.py --output_file analysis_local_redundancy_group_20250328_075040.txt --m 40 --k 4 --intra_replicas 2 --cached_ssds 4 --cached_write_ratio 0.4437 --capacity 128_000_000_000_000 --config_file 2tier.json --simulation --total_ssds 48 --dwpd 0.033 --qlc
```

---

## Output File Interpretation

Simulation results include:

- availability: uptime ratio during simulation
- effective_availability: average maxflow ratio during simulation
- credit_availability: minimum of uptime ratio and maxflow ratio per event interval
- initial_cost: total initial hardware cost
- repair_cost_for_10 years: hardware replacement costs over 10 years
- down_cost_for_10 years: SLA penalties over 10 years
- total_cost_for_10 years: cumulative cost over 10 years
- cached_ssd_repair_cost_per_year: annual TLC replacement cost
- uncached_ssd_repair_cost_per_year: annual QLC replacement cost

---

## Input JSON File

Defines graph structure, bandwidth, and hardware modules. Comments should be excluded.

Example structure:

- **lowest_common_module**: module defining the boundary for local modules during inter-erasure tests.
- **edges**: defines connections and bandwidth between modules.
- **enclosures**: groups hardware modules.

(Complete JSON example omitted for brevity.)

---

## Graph Visualization

To visualize the JSON-defined graph:

```
python3 ./graph_visualization.py --input_file 2tier.json
```

To save graph visualization to file:

```
python3 ./graph_visualization.py --input_file 2tier.json --output_file 2tier.png
```

