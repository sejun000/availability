SSD_module_name = "SSD"

ssd_info = {}

current_allocated_ssd_group_index = 0
current_allocated_ssd_index = 0

class SSDRedundancyScheme:
    def __init__(self, write_bw, read_bw, read_latency, mttf, cached_write_ratio, cached_write_bw, cached_read_bw, cached_read_latency, cached_mttf, m, k, l, cached_m, cached_k, cached_l, network_m, network_k, network_l, cached_network_m, cached_network_k, cached_network_l, cached_ssds, total_ssds, inter_replicas, intra_replicas):
        self.write_bw = write_bw
        self.read_bw = read_bw
        self.read_latency = read_latency
        self.mttf = mttf
        self.cached_write_ratio = cached_write_ratio
        self.cached_write_bw = cached_write_bw
        self.cached_read_bw = cached_read_bw
        self.cached_read_latency = cached_read_latency
        self.cached_mttf = cached_mttf
        self.m = m
        self.k = k
        self.l = l
        self.cached_m = cached_m
        self.cached_k = cached_k
        self.cached_l = cached_l
        self.network_m = network_m
        self.network_k = network_k
        self.network_l = network_l
        self.cached_network_m = cached_network_m
        self.cached_network_k = cached_network_k
        self.cached_network_l = cached_network_l
        self.cached_ssds = cached_ssds
        self.total_ssds = total_ssds 
        self.ssd_group_size = m + k + l
        self.cached_ssd_group_size = cached_m + cached_k + cached_l   
        self.inter_replicas = inter_replicas
        self.intra_replicas = intra_replicas

    def is_ssd_index_cached(self, ssd_index):
        return ssd_index >= self.total_ssds - self.cached_ssds

    def is_ssd_group_index_cached(self, ssd_group_index):
        return ssd_group_index >= (self.total_ssds - self.cached_ssds) //  self.ssd_group_size
    
    def get_start_ssd_index(self, ssd_group_index):
        if not self.is_ssd_group_index_cached(ssd_group_index):
            return ssd_group_index * self.ssd_group_size
        else:
            return (self.total_ssds - self.cached_ssds) + (ssd_group_index - (self.total_ssds - self.cached_ssds) // self.ssd_group_size) * self.cached_ssd_group_size

    def get_total_group_count(self):
        if self.cached_ssds == 0:
            return self.total_ssds // self.ssd_group_size
        return (self.total_ssds - self.cached_ssds) // self.ssd_group_size + self.cached_ssds // self.cached_ssd_group_size

    def get_ssd_group_index(self, ssd_index):
        if not self.is_ssd_index_cached(ssd_index):
            return ssd_index // self.ssd_group_size
        else:
            cached_ssd_group_start_index = (self.total_ssds - self.cached_ssds) // self.ssd_group_size
            return (ssd_index - (self.total_ssds - self.cached_ssds)) // self.cached_ssd_group_size + cached_ssd_group_start_index

    def get_read_bw(self, cached):
        return self.cached_read_bw if cached else self.read_bw
    def get_write_bw(self, cached):
        return self.cached_write_bw if cached else self.write_bw
    def get_read_latency(self, cached):
        return self.cached_read_latency if cached else self.read_latency
    def get_mttf(self, cached):
        return self.cached_mttf if cached else self.mttf
    def get_m(self, cached):
        return self.cached_m if cached else self.m
    def get_k(self, cached):
        return self.cached_k if cached else self.k
    def get_l(self, cached):
        return self.cached_l if cached else self.l
    def get_network_m(self, cached):
        return self.cached_network_m if cached else self.network_m
    def get_network_k(self, cached):
        return self.cached_network_k if cached else self.network_k
    def get_network_l(self, cached):
        return self.cached_network_l if cached else self.network_l
    def get_total_ssds(self):
        return self.total_ssds
    def get_tiered_ssds(self, cached):
        return self.cached_ssds if cached else self.total_ssds - self.cached_ssds
    def get_cached_prefix(self, cached):
        return "cached_" if cached else ""
    def get_inter_replicas(self, cached):
        return self.inter_replicas if cached else 0
    def get_intra_replicas(self, cached):
        return self.intra_replicas if cached else 0

def get_ssd_group_index_from_group_name(ssd_group_name):
    return int(ssd_group_name[len(SSD_module_name) + 6:])

def get_ssd_group_name(ssd_group_index):
    return f"{SSD_module_name}_group_{ssd_group_index}"

def get_ssd_name(ssd_index):
    return f"{SSD_module_name}{ssd_index}"

def get_ssd_index(ssd_name):
    return int(ssd_name[len(SSD_module_name):])

def is_event_node_ssd(event_node):
    return event_node.startswith(SSD_module_name)