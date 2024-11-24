SSD_module_name = "SSD"
network_module_name = "network"


def get_network_group_name(network_index):
    return f"{network_module_name}_group_{network_index}"

def get_ssd_group_index_from_network_group_name(network_group_name):
    return int(network_group_name[len(f"{network_module_name}_group"):])

def get_ssd_group_name(ssd_group_index):
    return f"{SSD_module_name}_group_{ssd_group_index}"

def get_ssd_name(ssd_index):
    return f"{SSD_module_name}{ssd_index}"

def get_ssd_index(ssd_name):
    return int(ssd_name[len(SSD_module_name):])