data1 = frozenset({
    'SSD_group_0': frozenset({'failure_count': 1, 'network_failure_count': 0}.items()),
    'SSD_group_1': frozenset({'failure_count': 0, 'network_failure_count': 0}.items()),
    'SSD_group_2': frozenset({'failure_count': 0, 'network_failure_count': 0}.items())
}.items())
data2 = frozenset({
    'SSD_group_0': frozenset({'failure_count': 0, 'network_failure_count': 0}.items()),
    'SSD_group_1': frozenset({'failure_count': 1, 'network_failure_count': 0}.items()),
    'SSD_group_2': frozenset({'failure_count': 0, 'network_failure_count': 0}.items())
}.items())

data3 = frozenset([
    frozenset({'failure_count': 1, 'network_failure_count': 0}.items()),
    frozenset({'failure_count': 1, 'network_failure_count': 2}.items()),
    frozenset({'failure_count': 0, 'network_failure_count': 0}.items())])

data4 = frozenset([
    frozenset({'failure_count': 2, 'network_failure_count': 1}.items()),
    frozenset({'failure_count': 0, 'network_failure_count': 0}.items()),
    frozenset({'failure_count': 1, 'network_failure_count': 0}.items())])


print (data1)
print (data2)
print (data1 == data2)
print (data3 == data4)
