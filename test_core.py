from graph_structure import GraphStructure
from core import _calculate_connected_ssd
from core import generate_first_failure_events
from core import count_failed_node
from core import push_failed_event
from core import pop_event
from core import push_repair_event
import heapq
import random

def run_test(test_func):
    try:
        test_func()
        # print green if no assertion error
        print (f"\033[92m{test_func.__name__}: PASS\033[0m")
    except AssertionError as e:
        # print red if assertion error
        print (f"\033[91m{test_func.__name__}: FAIL - {e}\033[0m")

def test_calculate_connected_ssd_case_all_disconnected():
    graph_structure = GraphStructure()

    # Set up your graph and enclosures
    graph_structure.G.add_edge("start1", "ssd1")
    graph_structure.G.add_edge("start2", "ssd2")
    graph_structure.G.add_edge("switch0", "start1")
    graph_structure.G.add_edge("switch0", "start2")
    graph_structure.enclosures = {
        "Enclosure1": ["module1", "module2"]
    }

    # Input parameters
    key = "test_key"
    failed_node = ["switch0"]
    lowest_common_level_module = "switch"
    disconnected_ssd_table = {}
    lowest_common_level_module_connected_table = {}

    # Call the function
    _calculate_connected_ssd(
        graph_structure_origin=graph_structure, 
        key=key, 
        failed_node=failed_node, 
        lowest_common_level_module=lowest_common_level_module, 
        disconnected_ssd_table=disconnected_ssd_table, 
        lowest_common_level_module_connected_table=lowest_common_level_module_connected_table
    )

    # Assert expected outputs
    assert len(disconnected_ssd_table) >= 1, "Key should exist in disconnected_ssd_table"
    assert lowest_common_level_module_connected_table[key] == False, "lowest_common_level_module_connected_table shall be false"
    print(f"Disconnected SSD Table: {disconnected_ssd_table}")
    print(f"Lowest Common Level Module Connected Table: {lowest_common_level_module_connected_table}")

def test_calculate_connected_ssd_case_parital_connected():
    graph_structure = GraphStructure()

    # Set up your graph and enclosures
    graph_structure.G.add_edge("start1", "ssd1")
    graph_structure.G.add_edge("start2", "ssd2")
    graph_structure.G.add_edge("switch0", "start1")
    graph_structure.G.add_edge("switch0", "start2")
    graph_structure.enclosures = {
        "Enclosure1": ["module1", "module2"]
    }

    # Input parameters
    key = "test_key"
    failed_node = ["start1"]
    lowest_common_level_module = "switch"
    disconnected_ssd_table = {}
    lowest_common_level_module_connected_table = {}

    # Call the function
    _calculate_connected_ssd(
        graph_structure_origin=graph_structure, 
        key=key, 
        failed_node=failed_node, 
        lowest_common_level_module=lowest_common_level_module, 
        disconnected_ssd_table=disconnected_ssd_table, 
        lowest_common_level_module_connected_table=lowest_common_level_module_connected_table
    )

    # Assert expected outputs
    assert len(disconnected_ssd_table) >= 1, "Key should exist in disconnected_ssd_table"
    assert lowest_common_level_module_connected_table[key] == True, "lowest_common_level_module_connected_table shall be true"
    print(f"Disconnected SSD Table: {disconnected_ssd_table}")
    print(f"Lowest Common Level Module Connected Table: {lowest_common_level_module_connected_table}")


def test_generate_first_failure_events():
    # Mock graph structure
    graph_structure = GraphStructure()

    # Adding nodes to the graph
    graph_structure.G.add_nodes_from(["ssd0", "switch0", "ssd1", "switch1"])

    # Matched modules
    matched_module = {
        "ssd0": "ssd",
        "ssd1": "ssd",
        "switch0": "switch",
        "switch1": "switch",
        "Enclosure0": "Enclosure",
        "Enclosure1": "Enclosure"
    }

    # Matched enclosures
    matched_enclosure = {
        "ssd0": "Enclosure0",  # this node should be skipped
        "switch0": "Enclosure0",
        "ssd1": "Enclosure1",
        "switch1": "Enclosure1"
    }

    # Mock enclosures in the graph structure
    graph_structure.enclosures = {
        "Enclosure0": ["ssd0", "switch0"],
        "Enclosure1": ["switch1"]
    }

    # Setting MTTF and MTR for modules
    graph_structure.mttfs = {
        "ssd": 1000,  # Mean Time To Failure
        "switch": 2000,
        "Enclosure": 3000
    }
    graph_structure.mtrs = {
        "ssd": 50,  # Mean Time To Repair
        "switch": 100,
        "Enclosure": 150
    }

    # Mock random failure time generation
    random.seed(0)  # Seed to make the test deterministic

    # Call the function to generate failure events
    events = generate_first_failure_events(graph_structure, matched_module, matched_enclosure)

    # Check if the events generated match the expected output
    assert len(events) == len(matched_module), "Number of events doesn't match"
    
    for event in events:
        (time, action, node, current_time) = event
        assert action == "fail", f"Event {event} shall be failed event"
        assert node in matched_module, f"Event {event} not in expected events"

def test_count_failed_node():
    # Initial state
    failed_node = {}
    current_failures_except_ssd = {
        "group1": 0,
        "ssd_group2": 0
    }
    current_failures = {
        "group1": 0,
        "ssd_group2": 0
    }
    
    # Mapping of nodes to their groups
    matched_group = {
        "node1": "group1",
        "ssd2": "ssd_group2",
        "ssd1": "ssd_group2"  # This should be treated as an SSD node
    }

    # Test 1: Fail "node1" (non-SSD node in group1)
    count_failed_node('fail', 'node1', failed_node, current_failures_except_ssd, current_failures, matched_group)
    
    assert failed_node["node1"] == 1, "node1 should have failed count of 1"
    assert current_failures["group1"] == 1, "group1 should have 1 current failure"
    assert current_failures_except_ssd["group1"] == 1, "group1 (non-SSD) should have 1 failure"

    # Test 2: Fail "leaf_node" (SSD node in group2)
    count_failed_node('fail', 'ssd1', failed_node, current_failures_except_ssd, current_failures, matched_group)
    
    assert failed_node["ssd1"] == 1, "ssd1 should have failed count of 1"
    assert current_failures["ssd_group2"] == 1, "ssd_group2 should have 1 current failure"
    print (current_failures_except_ssd)
    assert current_failures_except_ssd["ssd_group2"] == 0, "ssd_group2 (SSD) should have 0 non-SSD failures"

    # Test 3: Repair "node1" (non-SSD node in group1)
    count_failed_node('repair', 'node1', failed_node, current_failures_except_ssd, current_failures, matched_group)
    
    assert "node1" not in failed_node, "node1 should be removed from failed_node after repair"
    assert current_failures["group1"] == 0, "group1 should have 0 current failures after repair"
    assert current_failures_except_ssd["group1"] == 0, "group1 (non-SSD) should have 0 failures after repair"

    # Test 4: Fail "leaf_node" again (SSD node in group2)
    count_failed_node('fail', 'ssd1', failed_node, current_failures_except_ssd, current_failures, matched_group)
    assert failed_node["ssd1"] == 2, "ssd1 should have failed count of 1"
    assert current_failures["ssd_group2"] == 1, "ssd_group2 should have 1 current failure"
    print (current_failures_except_ssd)
    assert current_failures_except_ssd["ssd_group2"] == 0, "ssd_group2 (SSD) should have 0 non-SSD failures"

    # Test 5: Repair "leaf_node", current ssd1's failure count should be 1
    count_failed_node('repair', 'ssd1', failed_node, current_failures_except_ssd, current_failures, matched_group)
    
    assert "ssd1" in failed_node, "ssd1 shall be in failed_node even after repair"
    assert current_failures["ssd_group2"] == 1, "ssd_group2 should have 0 current failures after repair"
    assert current_failures_except_ssd["ssd_group2"] == 0, "ssd_group2 (SSD) should have 0 non-SSD failures after repair"

    # Test 4: Repair "leaf_node" (SSD node in group2)
    count_failed_node('repair', 'ssd1', failed_node, current_failures_except_ssd, current_failures, matched_group)
    
    assert "ssd1" not in failed_node, "ssd1 should be removed from failed_node after repair"
    assert current_failures["ssd_group2"] == 0, "ssd_group2 should have 0 current failures after repair"
    assert current_failures_except_ssd["ssd_group2"] == 0, "ssd_group2 (SSD) should have 0 non-SSD failures after repair"

def test_push_failed_event():
    # Mocking graph structure
    graph_structure = GraphStructure()
    graph_structure.mttfs = {
        "module1": 1000,  # MTTF for regular module
        "module2": 2000,  # MTTF for SSD module
    }
    
    # Mock matched_module mapping nodes to modules
    matched_module = {
        "node1": "module1",  # Regular module
        "node2": "module2",  # SSD module
    }
    
    # Mock failed_events heap and random seed for deterministic behavior
    failed_events = []
    random.seed(0)  # To ensure predictable failure time
    
    # Test 1: Push event for a regular node (non-SSD)
    push_failed_event(failed_events, "node1", 0, matched_module, graph_structure)
    
    assert len(failed_events) == 1, "Failed event for node1 not added correctly"
    event_time, event_type, event_node, event_current_time = failed_events[0]
    assert event_type == 'fail', "Event type should be 'fail'"
    assert event_node == "node1", "Event node should be 'node1'"
    assert event_current_time == 0, "Current time should be 0 for node1 event"
    assert event_time > 0, "Failure time should be greater than current time for node1"

    # Test 2: Push event for an SSD node
    failed_events = []  # Reset the heap for a clean test
    push_failed_event(failed_events, "node2", 0, matched_module, graph_structure)
    
    assert len(failed_events) == 1, "Failed event for SSD node2 not added correctly"
    event_time, event_type, event_node, event_current_time = failed_events[0]
    assert event_type == 'fail', "Event type should be 'fail' for SSD node"
    assert event_node == "node2", "Event node should be 'node2'"
    assert event_current_time == 0, "Current time should be 0 for SSD node event"
    assert event_time > 0, "Failure time should be greater than current time for SSD node"

def test_pop_event():
    # Case 1: Only repair_events has events
    events = []
    repair_events = [(5, 'repair', 'node1'), (10, 'repair', 'node2')]
    heapq.heapify(repair_events)
    
    popped = pop_event(events, repair_events)
    assert popped == (5, 'repair', 'node1'), f"Expected (5, 'repair', 'node1'), but got {popped}"

    popped = pop_event(events, repair_events)
    assert popped == (10, 'repair', 'node2'), f"Expected (10, 'repair', 'node2'), but got {popped}"

    # Case 2: Only events has events
    events = [(3, 'fail', 'node3'), (8, 'fail', 'node4')]
    repair_events = []
    heapq.heapify(events)
    
    popped = pop_event(events, repair_events)
    assert popped == (3, 'fail', 'node3'), f"Expected (3, 'fail', 'node3'), but got {popped}"

    popped = pop_event(events, repair_events)
    assert popped == (8, 'fail', 'node4'), f"Expected (8, 'fail', 'node4'), but got {popped}"

    # Case 3: Both events and repair_events have events, repair_events first
    events = [(7, 'fail', 'node5')]
    repair_events = [(6, 'repair', 'node6')]
    heapq.heapify(events)
    heapq.heapify(repair_events)

    popped = pop_event(events, repair_events)
    assert popped == (6, 'repair', 'node6'), f"Expected (6, 'repair', 'node6'), but got {popped}"

    popped = pop_event(events, repair_events)
    assert popped == (7, 'fail', 'node5'), f"Expected (7, 'fail', 'node5'), but got {popped}"

    # Case 4: Both events and repair_events have events, events first
    events = [(4, 'fail', 'node7')]
    repair_events = [(5, 'repair', 'node8')]
    heapq.heapify(events)
    heapq.heapify(repair_events)

    popped = pop_event(events, repair_events)
    assert popped == (4, 'fail', 'node7'), f"Expected (4, 'fail', 'node7'), but got {popped}"

    popped = pop_event(events, repair_events)
    assert popped == (5, 'repair', 'node8'), f"Expected (5, 'repair', 'node8'), but got {popped}"

def test_push_repair_event():
    # Mocking the graph structure
    graph_structure = GraphStructure()
    graph_structure.mtrs = {
        "module1": 1000,  # MTR for a regular module
        "ssd": 2000,  # MTR for an SSD module
    }

    # Matched modules and groups
    matched_module = {
        "node1": "module1",  # Regular module
        "ssd2": "ssd",  # SSD module
    }
    matched_group = {
        "node1": "group1",
        "ssd2": "ssd_group",
    }

    # Mock rebuild tables and SSD state
    max_flow_for_rebuild_table = {
        "key1": {"ssd_group": 500}  # Rebuild bandwidth for group2
    }
    max_flow_for_network_rebuild_table = {
    }
    SSDs = {}
    #SSDs["node1"] = {0, 0, 0}
    #SSDs["node2"] = {0, 0, 0}
    disconnected_ssds = set()
    failed_node = {}

    # Mock repair events heap and time period
    repair_events = []
    time_period = 1000  # Example time period

    # Test 1: Regular node repair event
    push_repair_event(repair_events, "node1", 0, matched_module, matched_group, graph_structure, "key1", SSDs, disconnected_ssds, failed_node, time_period, max_flow_for_rebuild_table, max_flow_for_network_rebuild_table)
    assert len(repair_events) == 1, "Failed to add repair event for regular node"
    repair_time, event_type, event_node, event_time = repair_events[0]
    assert event_type == 'repair', "Event type should be 'repair'"
    assert event_node == 'node1', "Event node should be 'node1'"
    assert repair_time > 0, "Repair time should be greater than 0"

    # Test 2: SSD node repair event with rebuild bandwidth
    repair_events = []  # Reset the heap
    push_repair_event(repair_events, "ssd2", 0, matched_module, matched_group, graph_structure, "key1", SSDs, disconnected_ssds, failed_node, time_period, max_flow_for_rebuild_table, max_flow_for_network_rebuild_table)
    assert len(repair_events) == 1, "Failed to add repair event for SSD node"
    repair_time, event_type, event_node, event_time = repair_events[0]
    assert event_type == 'repair', "Event type should be 'repair'"
    assert event_node == 'ssd2', "Event node should be 'ssd2'"
    assert repair_time > 0, "Repair time should be greater than 0"
    repair_time_test2 = repair_time
    assert event_time == 0, "Event time should be the current time"
    assert "ssd2" in SSDs, "SSDs dictionary should contain ssd2"

    # Test 3: Disconnected SSD node should not add event
    disconnected_ssds.add("ssd2")
    repair_events = []  # Reset the heap
    push_repair_event(repair_events, "ssd2", 0, matched_module, matched_group, graph_structure, "key1", SSDs, disconnected_ssds, failed_node, time_period, max_flow_for_rebuild_table, max_flow_for_network_rebuild_table)
    assert len(repair_events) == 0, "Disconnected SSD should not add a repair event"

    # Test 4: Regular node repair event with network rebuild bandwidth
    repair_events = []  # Reset the heap
    push_repair_event(repair_events, "node1", 0, matched_module, matched_group, graph_structure, "key1", SSDs, disconnected_ssds, failed_node, time_period, max_flow_for_rebuild_table, max_flow_for_network_rebuild_table)
    assert len(repair_events) == 1, "Failed to add repair event for regular node"
    repair_time, event_type, event_node, event_time = repair_events[0]
    assert event_type == 'repair', "Event type should be 'repair'"
    assert event_node == 'node1', "Event node should be 'node1'"
    assert repair_time > 0, "Repair time should be greater than 0"
    assert event_time == 0, "Event time should be the current time"

    # Test 5: Regular node repair event with network rebuild bandwidth
    repair_events = []  # Reset the heap
    max_flow_for_rebuild_table = {
        "key1": {}
    }
    max_flow_for_network_rebuild_table = {
        "key1": {"ssd_group": 1500}  # Rebuild bandwidth for group2
    }
    disconnected_ssds = []
    push_repair_event(repair_events, "ssd2", 0, matched_module, matched_group, graph_structure, "key1", SSDs, disconnected_ssds, failed_node, time_period, max_flow_for_rebuild_table, max_flow_for_network_rebuild_table)
    assert len(repair_events) == 1, "Failed to add repair event for SSD node"
    repair_time, event_type, event_node, event_time = repair_events[0]
    assert event_type == 'repair', "Event type should be 'repair'"
    assert event_node == 'ssd2', "Event node should be 'ssd2'"
    assert repair_time > 0, "Repair time should be greater than 0"
    assert event_time == 0, "Event time should be the current time"
    assert "ssd2" in SSDs, "SSDs dictionary should contain ssd2"
    # 3 times the repair time (1500 vs 500) should be the same (2% tolerance)
    assert abs((repair_time_test2 - repair_time * 3) / (repair_time * 3)) <= 0.02, "Repair time should be the same as the previous repair time"

# Run the test
run_test(test_calculate_connected_ssd_case_all_disconnected)
run_test(test_calculate_connected_ssd_case_parital_connected)
run_test(test_generate_first_failure_events)
run_test(test_count_failed_node)
run_test(test_push_failed_event)
run_test(test_pop_event)
run_test(test_push_repair_event)
