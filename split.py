import numpy as np

def simulate_path(threshold, split_factor):
    path_value = 0
    paths = [path_value]
    
    while len(paths) > 0:
        new_paths = []
        for path in paths:
            step = np.random.rand()  # 0과 1 사이의 난수를 생성
            new_value = path + step
            if new_value >= threshold:
                return True  # 희귀 사건 발생
            new_paths.extend([new_value] * split_factor)  # 경로 분할
        paths = new_paths
    
    return False  # 희귀 사건 미발생

def estimate_rare_event_probability(threshold, split_factor, num_simulations):
    successes = 0
    for _ in range(num_simulations):
        if simulate_path(threshold, split_factor):
            successes += 1
    return successes / num_simulations

# 설정
threshold = 10
split_factor = 2
num_simulations = 10000

# 희귀 사건 발생 확률 추정
probability = estimate_rare_event_probability(threshold, split_factor, num_simulations)
print(f"희귀 사건 발생 확률: {probability}")