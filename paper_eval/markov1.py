import numpy as np

def build_generator_matrix(N, p, lambd, mu):
    """
    N : 전체 디스크 수
    p : 패리티 디스크 수
    lambd : 디스크 고장률 (1/MTTF)
    mu : 재구축률 (1/MTTR)

    상태: 0, 1, ..., p   (총 p+1개)
    여기서는 "p+1" 상태(흡수, 데이터 손실)를 행렬에 넣지 않고,
    p 상태에서 흡수로 빠져나가는 전이율을 대각원소에만 반영한다.
    """
    size = p + 1  # 0~p
    Q = np.zeros((size, size))

    for i in range(size):
        # i -> i+1 고장 발생
        if i < p:
            Q[i, i+1] = (N - i) * lambd

        # i -> i-1 재구축
        if i > 0:
            Q[i, i-1] = i * mu

    # 마지막 상태 p에서 흡수(= p+1)로 빠져나가는 전이율도 고려
    # Q[p,p+1] 대신 "Q[p,p]" 대각에서 빼 주어야 함
    outflow_absorb = (N - p) * lambd  # p번째 -> p+1(흡수) 전이
    Q[p,p] -= outflow_absorb

    # 이제 각 행의 대각원소 설정
    for i in range(size):
        Q[i, i] = -np.sum(Q[i, :])

    return Q

def mttdl_from_ctmc(N, p, lambd, mu):
    """
    Q 행렬을 만든 뒤, 0~p 상태가 모두 일시적 상태이므로
    F = (-Q)^{-1} 를 통해 MTTDL 계산.
    p+1(흡수)는 행렬에 직접 넣지 않음(방법 B).
    """
    Q = build_generator_matrix(N, p, lambd, mu)
    F = np.linalg.inv(-Q)
    # MTTDL = F[0,:]의 합
    return np.sum(F[0, :])

def main():
    # 주어진 파라미터
    DWPD = 0.26
    qlc_mttf_hours = 5 * 365 * 24 * 0.26 / 0.1  # 113,880시간
    lambd = 1.0 / qlc_mttf_hours
    N = 48
    rebuild_speed_gbps = 0.8  # 4 GB/s의 20% = 0.8

    capacities_tb = [8, 16, 32, 64, 128, 256]

    results = []

    for cap in capacities_tb:
        gb = cap * 1000
        rebuild_time_sec = gb / rebuild_speed_gbps
        rebuild_time_hour = rebuild_time_sec / 3600.0
        mu = 1.0 / rebuild_time_hour

        for p in [1, 2, 3, 4]:
            mttdl_hours = mttdl_from_ctmc(N, p, lambd, mu)
            mttdl_years = mttdl_hours / 8760.0
            results.append((cap, p, mttdl_hours, mttdl_years))

    print("Capacity(TB) | Parity | MTTDL(hours)      | MTTDL(years)")
    print("--------------------------------------------------------")
    for (cap, p, h, y) in results:
        print(f"{cap:12} | {p:6} | {h:16.4f} | {y:12.4f}")

if __name__ == "__main__":
    main()
