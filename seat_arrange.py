import numpy as np
import itertools
import math
import time

def seat_arrange(N, seatPreferenceList, seatCoordinate, M):
    """
    例：
      N: 人数や席数の整数
      seatPreferenceList: 各人の希望配置 (N×N のリスト)
      seatCoordinate: 各席の座標 (N×2 のリスト)
      M: 距離モデル (1 ならガウス分布、2 なら逆二乗)
    """

    # 影響関数(効用関数)
    def distance_utility(d):
        if M == 1:
            return np.exp(-d**2)
        elif M == 2:
            return 1/(d**2) if d != 0 else float('inf')
        else:
            raise ValueError("Mの値が不正です")

    # **座席間の距離行列を事前計算**
    seat_distance_cache = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            seat_distance_cache[i][j] = np.linalg.norm(seatCoordinate[i] - seatCoordinate[j])

    # **距離効用関数のキャッシュ**
    utility_cache = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            utility_cache[i][j] = distance_utility(seat_distance_cache[i][j])

    def cost_function(proposal, seatPrefList, i):
        """
        各 proposal のコストを計算（キャッシュを利用）
        """
        cost_val = 0
        for j in range(N):
            if i != j:
                cost_val += (utility_cache[proposal[i]][proposal[j]] -
                             utility_cache[seatPrefList[i][i]][seatPrefList[i][j]]) ** 2
        return cost_val

    all_permutations = list(itertools.permutations(range(N)))
    
    # **コスト計算を高速化**
    E = np.zeros(len(all_permutations))
    
    for idx, proposal in enumerate(all_permutations):
        proposal = list(proposal)  # タプルをリストに変換
        saveCost = np.zeros(N)
        for i in range(N):
            saveCost[i] = cost_function(proposal, seatPreferenceList, i)
        E[idx] = np.sum(saveCost)

    # 最小コスト
    min_cost = np.min(E)
    # そのときの配置 (複数あるかもしれない)
    min_indices = np.where(E == min_cost)[0]
    best_configurations = [list(all_permutations[i]) for i in min_indices]

    # 計算結果を「辞書」でまとめて返す
    return {
        "min_cost": float(min_cost),
        "best_configurations": best_configurations,
        "all_costs": E.tolist()
    }