import numpy as np
import math
import random

def seat_arrange_EMC(N, seatPreferenceList, seatCoordinate, M, 
                            L=None, steps=2000, 
                            init_method='use_preference'):
    """
    席替え問題を 交換モンテカルロ法(Replica Exchange MCMC) で解くサンプルコード

    [引数]
      N: 人数 (席数と同じ)
      seatPreferenceList: 各人の希望配置 (N×N のリスト)
      seatCoordinate: 各席の座標 (N×2 のリスト)
      M: 距離モデル (1 -> exp(-d^2), 2 -> 1/d^2)
      L: レプリカ数 (デフォルト: None の場合 N に合わせる)
      steps: MCMCステップ数 (合計イテレーション)
      init_method: 初期化方法 ('use_preference' または 'random')

    [戻り値]
      results: {
         'best_config' : 全ステップ・全レプリカの中で最もコストが低かった配置,
         'best_cost'   : そのコスト,
         'replicas'    : 最終ステップでの全レプリカの配置リスト,
         'replica_cost': 最終ステップでの全レプリカのコストリスト,
         'betas'       : 設定した逆温度のリスト
      }
    """

    # -------------------- 0. パラメータ設定 --------------------
    if L is None:
        L = N  # 例としてレプリカ数を N に設定

    # (1) 逆温度ベクトル beta の作成
    #  ここでは、あなたのコード例でよく見る
    #  beta[0] = 0
    #  if L >= 2: beta[1] = gamma^(2-L)
    #  for l in range(L-2): beta[l+2] = gamma * beta[l+1]
    #  のイメージを再現
    gamma = 1.11  # 適当な値。実験で調整してください
    betas = np.zeros(L)
    betas[0] = 0.0
    if L >= 2:
        betas[1] = gamma**(2 - L)
        for i in range(L - 2):
            betas[i+2] = gamma * betas[i+1]

    # 距離効用関数
    def distance_utility(d):
        if M == 1:
            return math.exp(-d**2)
        elif M == 2:
            return 1/(d**2) if d != 0 else float('inf')
        else:
            raise ValueError("Mの値が不正です")

    # 距離キャッシュ
    seat_distance_cache = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            seat_distance_cache[i][j] = np.linalg.norm(seatCoordinate[i] - seatCoordinate[j])

    # 距離効用キャッシュ
    utility_cache = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            utility_cache[i][j] = distance_utility(seat_distance_cache[i][j])

    # -------------------- 1. コスト計算関数 --------------------
    def cost_function_all(proposal):
        """
        proposal: 長さNのリスト [p0, p1, ..., p_{N-1}]
                  i番目の人が proposal[i]番席に座る
        戻り値: 
          total_cost: float (配置のコスト)
          partial_cost: 各人 i のコスト (長さNの配列)
        """
        partial = np.zeros(N)
        for i in range(N):
            c_i = 0.0
            for j in range(N):
                if i != j:
                    c_i += (utility_cache[proposal[i]][proposal[j]] -
                            utility_cache[seatPreferenceList[i][i]][seatPreferenceList[i][j]])**2
            partial[i] = c_i
        total = np.sum(partial)
        return total, partial

    # -------------------- (2) 初期レプリカの生成 --------------------
    def make_all_permutations():
        from itertools import permutations
        return list(permutations(range(N)))

    if init_method == 'use_preference':
        # 1. seatPreferenceList の各配置のコストを計算
        cost_list = []
        for cfg in seatPreferenceList:
            c, _ = cost_function_all(cfg)
            cost_list.append((c, cfg))  # (コスト, 配置) のタプルを作成

        # 2. コストが低い順にソート（低温レプリカは低コスト配置を優先）
        cost_list.sort(key=lambda x: x[0])  # コスト順にソート（昇順）

        # 3. 上位から順にレプリカに割り当て
        replica_configs = [list(cfg) for _, cfg in cost_list[:L]]  # 上位 L 個を選択

        # もし L 個に満たなかったら、ランダムに追加
        base = list(range(N))
        while len(replica_configs) < L:
            perm = base[:]
            np.random.shuffle(perm)
            replica_configs.append(perm)

    else:
        # ランダム初期化
        replica_configs = []
        base = list(range(N))
        for _ in range(L):
            perm = base[:]
            np.random.shuffle(perm)
            replica_configs.append(perm)

    replica_costs = []
    replica_partials = []
    for cfg in replica_configs:
        c, p = cost_function_all(cfg)
        replica_costs.append(c)
        replica_partials.append(p)

    # --- 「最良の解」を記録する変数 ---
    best_global_cost = float('inf')
    best_global_config = None

    # 全レプリカ初期状態の中で最良を探す
    for l in range(L):
        if replica_costs[l] < best_global_cost:
            best_global_cost = replica_costs[l]
            best_global_config = replica_configs[l][:]

    # -------------------- 3. MCMC のループ --------------------
    def pick_two_seats_by_cost(partial):
        """
        partialの大きい座席が高確率で選ばれるように、
        partial[i]/sum(partial) を用いて1つ目の座席 i を選び、
        i を除いた partial[j]/(sum(partial)-partial[i]) で2つ目の座席 j を選ぶ。
        """
        sum_p = sum(partial)
        if sum_p <= 0:
            # コストがすべて0以下なら一様ランダムに選ぶ
            i = random.randrange(N)
            j = random.randrange(N)
            while j == i:
                j = random.randrange(N)
            return i, j
        
        # 1つ目 i を選ぶ
        r = random.uniform(0, sum_p)
        cum = 0.0
        i = 0
        for seat_i in range(N):
            cum += partial[seat_i]
            if r <= cum:
                i = seat_i
                break

        # 2つ目 j を選ぶ（iを除く）
        sum_p2 = sum_p - partial[i]
        if sum_p2 <= 0:
            # partial[i] 以外が0なら、仕方ないので一様に選ぶ
            j = random.randrange(N)
            while j == i:
                j = random.randrange(N)
            return i, j

        r2 = random.uniform(0, sum_p2)
        cum2 = 0.0
        j = 0
        for seat_j in range(N):
            if seat_j == i:
                continue
            cum2 += partial[seat_j]
            if r2 <= cum2:
                j = seat_j
                break
        return i, j

    for step in range(steps):
        # === (a) 各レプリカで局所更新 ===
        for l in range(L):
            current_cfg = replica_configs[l]
            current_cost = replica_costs[l]
            current_partial = replica_partials[l]

            # 1. コストが高い座席ほど高確率で選択
            i_max, j_max = pick_two_seats_by_cost(current_partial)
            new_cfg = current_cfg[:]
            new_cfg[i_max], new_cfg[j_max] = new_cfg[j_max], new_cfg[i_max]

            # 4. 新コストを計算
            new_cost, new_partial = cost_function_all(new_cfg)

            # 5. メトロポリス判定
            dE = new_cost - current_cost
            beta_l = betas[l]
            # エネルギー(=コスト)が下がったら無条件採択
            # 上がった場合は exp(-beta_l * dE) の確率で採択
            if dE < 0:
                accept = True
            else:
                # random.random()は [0,1)
                accept = (random.random() < math.exp(-beta_l * dE))

            if accept:
                # 採択
                replica_configs[l] = new_cfg
                replica_costs[l] = new_cost
                replica_partials[l] = new_partial
                # 最良解を更新
                if new_cost < best_global_cost:
                    best_global_cost = new_cost
                    best_global_config = new_cfg[:]

        # === (b) レプリカ間の交換ステップ ===
        #     l=0とl=1, l=1とl=2, ... l=L-2とl=L-1 の間で交換を試行
        for l in range(L-1):
            cfg_l   = replica_configs[l]
            cfg_lp1 = replica_configs[l+1]
            cost_l   = replica_costs[l]
            cost_lp1 = replica_costs[l+1]
            beta_l   = betas[l]
            beta_lp1 = betas[l+1]

            # Δ = (β_{l+1} - β_l) * (E_l - E_{l+1})
            delta = (beta_lp1 - beta_l) * (cost_l - cost_lp1)
            if delta < 0:
                p_ex = 1.0
            else:
                p_ex = math.exp(-delta)

            if random.random() < p_ex:
                # 交換を実施
                replica_configs[l], replica_configs[l+1] = cfg_lp1, cfg_l
                replica_costs[l], replica_costs[l+1] = cost_lp1, cost_l
                replica_partials[l], replica_partials[l+1] = (
                    replica_partials[l+1], replica_partials[l]
                )
                # 交換後にも、より良いコストがあるかもしれないのでチェック
                # （例えば高温レプリカから良い解が移動してきた場合）
                if replica_costs[l] < best_global_cost:
                    best_global_cost = replica_costs[l]
                    best_global_config = replica_configs[l][:]
                if replica_costs[l+1] < best_global_cost:
                    best_global_cost = replica_costs[l+1]
                    best_global_config = replica_configs[l+1][:]

    # -------------------- 4. 結果の取り出し --------------------
    # -> steps 回のイテレーションと L 個のレプリカの探索で、
    #    最もコストが低かった配置を best_global_* に保持

    results = {
        "best_configurations": [best_global_config],
        'best_cost': best_global_cost,
        'replicas': replica_configs,       # 最終ステップ時点
        'replica_cost': replica_costs,     # 最終ステップ時点
        'betas': betas.tolist()
    }
    return results



















# import time
# import numpy as np
# import math
# import random

# def seat_arrange(N, seatPreferenceList, seatCoordinate, M, 
#                  L=None, steps=3000, 
#                  init_method='use_preference'):
#     """ 席替え問題を 交換モンテカルロ法(Replica Exchange MCMC) で解く """
    
#     if L is None:
#         L = N  # 例としてレプリカ数を N に設定
    
#     start_time = time.time()  # 全体の開始時間
    
#     # 1. 距離キャッシュの計算
#     t1 = time.time()
#     seat_distance_cache = np.linalg.norm(seatCoordinate[:, None] - seatCoordinate, axis=2)
#     print(f"✅ 距離行列計算: {time.time() - t1:.2f} 秒")
    
#     # 2. 距離効用キャッシュの計算
#     t2 = time.time()
#     def distance_utility(d):
#         if M == 1:
#             return np.exp(-d**2)
#         elif M == 2:
#             return np.where(d != 0, 1/(d**2), float('inf'))
#         else:
#             raise ValueError("Mの値が不正です")
    
#     utility_cache = distance_utility(seat_distance_cache)
#     print(f"✅ 距離効用キャッシュ計算: {time.time() - t2:.2f} 秒")
    
#     # 3. 初期レプリカの生成
#     t3 = time.time()
#     base = list(range(N))
#     replica_configs = [random.sample(base, N) for _ in range(L)]
#     print(f"✅ レプリカ初期化: {time.time() - t3:.2f} 秒")
    
#     # 4. コスト計算関数 (O(N))
#     def cost_function_partial(proposal, i, j, prev_cost, prev_partial):
#         new_partial = prev_partial.copy()
        
#         for idx in [i, j]:
#             new_partial[idx] = sum(
#                 (utility_cache[proposal[idx], proposal[k]] -
#                  utility_cache[seatPreferenceList[idx, idx], seatPreferenceList[idx, k]])**2
#                 for k in range(N) if k != idx
#             )
        
#         new_cost = prev_cost - prev_partial[i] - prev_partial[j] + new_partial[i] + new_partial[j]
#         return new_cost, new_partial
    
#     # 5. 初期コスト計算
#     t4 = time.time()
#     def cost_function_all(proposal):
#         partial = np.zeros(N)
#         for i in range(N):
#             partial[i] = sum(
#                 (utility_cache[proposal[i], proposal[j]] -
#                  utility_cache[seatPreferenceList[i, i], seatPreferenceList[i, j]])**2
#                 for j in range(N) if i != j
#             )
#         return np.sum(partial), partial
    
#     replica_costs, replica_partials = zip(*[cost_function_all(cfg) for cfg in replica_configs])
#     replica_costs, replica_partials = list(replica_costs), list(replica_partials)
#     print(f"✅ 初期コスト計算: {time.time() - t4:.2f} 秒")
    
#     # 6. MCMC ループ (O(N) のコスト更新)
#     t5 = time.time()
#     best_global_cost = float('inf')
#     best_global_config = None
    
#     for step in range(steps):
#         for l in range(L):
#             current_cfg = replica_configs[l]
#             current_cost = replica_costs[l]
#             current_partial = replica_partials[l]
            
#             i, j = random.sample(range(N), 2)  # 交換する2つの座席を選択
#             new_cfg = current_cfg[:]
#             new_cfg[i], new_cfg[j] = new_cfg[j], new_cfg[i]
            
#             new_cost, new_partial = cost_function_partial(new_cfg, i, j, current_cost, current_partial)
            
#             # 受理判定
#             if new_cost < current_cost or random.random() < math.exp(current_cost - new_cost):
#                 replica_configs[l] = new_cfg
#                 replica_costs[l] = new_cost
#                 replica_partials[l] = new_partial
                
#                 if new_cost < best_global_cost:
#                     best_global_cost = new_cost
#                     best_global_config = new_cfg[:]
    
#     print(f"✅ MCMC計算 ({steps} ステップ): {time.time() - t5:.2f} 秒")
#     print(f"🚀 全体処理時間: {time.time() - start_time:.2f} 秒")
    
#     return {
#         'best_configurations': [best_global_config],
#         'min_cost': best_global_cost
#     }











# import time
# import numpy as np
# import math
# import random

# def seat_arrange(N, seatPreferenceList, seatCoordinate, M, 
#                             L=None, steps=3000, 
#                             init_method='use_preference'):
#     """ 席替え問題を 交換モンテカルロ法(Replica Exchange MCMC) で解く """

#     if L is None:
#         L = N  # 例としてレプリカ数を N に設定

#     start_time = time.time()  # 全体の開始時間

#     # 1. 距離キャッシュの計算
#     t1 = time.time()
#     seat_distance_cache = np.zeros((N, N))
#     for i in range(N):
#         for j in range(N):
#             seat_distance_cache[i][j] = np.linalg.norm(seatCoordinate[i] - seatCoordinate[j])
#     print(f"✅ 距離行列計算: {time.time() - t1:.2f} 秒")

#     # 2. 距離効用キャッシュの計算
#     t2 = time.time()
#     def distance_utility(d):
#         if M == 1:
#             return math.exp(-d**2)
#         elif M == 2:
#             return 1/(d**2) if d != 0 else float('inf')
#         else:
#             raise ValueError("Mの値が不正です")

#     utility_cache = np.zeros((N, N))
#     for i in range(N):
#         for j in range(N):
#             utility_cache[i][j] = distance_utility(seat_distance_cache[i][j])
#     print(f"✅ 距離効用キャッシュ計算: {time.time() - t2:.2f} 秒")

#     # 3. 初期レプリカの生成
#     t3 = time.time()
#     replica_configs = []
#     base = list(range(N))
#     for _ in range(L):
#         perm = base[:]
#         np.random.shuffle(perm)
#         replica_configs.append(perm)
#     print(f"✅ レプリカ初期化: {time.time() - t3:.2f} 秒")

#     # 4. 初期コスト計算
#     t4 = time.time()
#     def cost_function_all(proposal):
#         """ コスト計算関数 """
#         partial = np.zeros(N)
#         for i in range(N):
#             c_i = 0.0
#             for j in range(N):
#                 if i != j:
#                     c_i += (utility_cache[proposal[i]][proposal[j]] -
#                             utility_cache[seatPreferenceList[i][i]][seatPreferenceList[i][j]])**2
#             partial[i] = c_i
#         total = np.sum(partial)
#         return total, partial

#     replica_costs = []
#     replica_partials = []
#     for cfg in replica_configs:
#         c, p = cost_function_all(cfg)
#         replica_costs.append(c)
#         replica_partials.append(p)
#     print(f"✅ 初期コスト計算: {time.time() - t4:.2f} 秒")

#     # 5. MCMC ループ
#     t5 = time.time()
#     best_global_cost = float('inf')
#     best_global_config = None

#     for step in range(steps):
#         for l in range(L):
#             current_cfg = replica_configs[l]
#             current_cost = replica_costs[l]

#             # 2つの座席を選び交換
#             i, j = random.sample(range(N), 2)
#             new_cfg = current_cfg[:]
#             new_cfg[i], new_cfg[j] = new_cfg[j], new_cfg[i]

#             # 新コスト計算
#             new_cost, _ = cost_function_all(new_cfg)

#             # 受理判定
#             if new_cost < current_cost or random.random() < math.exp(-new_cost + current_cost):
#                 replica_configs[l] = new_cfg
#                 replica_costs[l] = new_cost

#                 # 最良解更新
#                 if new_cost < best_global_cost:
#                     best_global_cost = new_cost
#                     best_global_config = new_cfg[:]

#     print(f"✅ MCMC計算 ({steps} ステップ): {time.time() - t5:.2f} 秒")

#     # 6. 全体の時間を記録
#     total_time = time.time() - start_time
#     print(f"🚀 全体処理時間: {total_time:.2f} 秒")

#     return {
#         'best_configurations': [best_global_config],
#         'min_cost': best_global_cost
#     }

