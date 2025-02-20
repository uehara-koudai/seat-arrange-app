import numpy as np
import math
import random


def seat_arrange_genetic(N, seatPreferenceList, seatCoordinate, M, population_size=100, generations=2400, mutation_rate=0.1):
    """
    遺伝的アルゴリズムを用いた座席最適配置
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
    
    # コスト関数
    def cost_function(proposal):
        cost_val = 0
        for i in range(N):
            for j in range(N):
                if i != j:
                    cost_val += (utility_cache[proposal[i]][proposal[j]] - utility_cache[seatPreferenceList[i][i]][seatPreferenceList[i][j]]) ** 2
        return cost_val
    
    # 初期集団を生成
    def generate_initial_population():
        population = [list(seatPreferenceList[i]) for i in range(N)]  # 各人の希望配置を使用
        while len(population) < population_size:
            population.append(random.sample(range(N), N))  # ランダム配置
        return population
    
    # 選択（ルーレット選択）
    def selection(population, fitness):
        total_fitness = sum(fitness)
        probs = [f / total_fitness for f in fitness]
        selected = np.random.choice(len(population), size=2, p=probs, replace=False)
        return population[selected[0]], population[selected[1]]
    
    # 交叉（部分交叉）
    def crossover(parent1, parent2):
        cut = random.randint(1, N-1)
        child1 = parent1[:cut] + [p for p in parent2 if p not in parent1[:cut]]
        child2 = parent2[:cut] + [p for p in parent1 if p not in parent2[:cut]]
        return child1, child2
    
    # 突然変異（ランダムな入れ替え）
    def mutate(individual):
        if random.random() < mutation_rate:
            i, j = random.sample(range(N), 2)
            individual[i], individual[j] = individual[j], individual[i]
    
    # 遺伝的アルゴリズムの実行
    population = generate_initial_population()
    best_solution = None
    best_cost = float('inf')
    
    for generation in range(generations):
        fitness = [1 / (cost_function(ind) + 1e-6) for ind in population]  # コストを最小化
        new_population = []
        
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population, fitness)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
        
        population = new_population
        best_idx = np.argmax(fitness)
        if cost_function(population[best_idx]) < best_cost:
            best_cost = cost_function(population[best_idx])
            best_solution = population[best_idx]
    
    return {
        "min_cost": float(best_cost),
        "best_configurations": [best_solution]
    }





