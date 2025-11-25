import numpy as np

import numpy as np


class Splitor_All(object):
    def __init__(self, ES_list, Client_list, env=None):
        self.ES_list = ES_list
        self.Client_list = Client_list
        # 可以保留，或者直接从 env 读取
        self.ES_power_list = np.array([es.MFLOPs for es in self.ES_list])
        self.env = env

    def get_split_numlist(self):
        """
        修改后的逻辑：
        对列表中的每一个 Client，都分别进行 GWO 规划，
        在 [4, 9, 16] 中寻找使其自身完成时间最短的切分方案。
        """
        num_clients = len(self.Client_list)
        best_distribution_num_list = np.zeros(num_clients)
        best_dist_list = []

        print("Start GWO planning for ALL clients...")

        # 直接按照索引顺序遍历所有客户端
        for index in range(num_clients):
            client_best_time = float('inf')
            client_best_split_num = 0
            client_best_dist = []

            # 尝试不同的切分数量
            for n in [4, 9, 16]:
                # 调用 GWO 算法寻找当前 n 下的最优分配和时间
                distribution, time = self.task_distribution(index, n)

                # 记录该客户端的最优解
                if time < client_best_time:
                    client_best_time = time
                    client_best_split_num = n
                    client_best_dist = distribution

            # 存入结果列表 (按 index 顺序)
            best_distribution_num_list[index] = client_best_split_num
            best_dist_list.append(client_best_dist)

            print(f"Client {index} (Rank {self.Client_list[index].rank}): "
                  f"Best Split={client_best_split_num}, Time={client_best_time:.4f}")

        return best_distribution_num_list, np.array(best_dist_list)

    def task_distribution(self, index, n):
        """
        使用灰狼优化算法（GWO）求解单组任务的最短并行时间分配。
        n: 总任务数量
        index: 当前规划的 Client 索引
        """
        # [修改点 1] 移除了 env.renew_one_time(index, n)，新环境不需要显式刷新

        upbound = len(self.env.ES_list) + 1
        M = upbound

        # GWO 参数
        wolf_pop = 15
        max_iter = 50
        lb = np.zeros(M)
        ub = np.ones(M) * n

        # 初始化狼群
        wolves = np.random.uniform(lb, ub, (wolf_pop, M))
        # 归一化并离散化，确保和为 n
        wolves = np.round(wolves * n / np.sum(wolves, axis=1)[:, None])
        for i in range(wolf_pop):
            wolves[i] = self.adjust_allocation(wolves[i], n, index)

        # 初始化 alpha, beta, delta 狼
        alpha_pos = np.zeros(M)
        alpha_score = float('inf')
        beta_pos = np.zeros(M)
        beta_score = float('inf')
        delta_pos = np.zeros(M)
        delta_score = float('inf')

        # 主循环
        for t in range(max_iter):
            for i in range(wolf_pop):
                # [修改点 2] 使用新环境的 simulate_client_time 接口计算适应度
                # 注意：这里模拟的是单客户端视角下的时间
                fitness = self.env.simulate_client_time(index, wolves[i], n)

                # 更新 alpha, beta, delta
                if fitness < alpha_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = alpha_score
                    beta_pos = alpha_pos.copy()
                    alpha_score = fitness
                    alpha_pos = wolves[i].copy()
                elif fitness < beta_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = fitness
                    beta_pos = wolves[i].copy()
                elif fitness < delta_score:
                    delta_score = fitness
                    delta_pos = wolves[i].copy()

            # 更新狼群位置
            a = 2 - t * (2 / max_iter)  # a 从 2 线性递减到 0
            for i in range(wolf_pop):
                for j in range(M):
                    # 更新基于 alpha
                    r1 = np.random.random()
                    r2 = np.random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - wolves[i][j])
                    X1 = alpha_pos[j] - A1 * D_alpha

                    # 更新基于 beta
                    r1 = np.random.random()
                    r2 = np.random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[j] - wolves[i][j])
                    X2 = beta_pos[j] - A2 * D_beta

                    # 更新基于 delta
                    r1 = np.random.random()
                    r2 = np.random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta_pos[j] - wolves[i][j])
                    X3 = delta_pos[j] - A3 * D_delta

                    # 平均位置并限制范围
                    wolves[i][j] = (X1 + X2 + X3) / 3
                    wolves[i][j] = np.clip(wolves[i][j], lb[j], ub[j])

                # 离散化和约束修正
                wolves[i] = np.round(wolves[i] * n / np.sum(wolves[i]))
                wolves[i] = self.adjust_allocation(wolves[i], n, index)

        # 返回最优解
        best_allocation = np.round(alpha_pos).astype(int)
        best_allocation = self.adjust_allocation(best_allocation, n, index)

        # [修改点 3] 计算最终时间同样使用 simulate 接口
        best_time = self.env.simulate_client_time(index, best_allocation, n)

        return best_allocation.tolist(), best_time

    def adjust_allocation(self, allocation, n, index):
        """
        调整分配方案，确保任务总数为 n 且为整数。
        """
        M = len(allocation)
        # 获取各设备算力
        powers = np.zeros(M)
        for i in range(M):
            # i=0 是 Client (Local), i>0 是 ES
            # 依然通过 env 获取对象属性，这是兼容的
            powers[i] = self.env.ES_list[i - 1].MFLOPs if i > 0 else self.env.Client_list[index].MFLOPs

        allocation = np.round(allocation).astype(int)
        current_sum = np.sum(allocation)
        # 逐步调整直到总和等于 n
        while current_sum != n:
            diff = n - current_sum
            if diff > 0:
                # 增加任务：选择分配最少且算力最大的设备
                min_value = np.min(allocation)
                min_idxs = np.where(allocation == min_value)[0]
                idx = min_idxs[np.argmax(powers[min_idxs])]
                allocation[idx] += 1
            else:
                # 减少任务：选择分配最多且算力最小的设备（或者简单地选最多的）
                # 原逻辑：选择分配最多的设备
                valid_indices = np.where(allocation > 0)[0]
                if len(valid_indices) == 0:
                    allocation[np.random.randint(len(allocation))] += 1
                    current_sum += 1
                    continue
                idx = valid_indices[np.argmax(allocation[valid_indices])]
                allocation[idx] -= 1
            current_sum = np.sum(allocation)
        return allocation