import numpy as np


class Splitor(object):
    def __init__(self, ES_list, Client_list, env=None):
        self.ES_list = ES_list
        self.Client_list = Client_list
        self.ES_power_list = np.array([es.MFLOPs for es in self.ES_list])
        self.env = env

    def get_split_numlist(self):
        client_batches = [client.num_batches for client in self.Client_list]
        client_MFLOPs = [client.MFLOPs for client in self.Client_list]
        ESs_MFLOPs = sum([es.MFLOPs for es in self.ES_list])
        client_local_time = [client_batches[i] / (client_MFLOPs[i] + ESs_MFLOPs) for i in range(len(client_batches))]
        client_local_time_index = np.argsort(client_local_time)[::-1]  # 从大到小排序索引
        best_distribution_num_list = np.zeros(len(client_local_time_index))
        best_dist_list = []

        # 最慢设备进行最优分配（使用 GWO）
        # slowest_powerlist = np.insert(self.ES_power_list, 0, client_MFLOPs[client_local_time_index[0]])
        slowest_best_time = 999.9
        slowest_best_split_num = 0
        slowest_device_best_dist = []
        for n in [4, 9, 16]:
            best_dist, best_time = self.task_distribution(client_local_time_index[0], n)
            if best_time < slowest_best_time:
                slowest_best_time = best_time
                slowest_best_split_num = n
                slowest_device_best_dist = best_dist
        print(f"slowest_idx:{client_local_time_index[0]} ,slowest best time:{slowest_best_time}")
        best_dist_list.append(slowest_device_best_dist)
        best_distribution_num_list[client_local_time_index[0]] = slowest_best_split_num
        # 非最慢设备使用贪心分配
        # 贪心必须读取env才能当场算
        for index in client_local_time_index[1:]:
            # now_size = 1.0
            # current_powerlist = np.insert(self.ES_power_list, 0, client_MFLOPs[index])
            available_n = [4, 9, 16]
            batches = client_batches[index]
            current_best_dist = []
            available_time = []
            for n in [4, 9, 16]:
                distribution, time = self.close_to_target_distribution(n, index, slowest_best_time, self.env)
                # now_size = now_size * growth_rate
                available_time.append(time)
                current_best_dist.append(distribution)
                if time < slowest_best_time:
                    break
            best_distribution_num_list[index] = available_n[np.argmin(available_time)]
            best_dist_list.append(current_best_dist[np.argmin(available_time)])

        best_dist_list_sort = best_dist_list.copy()
        for i in range(len(client_local_time_index)):
            ind = client_local_time_index[i]
            best_dist_list_sort[ind] = best_dist_list[i]

        return best_distribution_num_list, np.array(best_dist_list_sort)

    def task_distribution(self, index, n):
        """client_local_time_index[0], n
        使用灰狼优化算法（GWO）求解单组任务的最短并行时间分配。
        n: 总任务数量
        powers: 各设备算力列表
        now_size: 当前任务粒度的计算量因子
        """
        # 预准备工作，更新env
        self.env.renew_one_time(index, n)

        upbound = len(self.env.ES_list) + 1

        M = upbound
        wolf_pop = 15  # 狼群大小
        max_iter = 50  # 最大迭代次数
        lb = np.zeros(M)  # 任务分配下界
        ub = np.ones(M) * n  # 任务分配上界

        # 初始化狼群（随机分配任务）
        wolves = np.random.uniform(lb, ub, (wolf_pop, M))
        wolves = np.round(wolves * n / np.sum(wolves, axis=1)[:, None])  # 归一化并离散化
        for i in range(wolf_pop):
            wolves[i] = self.adjust_allocation(wolves[i], n, index)  # 确保总任务数为 n

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
                # 计算适应度（最大并行时间）
                # fitness = self.fitness(wolves[i], powers, now_size)
                fitness = self.env.get_client_time(wolves[i], index)
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
                    # 更新基于 alpha 的位置
                    r1 = np.random.random()
                    r2 = np.random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - wolves[i][j])
                    X1 = alpha_pos[j] - A1 * D_alpha

                    # 更新基于 beta 的位置
                    r1 = np.random.random()
                    r2 = np.random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[j] - wolves[i][j])
                    X2 = beta_pos[j] - A2 * D_beta

                    # 更新基于 delta 的位置
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
        best_time = self.env.get_client_time(best_allocation, index)

        return best_allocation.tolist(), best_time

    def adjust_allocation(self, allocation, n, index):
        """
        调整分配方案，确保任务总数为 n 且为整数。
        """
        M = len(allocation)
        # 获取各设备算力
        powers = np.zeros(M)
        for i in range(M):
            powers[i] = self.env.ES_list[i - 1].MFLOPs if i > 0 else self.env.Client_list[index].MFLOPs

        allocation = np.round(allocation).astype(int)
        current_sum = np.sum(allocation)
        # 逐步调整直到总和等于 n
        while current_sum != n:
            diff = n - current_sum
            if diff > 0:
                # 增加任务：选择分配最少的设备
                # idx = np.argmin(allocation)
                min_value = np.min(allocation)
                min_idxs = np.where(allocation == min_value)[0]
                # 在这些设备中选择算力最大的
                idx = min_idxs[np.argmax(powers[min_idxs])]

                allocation[idx] += 1
            else:
                # 减少任务：选择分配最多的设备，且确保不产生负值
                valid_indices = np.where(allocation > 0)[0]
                if len(valid_indices) == 0:
                    # 如果所有设备分配为 0，随机选择一个增加（稍后会减少）
                    allocation[np.random.randint(len(allocation))] += 1
                    current_sum += 1
                    continue
                idx = valid_indices[np.argmax(allocation[valid_indices])]
                allocation[idx] -= 1
            current_sum = np.sum(allocation)
        return allocation

    @staticmethod
    def close_to_target_distribution(n, index, target_time, env=None):  # 也就是贪心算法
        """
        按指定规则分配任务：优先分配给index=0（也就是本地）的设备，其次按贪心策略分配。
        新贪心，由于需要从env中读取信息求时间，所以不能用估算方式，放弃优先给client卸载的策略
        计算时间无法再按照一条路径一条路径计算了，因为在多个阶段都有并行化等待
        """
        M = len(env.ES_list) + 1  # 设备数量
        allocation = [0] * M  # 可卸载的设备数量
        remaining = n  # 路径分割数量
        # 预准备，根据当前客户端索引和分割数量更新一次时间
        env.renew_one_time(index, n)

        # 获取各设备算力
        powers = np.zeros(M)
        for i in range(M):
            powers[i] = env.ES_list[i - 1].MFLOPs if i > 0 else env.Client_list[index].MFLOPs

        # 优先分配给本地设备（index=0）
        max_tasks_0 = n  # 初始假设本地可分配所有任务
        for k in range(1, n + 1):
            temp_allocation = np.zeros(M, dtype=int)
            temp_allocation[0] = k
            time = env.get_client_time(temp_allocation, index)
            if time > target_time:
                max_tasks_0 = k - 1
                break
        allocation[0] = min(max_tasks_0, remaining)
        remaining -= allocation[0]

        # 剩余任务分配给负载最小的设备
        while remaining > 0:
            loads = np.zeros(M)
            for i in range(M):
                temp_allocation = allocation.copy()
                temp_allocation[i] += 1
                loads[i] = env.get_client_time(temp_allocation, index)
            # 找出最小负载
            min_load = np.min(loads)
            # 收集负载等于 min_load 的设备索引
            min_load_indices = np.where(loads == min_load)[0]
            # 在这些设备中选择算力最大的
            idx = min_load_indices[np.argmax(powers[min_load_indices])]
            allocation[idx] += 1
            remaining -= 1

        # 修正分配
        completion_time = env.get_client_time(allocation, index)

        return allocation, completion_time
