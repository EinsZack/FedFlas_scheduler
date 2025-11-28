import numpy as np


class Splitor_all_random(object):
    def __init__(self, ES_list, Client_list, env):
        """
        初始化随机基线算法。
        Args:
            ES_list: 边缘服务器列表
            Client_list: 客户端列表
            env: 适配好的 TaskAssignmentEnv 实例
        """
        self.ES_list = ES_list
        self.Client_list = Client_list
        self.env = env
        self.num_clients = len(Client_list)
        self.num_devices = len(ES_list) + 1  # Local + M ES

    def run_simulation(self, num_episodes=5000):
        """
        执行 RS+RO (随机切分+随机卸载) 5000次，计算平均 Makespan。

        Returns:
            avg_makespan: 平均最大完成时间
            best_config: (best_split_nums, best_allocation) 最好的一次配置（可选用于展示）
        """
        makespan_list = []
        split_options = [4, 9, 16]

        best_makespan = float('inf')
        best_config = (None, None)

        print(f"Running RS+RO baseline for {num_episodes} episodes...")

        for _ in range(num_episodes):
            # 1. 随机生成 Split Nums (N,)
            split_nums = np.random.choice(split_options, size=self.num_clients).astype(np.int32)

            # 2. 随机生成 Allocation Matrix (N, M+1)
            allocation_matrix = np.zeros((self.num_clients, self.num_devices), dtype=np.int32)

            # 均匀概率分布 (也可以改为随机概率分布 np.random.dirichlet 以增加随机性，但这里用均匀即可)
            probs = np.ones(self.num_devices) / self.num_devices

            for i in range(self.num_clients):
                n = split_nums[i]
                # 使用多项式分布生成随机整数分配，确保总和为 n
                allocation_matrix[i] = np.random.multinomial(n, probs)

            # 3. 调用新环境的核心计算逻辑
            # 关键：必须传入 split_nums_override，因为这是每轮随机生成的，不是环境内部存储的
            client_latencies, _ = self.env._calculate_system_metrics(
                allocation_matrix,
                split_nums_override=split_nums
            )

            # 4. 记录 Makespan (所有客户端中最慢的时间)
            current_makespan = np.max(client_latencies)
            makespan_list.append(current_makespan)

            # 记录最优解（可选）
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_config = (split_nums.copy(), allocation_matrix.copy())

        # 5. 计算统计数据
        avg_makespan = np.mean(makespan_list)
        std_makespan = np.std(makespan_list)

        print(f"RS+RO Results: Avg Makespan={avg_makespan:.4f}, Std={std_makespan:.4f}, Best={best_makespan:.4f}")

        # 返回平均值用于绘图对比，同时返回最优配置以备不时之需
        return avg_makespan, best_config