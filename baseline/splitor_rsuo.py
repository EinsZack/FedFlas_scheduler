import numpy as np


class Splitor_random_uniform_es_only(object):
    def __init__(self, ES_list, Client_list, env):
        """
        初始化 RS+UO (Random Split + Uniform Offloading to ES) 算法。
        Args:
            ES_list: 边缘服务器列表
            Client_list: 客户端列表
            env: 适配好的 TaskAssignmentEnv 实例
        """
        self.ES_list = ES_list
        self.Client_list = Client_list
        self.env = env
        self.num_clients = len(Client_list)
        self.num_es = len(ES_list)
        self.num_devices = self.num_es + 1  # 0: Local, 1..M: ES

    def run_simulation(self, num_episodes=5000):
        """
        执行 RS+UO 5000次，由于切分数量是随机的，需要多次运行取平均。
        分配策略是确定的：对于给定的 n，均匀分配给 ES。

        Returns:
            avg_makespan: 平均最大完成时间
            best_config: 最优配置
        """
        makespan_list = []
        split_options = [4, 9, 16]

        best_makespan = float('inf')
        best_config = (None, None)

        print(f"Running RS+UO (ES Only) baseline for {num_episodes} episodes...")

        for _ in range(num_episodes):
            # 1. 随机选择分割数量 (N,)
            split_nums = np.random.choice(split_options, size=self.num_clients).astype(np.int32)

            # 2. 构建均匀分配矩阵 (N, M+1)
            allocation_matrix = np.zeros((self.num_clients, self.num_devices), dtype=np.int32)

            for c in range(self.num_clients):
                n = split_nums[c]

                # 均匀分配给 ES (索引 1 到 M)
                base = n // self.num_es
                remainder = n % self.num_es

                # 基础分配
                allocation_matrix[c, 1:] = base

                # 余数处理：简单地分配给前 remainder 个 ES
                if remainder > 0:
                    allocation_matrix[c, 1: 1 + remainder] += 1

                # 验证：本地 (索引0) 应为 0，总和应为 n
                # assert allocation_matrix[c, 0] == 0
                # assert np.sum(allocation_matrix[c]) == n

            # 3. 调用环境计算核心
            # 传入 split_nums_override，因为这是随机生成的
            client_latencies, _ = self.env._calculate_system_metrics(
                allocation_matrix,
                split_nums_override=split_nums
            )

            current_makespan = np.max(client_latencies)
            makespan_list.append(current_makespan)

            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_config = (split_nums.copy(), allocation_matrix.copy())

        # 4. 统计结果
        avg_makespan = np.mean(makespan_list)
        std_makespan = np.std(makespan_list)

        print(f"RS+UO Results: Avg Makespan={avg_makespan:.4f}, Std={std_makespan:.4f}, Best={best_makespan:.4f}")

        return avg_makespan, best_config