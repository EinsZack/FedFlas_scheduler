import numpy as np


class Splitor_RS_GO(object):
    def __init__(self, ES_list, Client_list, env):
        """
        初始化 RS+GO (随机切分 + 基于负载的贪婪分配) 算法。
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

        # 预先获取设备算力
        # ES 算力 (M,)
        self.es_flops = np.array([es.MFLOPs for es in self.ES_list])
        # Client 算力 (C,)
        self.client_flops = np.array([c.MFLOPs for c in self.Client_list])

    def run_simulation(self, num_episodes=5000):
        """
        执行 RS+GO 5000次。

        新的贪婪逻辑：
        不再试错最终时延，而是维护一个实时的 'accumulated_comp_time' (累计计算负载/算力)。
        对每条路径，分配给当前累计计算时间最短的设备（Least Loaded / Earliest Available）。

        Returns:
            avg_makespan: 平均最大完成时间
            best_config: (split_nums, allocation_matrix)
        """
        makespan_list = []
        split_options = [4, 9, 16]

        best_makespan = float('inf')
        best_config = (None, None)

        print(f"Running RS+GO (Load-based Greedy) baseline for {num_episodes} episodes...")

        for episode in range(num_episodes):
            # 1. 随机生成所有客户端的 Split Nums
            split_nums = np.random.choice(split_options, size=self.num_clients).astype(np.int32)

            # 2. 初始化分配矩阵
            allocation_matrix = np.zeros((self.num_clients, self.num_devices), dtype=np.int32)

            # 3. 初始化设备当前的计算负载 (Accumulated Computation Time)
            # 注意：ES 的负载是所有 Client 共享的，而 Local 的负载是各 Client 独立的
            # 为了统一比较，我们维护：
            # es_busy_time: 记录每个 ES 当前堆积的任务需要处理多久
            es_busy_time = np.zeros(self.num_es, dtype=np.float32)

            # 4. 逐个客户端、逐条路径进行分配
            for c in range(self.num_clients):
                n = split_nums[c]
                s = n  # 当前切分数量

                # 获取该切分下，单条路径的计算量 (FLOPs)
                # 计算量 = (Stage 1 + Stage 4) * Batch
                # 这里我们假设贪婪算法只关注核心计算负载
                b = self.Client_list[c].num_batches
                # 使用环境中的 database 获取计算量
                comp_size_per_path = (self.env.database.get_compsize((s, 0), self.env.model_type) +
                                      self.env.database.get_compsize((s, 3), self.env.model_type)) * b

                # 获取当前客户端本地的算力
                local_flop = self.client_flops[c]
                # 获取当前客户端本地已分配的负载 (仅针对当前Client)
                # 注意：如果是实时的，Local 的负载只受自己分配的任务影响
                local_busy_time = 0.0

                # 逐条路径分配
                for _ in range(n):
                    # 决策：比较 Local 和 各个 ES 的当前忙碌程度
                    # 选项 0: Local
                    # 预计增加的时间 = 任务量 / 本地算力
                    cost_local = comp_size_per_path / local_flop
                    projected_local_time = local_busy_time + cost_local

                    # 选项 1..M: ES
                    # 预计增加的时间 = 任务量 / ES算力
                    costs_es = comp_size_per_path / self.es_flops
                    projected_es_times = es_busy_time + costs_es

                    # 寻找最小值
                    # 比较 [projected_local_time] 和 projected_es_times

                    min_es_idx = np.argmin(projected_es_times)
                    min_es_time = projected_es_times[min_es_idx]

                    if projected_local_time <= min_es_time:
                        # 分配给本地
                        allocation_matrix[c, 0] += 1
                        local_busy_time = projected_local_time
                    else:
                        # 分配给负载最小的 ES (注意 ES 索引偏移)
                        allocation_matrix[c, min_es_idx + 1] += 1
                        es_busy_time[min_es_idx] = min_es_time

            # 5. 全局评估 (Evaluation)
            # 分配完成后，使用环境的真实模型（包含传输时延、并行机制等）计算最终 Makespan
            client_latencies, _ = self.env._calculate_system_metrics(
                allocation_matrix,
                split_nums_override=split_nums
            )

            current_makespan = np.max(client_latencies)
            makespan_list.append(current_makespan)

            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_config = (split_nums.copy(), allocation_matrix.copy())

        # 6. 统计结果
        avg_makespan = np.mean(makespan_list)
        std_makespan = np.std(makespan_list)

        print(f"RS+GO Results: Avg Makespan={avg_makespan:.4f}, Std={std_makespan:.4f}, Best={best_makespan:.4f}")

        return avg_makespan, best_config