import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


# 假设 Database 类定义在外部，保持引用
# from your_module import Database

class TaskAssignmentEnv(gym.Env):
    def __init__(self, ES_list, Client_list, split_num_list, model_type="cifar10", bandwidth=60.0):
        super(TaskAssignmentEnv, self).__init__()

        # --- 1. 基础属性初始化 ---
        self.ES_list = ES_list
        self.Client_list = Client_list
        self.N = len(Client_list)  # 客户端数量
        self.M = len(ES_list)  # ES 数量
        self.num_devices = self.M + 1
        self.model_type = model_type

        # 将 split_num_list 转为数组
        self.split_num_list = np.array(split_num_list, dtype=np.int32) if split_num_list is not None else np.zeros(
            self.N, dtype=np.int32)
        self.batches_list = np.array([c.num_batches for c in Client_list])

        # --- 2. 设备性能与网络初始化 ---
        self.database = Database()
        self.bandwidth_map = self._init_bandwidths(bandwidth)

        # bandwidth_matrix: (N, M+1), 索引 0 为本地（无限带宽），1~M 为 ES
        self.bandwidth_matrix = np.full((self.N, self.num_devices), float('inf'), dtype=np.float32)
        for i, client in enumerate(self.Client_list):
            for j, es in enumerate(self.ES_list):
                self.bandwidth_matrix[i, j + 1] = self.bandwidth_map[(client.rank, es.rank)]

        # speed_matrix: (N, M+1)
        self.device_flops = np.zeros(self.num_devices, dtype=np.float32)
        # 填充 ES 算力 (索引 1~M)
        for j, es in enumerate(self.ES_list):
            self.device_flops[j + 1] = es.MFLOPs
        # 客户端算力列表
        self.client_flops_list = np.array([c.MFLOPs for c in Client_list])

        # --- 3. Gym 空间定义 ---
        self.action_space = spaces.Discrete(self.num_devices)
        state_dim = self.M + 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        # --- 4. 运行时状态 ---
        self.allocation = None
        self.path_queue = []
        self.current_step = 0
        self.total_paths = 0

        if np.sum(self.split_num_list) > 0:
            self._precompute_path_info()

    def _init_bandwidths(self, bandwidth):
        bw_map = {}
        for client in self.Client_list:
            for es in self.ES_list:
                bw_map[(client.rank, es.rank)] = random.uniform(bandwidth, bandwidth)
        return bw_map

    def _precompute_path_info(self):
        """预生成所有待分配路径的信息，并按计算量排序放入队列"""
        tasks = []
        for i, s in enumerate(self.split_num_list):
            if s <= 0: continue
            b = self.batches_list[i]

            # 仅用于排序的粗略估算
            c_size = (self.database.get_compsize((s, 0), self.model_type) +
                      self.database.get_compsize((s, 3), self.model_type)) * b

            t_size = (self.database.get_transsize((s, 0), self.model_type) +
                      self.database.get_transsize((s, 1), self.model_type) +
                      self.database.get_transsize((s, 2), self.model_type) +
                      self.database.get_transsize((s, 3), self.model_type)) * b

            task_info = {
                'client_id': i,
                'split_num': s,
                'batch': b,
                'comp1_size': self.database.get_compsize((s, 0), self.model_type) * b,
                'comp4_size': self.database.get_compsize((s, 3), self.model_type) * b,
                'trans_total_size': t_size,
                'raw_sort_metric': c_size
            }

            for _ in range(s):
                tasks.append(task_info)

        tasks.sort(key=lambda x: x['raw_sort_metric'], reverse=True)
        self.path_queue = tasks
        self.total_paths = len(tasks)

    # =========================================================================
    # 核心计算逻辑：修正版 (基于 Reference Env 的 Phase 逻辑)
    # =========================================================================
    def _calculate_system_metrics(self, allocation_matrix, split_nums_override=None):
        """
        根据 GymPPOEnv 的逻辑修正时延计算。
        包含 Phase1(Upload/Comp) -> Phase2&3(Local) -> Phase4(Download/Comp)
        以及 Mask 传输机制。
        """
        N, num_devs = allocation_matrix.shape
        M = num_devs - 1
        active_split_nums = split_nums_override if split_nums_override is not None else self.split_num_list

        # 1. 初始化结果容器
        client_latencies = np.zeros(N)
        # ES 的计算负载 (用于计算 ES 完成时间)，累计 (Comp1 + Comp4) * Batch
        es_total_loads = np.zeros(M)

        # 2. 遍历每个客户端计算时延
        for i in range(N):
            s = active_split_nums[i]
            if s <= 0: continue

            b = self.batches_list[i]
            # dist: (M+1,) 向量，表示 Client i 在 [Local, ES1, ..., ESM] 分配的路径数量
            dist = allocation_matrix[i]

            # === 准备数据 ===
            # 获取当前 Split 下的单路径数据量 (Unit Size)
            c1_unit = self.database.get_compsize((s, 0), self.model_type)
            c4_unit = self.database.get_compsize((s, 3), self.model_type)

            t1_unit = self.database.get_transsize((s, 0), self.model_type) + \
                      self.database.get_transsize((s, 1), self.model_type)
            t4_unit = self.database.get_transsize((s, 2), self.model_type) + \
                      self.database.get_transsize((s, 3), self.model_type)

            # === 计算 ES 累积负载 ===
            # 参考逻辑：ES 负载是 allocation * unit_comp。
            # 为了反映总占用时间，这里乘以 batch。
            # dist[1:] 是分配给 M 个 ES 的路径数
            es_total_loads += dist[1:] * (c1_unit + c4_unit) * b

            # === 计算 Client 侧详细时延 (Phase Logic) ===

            # 构建当前 Client 视角的算力向量 (M+1): [Client_i_Speed, ES1_Speed, ...]
            current_speeds = np.concatenate(([self.client_flops_list[i]], self.device_flops[1:]))
            # 当前 Client 的带宽向量 (M+1)
            current_bws = self.bandwidth_matrix[i]

            # --- Phase 1: Upload + Comp1 ---
            # 计算负载：dist * c1 (向量化计算 Local 和所有 ES 的负载)
            comp1_loads_vec = dist * c1_unit

            # 传输负载：Mask 机制
            # 如果分配了路径 (>0)，则产生一次传输开销；否则为 0。
            trans_mask = np.where(dist > 0, 1.0, 0.0)
            trans_mask[0] = 0.0  # 本地设备无传输
            trans_loads_vec1 = t1_unit * trans_mask

            # Phase 1 时间：(计算 / 速度) + (传输 / 带宽)
            # 这是一个 (M+1) 的向量，取 max 代表并行中的瓶颈
            time_phase1_vec = (comp1_loads_vec / current_speeds) + (trans_loads_vec1 / current_bws)
            # 注意：np.divide 在除以 0 时会报警，但在 env 初始化中 bandwidth 设置为 inf 或具体值，speed 也有值
            # 简单处理 NaN (虽然理论上不应出现，因为 dist>0 时 mask 才生效)
            time_phase1 = np.max(np.nan_to_num(time_phase1_vec))

            # --- Phase 2&3: Local Processing (Fusion/Intermediate) ---
            # 这部分只在本地运行，通常视为固定开销，不随 allocation 数量倍增，只随 split_num 变化
            c23_unit = self.database.get_compsize((s, 1), self.model_type) + \
                       self.database.get_compsize((s, 2), self.model_type)
            time_phase23 = c23_unit / self.client_flops_list[i]

            # --- Phase 4: Download + Comp4 ---
            comp4_loads_vec = dist * c4_unit
            trans_loads_vec4 = t4_unit * trans_mask  # 复用 mask，只要有去就有回

            time_phase4_vec = (comp4_loads_vec / current_speeds) + (trans_loads_vec4 / current_bws)
            time_phase4 = np.max(np.nan_to_num(time_phase4_vec))

            # --- Total Latency ---
            # 最终乘以 batch 数量
            client_latencies[i] = (time_phase1 + time_phase23 + time_phase4) * b

        # 3. 计算 ES 完成时间 (用于 Reward 和 Obs)
        es_speeds = self.device_flops[1:]
        es_comp_times = np.divide(es_total_loads, es_speeds, out=np.zeros_like(es_total_loads), where=es_speeds != 0)

        return client_latencies, es_comp_times

    # =========================================================================
    # Gym 接口实现 (保持原样，调用新的计算函数)
    # =========================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.allocation = np.zeros((self.N, self.num_devices), dtype=np.int32)
        self.current_step = 0

        if options and 'split_num_list' in options:
            self.split_num_list = np.array(options['split_num_list'], dtype=np.int32)
            self._precompute_path_info()
        elif self.total_paths == 0 and np.sum(self.split_num_list) > 0:
            self._precompute_path_info()

        return self._get_obs(), {}

    def step(self, action):
        if self.current_step >= self.total_paths:
            return self._get_obs(), 0, True, False, {}

        task = self.path_queue[self.current_step]
        client_id = task['client_id']
        self.allocation[client_id, action] += 1

        # 计算 Metrics (调用修正后的函数)
        client_latencies, es_times = self._calculate_system_metrics(self.allocation)
        current_makespan = np.max(client_latencies)
        load_std = np.std(es_times)

        # Reward
        reward = -current_makespan

        self.current_step += 1
        terminated = self.current_step >= self.total_paths
        truncated = False

        info = {
            "makespan": current_makespan,
            "load_std": load_std,
            "allocation": self.allocation.copy()
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        if self.current_step < self.total_paths:
            task = self.path_queue[self.current_step]
            feat_task_comp = task['comp1_size'] + task['comp4_size']
            feat_task_trans = task['trans_total_size']
            current_client_id = task['client_id']
        else:
            feat_task_comp = 0
            feat_task_trans = 0
            current_client_id = 0

        client_latencies, es_times = self._calculate_system_metrics(self.allocation)

        current_local_lat = client_latencies[current_client_id]
        avg_client_lat = np.mean(client_latencies)
        progress = self.current_step / self.total_paths if self.total_paths > 0 else 0

        obs = np.concatenate([
            [feat_task_comp / 1e6],
            [feat_task_trans / 1e3],
            [current_local_lat],
            es_times,
            [progress],
            [avg_client_lat]
        ]).astype(np.float32)

        return obs

    # =========================================================================
    # GWO 辅助接口
    # =========================================================================
    def simulate_client_time(self, client_index, allocation_vector, split_num):
        temp_alloc = np.zeros((self.N, self.num_devices), dtype=np.int32)
        temp_alloc[client_index] = allocation_vector

        temp_split_nums = np.zeros(self.N, dtype=np.int32)
        temp_split_nums[client_index] = split_num

        client_latencies, _ = self._calculate_system_metrics(temp_alloc, split_nums_override=temp_split_nums)
        return client_latencies[client_index]


class ES:
    def __init__(self, rank, MFLOPs):
        self.rank = rank
        self.MFLOPs = MFLOPs

class Client:
    def __init__(self, rank, MFLOPs, num_batches):
        self.rank = rank  # 设备id
        self.MFLOPs = MFLOPs  # 设备的处理速度，单位为MFlops
        self.num_batches = num_batches  # 设备本地数据量，单位为batch

class Database:
    """
    这是个用于存储各种要用到的数据的类，每一项在这里有明确说明
    (a,b)含义：
    a等分下，b类型
    b类型有0 1 2 3四种，
    在传输中分别代表：前向卸载、前向返回、反向卸载、反向返回 的传输量；单位为MB
    在计算中分别代表：前向卷积、前向全连接、反向全连接、反向卷积 的计算量；单位为MFlops
    1. cifar10在SplitLeNet12下的传输数据集
    """

    def __init__(self):
        self.cifar_transmission_database = {(4, 0): 4.23,
                                            (4, 1): 1.125,
                                            (4, 2): 1.125,
                                            (4, 3): 1.0948,
                                            (9, 0): 2.64,
                                            (9, 1): 0.5,
                                            (9, 2): 0.5,
                                            (9, 3): 1.0948,
                                            (16, 0): 1.98,
                                            (16, 1): 0.28125,
                                            (16, 2): 0.28125,
                                            (16, 3): 1.0948}
        self.cifar_computation_database = {(4, 0): 9424.0,
                                           (4, 1): 455.0,
                                           (4, 2): 910.0,
                                           (4, 3): 18848.0,
                                           (9, 0): 5268.0,
                                           (9, 1): 455.0,
                                           (9, 2): 910.0,
                                           (9, 3): 10536.0,
                                           (16, 0): 3654.0,
                                           (16, 1): 455.0,
                                           (16, 2): 910.0,
                                           (16, 3): 7308.0}
        self.fmnist_transmission_database = {(4, 0): 0.3164,
                                            (4, 1): 0.5625,
                                            (4, 2): 0.5625,
                                            (4, 3): 0.208,
                                            (9, 0): 0.1914,
                                            (9, 1): 0.25,
                                            (9, 2): 0.25,
                                            (9, 3): 0.208,
                                            (16, 0): 0.1406,
                                            (16, 1): 0.1406,
                                            (16, 2): 0.1406,
                                            (16, 3): 0.208}
        self.fmnist_computation_database = {(4, 0): 524.0,
                                           (4, 1): 71.478,
                                           (4, 2): 142.956,
                                           (4, 3): 1048.0,
                                           (9, 0): 239.0,
                                           (9, 1): 71.478,
                                           (9, 2): 142.956,
                                           (9, 3): 478.0,
                                           (16, 0): 138.0,
                                           (16, 1): 71.478,
                                           (16, 2): 142.956,
                                           (16, 3): 276.0}
        self.cifar100_transmission_database = {(4, 0): 4.23,
                                            (4, 1): 2.25,
                                            (4, 2): 2.25,
                                            (4, 3): 4.58,
                                            (9, 0): 2.64,
                                            (9, 1): 1.0,
                                            (9, 2): 1.0,
                                            (9, 3): 4.58,
                                            (16, 0): 1.98,
                                            (16, 1): 0.5625,
                                            (16, 2): 0.5625,
                                            (16, 3): 4.58}
        self.cifar100_computation_database = {(4, 0): 37090.0,
                                             (4, 1): 1218.0,
                                             (4, 2): 2436.0,
                                             (4, 3): 74180.0,
                                             (9, 0): 20700.0,
                                             (9, 1): 1218.0,
                                             (9, 2): 2436.0,
                                             (9, 3): 41400.0,
                                             (16, 0): 14338.0,
                                             (16, 1): 1218.0,
                                             (16, 2): 2436.0,
                                             (16, 3): 28676.0}
        self.mnist_transmission_database = {(4, 0): 0.3164,
                                            (4, 1): 0.5625,
                                            (4, 2): 0.5625,
                                            (4, 3): 0.208,
                                            (9, 0): 0.1914,
                                            (9, 1): 0.25,
                                            (9, 2): 0.25,
                                            (9, 3): 0.208,
                                            (16, 0): 0.1406,
                                            (16, 1): 0.1406,
                                            (16, 2): 0.1406,
                                            (16, 3): 0.208}
        self.mnist_computation_database = {(4, 0): 524.0,
                                             (4, 1): 5.898,
                                             (4, 2): 11.796,
                                             (4, 3): 1048.0,
                                             (9, 0): 239.0,
                                             (9, 1): 5.898,
                                             (9, 2): 11.796,
                                             (9, 3): 478.0,
                                             (16, 0): 138.0,
                                             (16, 1): 5.898,
                                             (16, 2): 11.796,
                                             (16, 3): 276.0}

        # 这个用于计算没有分割的模型的传输和计算用时
        self.nonsplit_transmission_database = {}


    def get_transsize(self, key_tuple, database_type):
        if database_type == "cifar10":
            return self.cifar_transmission_database[key_tuple]
        elif database_type == "fmnist":
            return self.fmnist_transmission_database[key_tuple]
        elif database_type == "cifar100":
            return self.cifar100_transmission_database[key_tuple]
        elif database_type == "mnist":
            return self.mnist_transmission_database[key_tuple]

    def get_compsize(self, key_tuple, database_type):
        if database_type == "cifar10":
            return self.cifar_computation_database[key_tuple]
        elif database_type == "fmnist":
            return self.fmnist_computation_database[key_tuple]
        elif database_type == "cifar100":
            return self.cifar100_computation_database[key_tuple]
        elif database_type == "mnist":
            return self.mnist_computation_database[key_tuple]
