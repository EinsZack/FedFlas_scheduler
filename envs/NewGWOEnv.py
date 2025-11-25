import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random


# 假设 Database 类定义在 utils_DQN 或当前文件中，这里保留引用
# from scheduler_1115.utils_DQN import Database

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

        # 将 split_num_list 转为数组，若初始化时为 None，则需要在 reset 中传入或处理
        self.split_num_list = np.array(split_num_list, dtype=np.int32) if split_num_list is not None else np.zeros(
            self.N, dtype=np.int32)
        self.batches_list = np.array([c.num_batches for c in Client_list])

        # --- 2. 设备性能与网络初始化 ---
        self.database = Database()
        self.bandwidth_map = self._init_bandwidths(bandwidth)

        # 构建矩阵形式的带宽和算力，加速计算
        # bandwidth_matrix: (N, M+1), 索引 0 为本地（无限带宽），1~M 为 ES
        self.bandwidth_matrix = np.full((self.N, self.num_devices), float('inf'), dtype=np.float32)
        for i, client in enumerate(self.Client_list):
            for j, es in enumerate(self.ES_list):
                self.bandwidth_matrix[i, j + 1] = self.bandwidth_map[(client.rank, es.rank)]

        # speed_matrix: (N, M+1), 用于快速索引算力
        # 行索引 i 代表 Client i 的视角，但实际上 ES 的算力是共享的，这里主要方便取用
        self.device_flops = np.zeros(self.num_devices, dtype=np.float32)
        # 填充 ES 算力 (索引 1~M)
        for j, es in enumerate(self.ES_list):
            self.device_flops[j + 1] = es.MFLOPs
        # 客户端算力在计算时根据 i 单独取，这里先存一个列表
        self.client_flops_list = np.array([c.MFLOPs for c in Client_list])

        # --- 3. Gym 空间定义 ---
        # 动作：为当前路径选择一个设备 (0: Local, 1~M: ES)
        self.action_space = spaces.Discrete(self.num_devices)

        # 状态：[当前计算量, 当前传输量, 本地时间负载, ES1时间负载... ESM时间负载, 进度, 平均负载]
        # 维度：2 + 1 + M + 2 = M + 5
        state_dim = self.M + 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        # --- 4. 运行时状态 ---
        self.allocation = None  # 分配矩阵 (N, M+1)
        self.path_queue = []  # 待分配任务队列
        self.current_step = 0
        self.total_paths = 0

        # 预计算路径开销 (依赖 split_num_list)
        if np.sum(self.split_num_list) > 0:
            self._precompute_path_info()

    def _init_bandwidths(self, bandwidth):
        bw_map = {}
        for client in self.Client_list:
            for es in self.ES_list:
                # 这里可以改为随机或读取真实配置
                bw_map[(client.rank, es.rank)] = random.uniform(bandwidth, bandwidth)
        return bw_map

    def _precompute_path_info(self):
        """预生成所有待分配路径的信息，并按计算量排序放入队列"""
        tasks = []
        for i, s in enumerate(self.split_num_list):
            if s <= 0: continue

            # 基础计算量和传输量 (单条路径)
            # 注意：这里存的是单条路径的 raw size，后续计算时再乘以 batch
            b = self.batches_list[i]

            # Stage 1 Comp (Client part + ES part total) -> 用于排序，实际计算分两段
            # 这里简化排序依据：取总计算量
            c_size = (self.database.get_compsize((s, 0), self.model_type) +
                      self.database.get_compsize((s, 3), self.model_type)) * b

            t_size = (self.database.get_transsize((s, 0), self.model_type) +
                      self.database.get_transsize((s, 1), self.model_type) +
                      self.database.get_transsize((s, 2), self.model_type) +
                      self.database.get_transsize((s, 3), self.model_type)) * b

            # 单次任务的具体参数
            task_info = {
                'client_id': i,
                'split_num': s,
                'batch': b,
                'comp1_size': self.database.get_compsize((s, 0), self.model_type) * b,
                'comp4_size': self.database.get_compsize((s, 3), self.model_type) * b,
                'trans_total_size': t_size,
                # 这里为了简化 Gym 观测，取一个总的特征值
                'raw_sort_metric': c_size
            }

            for _ in range(s):
                tasks.append(task_info)

        # 按计算量降序排列 (Heuristic: Longest Job First往往能获得更好Makespan)
        tasks.sort(key=lambda x: x['raw_sort_metric'], reverse=True)
        self.path_queue = tasks
        self.total_paths = len(tasks)

    # =========================================================================
    # 核心计算逻辑：统一处理 GWO (单行) 和 RL (全矩阵)
    # =========================================================================
    def _calculate_system_metrics(self, allocation_matrix, split_nums_override=None):
        """
        根据给定的分配矩阵，计算系统的关键指标。

        Args:
            allocation_matrix: (N, M+1) 矩阵，元素为分配给对应设备的路径数量。
            split_nums_override: (Optional, (N,)) 用于 GWO 阶段覆盖内部 self.split_num_list 的临时切分方案。

        Returns:
            client_latencies: (N,) 每个客户端的总时延
            es_comp_times: (M,) 每个 ES 的计算完成时间
        """
        N, num_devs = allocation_matrix.shape
        M = num_devs - 1

        # [修改点 1] 决定使用哪个切分列表
        # 如果传入了 override (GWO阶段)，则使用传入的；否则使用环境内部的 (RL阶段)
        active_split_nums = split_nums_override if split_nums_override is not None else self.split_num_list

        # 1. 计算所有 ES 的总计算负载 (Global ES Load)
        es_comp_loads_stage1 = np.zeros(M)
        es_comp_loads_stage4 = np.zeros(M)

        for i in range(N):
            s = active_split_nums[i]  # [修改点 2] 使用 active_split_nums
            if s <= 0: continue  # 此时如果 GWO 传入了 s=4/9/16，就不会被跳过

            b = self.batches_list[i]
            alloc_vec = allocation_matrix[i, 1:]  # ES 部分

            # 获取单条路径的计算量
            c1 = self.database.get_compsize((s, 0), self.model_type)
            c4 = self.database.get_compsize((s, 3), self.model_type)

            es_comp_loads_stage1 += alloc_vec * c1 * b
            es_comp_loads_stage4 += alloc_vec * c4 * b

        total_es_loads = es_comp_loads_stage1 + es_comp_loads_stage4

        # 2. 计算 ES 的处理时间
        es_speeds = self.device_flops[1:]
        es_comp_times = np.divide(total_es_loads, es_speeds, out=np.zeros_like(total_es_loads),
                                  where=es_speeds != 0)

        # 3. 计算每个客户端的时延
        client_latencies = np.zeros(N)

        for i in range(N):
            s = active_split_nums[i]  # [修改点 3] 同上
            if s <= 0: continue

            b = self.batches_list[i]
            dist = allocation_matrix[i]

            # --- Local Part ---
            fuse_comp = (self.database.get_compsize((4, 1), self.model_type) +
                         self.database.get_compsize((4, 2), self.model_type)) * b

            local_c1 = self.database.get_compsize((s, 0), self.model_type) * b
            local_c4 = self.database.get_compsize((s, 3), self.model_type) * b

            local_comp_time = (dist[0] * (local_c1 + local_c4) + fuse_comp) / self.client_flops_list[i]

            # --- Offloading Part ---
            offload_times = []

            t1 = self.database.get_transsize((s, 0), self.model_type) + self.database.get_transsize((s, 1),
                                                                                                    self.model_type)
            t4 = self.database.get_transsize((s, 2), self.model_type) + self.database.get_transsize((s, 3),
                                                                                                    self.model_type)
            total_trans_per_path = (t1 + t4) * b

            for j in range(M):
                num_paths = dist[j + 1]
                if num_paths > 0:
                    trans_time = (num_paths * total_trans_per_path) / self.bandwidth_matrix[i, j + 1]
                    wait_comp_time = es_comp_times[j]
                    offload_times.append(trans_time + wait_comp_time)
                else:
                    offload_times.append(0.0)

            max_offload_time = max(offload_times) if offload_times else 0.0
            client_latencies[i] = max(local_comp_time, max_offload_time)

        return client_latencies, es_comp_times

    # =========================================================================
    # Gym 接口实现
    # =========================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 重置分配状态
        self.allocation = np.zeros((self.N, self.num_devices), dtype=np.int32)
        self.current_step = 0

        # 如果 reset 时传入了新的 split_num_list (用于 Curriculum Learning 或不同场景)
        if options and 'split_num_list' in options:
            self.split_num_list = np.array(options['split_num_list'], dtype=np.int32)
            self._precompute_path_info()
        elif self.total_paths == 0 and np.sum(self.split_num_list) > 0:
            self._precompute_path_info()

        return self._get_obs(), {}

    def step(self, action):
        # 1. 执行动作
        if self.current_step >= self.total_paths:
            # 容错：如果越界直接结束
            return self._get_obs(), 0, True, False, {}

        task = self.path_queue[self.current_step]
        client_id = task['client_id']

        # 更新分配矩阵
        self.allocation[client_id, action] += 1

        # 2. 计算奖励
        # 计算当前分配下的 Makespan
        client_latencies, es_times = self._calculate_system_metrics(self.allocation)
        current_makespan = np.max(client_latencies)
        load_std = np.std(es_times)

        # 奖励设计 (与 gymEnv.py 保持一致或优化)
        # 这里使用负的 makespan 增量作为奖励，鼓励快速完成
        # 为了计算增量，需要保存上一步的 makespan。
        # 这里简化：直接给负的绝对值奖励 (Sparse Reward 最后给) 或者 Dense Reward

        # 简单 Dense Reward:
        # reward = -current_makespan * 0.1 - load_std * 1.0
        # 或者使用你提到的 tanh 形式
        reward = -current_makespan

        # 3. 状态更新
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
        """
        生成 Observation。
        为了支持 StableBaselines3，必须返回定长向量。
        """
        if self.current_step < self.total_paths:
            task = self.path_queue[self.current_step]
            # 特征 1: 当前任务特征 (计算量 + 传输量)
            # 归一化处理一下数值，防止过大
            feat_task_comp = task['comp1_size'] + task['comp4_size']
            feat_task_trans = task['trans_total_size']
        else:
            feat_task_comp = 0
            feat_task_trans = 0

        # 特征 2: 当前系统的负载状态
        # 利用核心函数计算当前状态
        client_latencies, es_times = self._calculate_system_metrics(self.allocation)

        # 构造向量
        # [Task_Comp, Task_Trans, Local_Lat(Current_Client?), ES_Times(M), Progress]
        # 注意：Client Latencies 有 N 个，但 Obs 维度不能随 N 变。
        # 策略：只放入 "当前正在分配任务的 Client" 的本地负载，以及 "全局平均 Client 负载"

        current_client_id = self.path_queue[self.current_step][
            'client_id'] if self.current_step < self.total_paths else 0
        current_local_lat = client_latencies[current_client_id]
        avg_client_lat = np.mean(client_latencies)

        progress = self.current_step / self.total_paths if self.total_paths > 0 else 0

        obs = np.concatenate([
            [feat_task_comp / 1e6],  # MFLOPs
            [feat_task_trans / 1e3],  # MB? 假设单位
            [current_local_lat],
            es_times,  # ES 负载是全局共享的，全部放入
            [progress],
            [avg_client_lat]
        ]).astype(np.float32)

        return obs

    # =========================================================================
    # GWO 辅助接口 (Stage 1)
    # =========================================================================
    def simulate_client_time(self, client_index, allocation_vector, split_num):
        """
        供 GWO Splitor 调用。
        计算在单客户端独立视角下（假设其他客户端无负载），采用该 allocation_vector 的时延。

        Args:
            client_index: int
            allocation_vector: (M+1,) 数组
            split_num: int, 当前 GWO 正在尝试的切分数量 (4, 9, 16)
        """
        # 构造一个临时的全局分配矩阵
        temp_alloc = np.zeros((self.N, self.num_devices), dtype=np.int32)
        temp_alloc[client_index] = allocation_vector

        # [修改点 4] 构造临时的 split_num_list
        # 仅给当前 client 赋值 split_num，其他为 0 (表示无负载)
        temp_split_nums = np.zeros(self.N, dtype=np.int32)
        temp_split_nums[client_index] = split_num

        # [修改点 5] 传入 split_nums_override
        client_latencies, _ = self._calculate_system_metrics(temp_alloc, split_nums_override=temp_split_nums)

        # 返回该客户端的时延
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
