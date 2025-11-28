import random

import gymnasium as gym
import numpy as np
from scheduler_1115.envs.Env import TaskAssignmentEnv
from typing import Tuple, List, Dict # 添加必要的类型提示


class GymPPOEnv(gym.Env):
    def __init__(self, ES_list, Client_list, split_num_list, model_type, bandwidth):
        super(GymPPOEnv, self).__init__()
        # 基础参数
        self.N = len(Client_list)  # Client数量
        self.M = len(ES_list)  # ES数量
        self.ES_list = ES_list  # Edge Server列表
        self.Client_list = Client_list  # Client列表
        self.num_devices = self.M + 1  # 设备数量，包括本地设备
        self.model_type = model_type  # 模型类型，不同模型的计算和传输开销不一样
        self.split_num_list = np.array(split_num_list, dtype=np.int32)  # 每个Client的切分数量
        self.batches_list = [client.num_batches for client in Client_list]  # 每个Client的数据批次数量
        self.bandwidth_map = self.init_bandwiths(bandwidth)  # 生成按照 Client-ES 对应的rank的带宽字典
        self.global_bandwidth = bandwidth  # 全局传输速度上限，单位MB/S
        self.bandwidth_matrix = np.zeros((self.N, self.num_devices), dtype=np.float32)  # 将字典转换为矩阵形式，方便索引
        for i, client in enumerate(self.Client_list):
            for j, es in enumerate(self.ES_list):
                self.bandwidth_matrix[i, j + 1] = self.bandwidth_map[(client.rank, es.rank)]
            self.bandwidth_matrix[i, 0] = float('inf')  # 本地带宽视为无限大
        self.devices_speed_map = self.read_speed()  # 读取所有设备的处理速度，存为字典形式
        self.devices_speed_matrix = np.zeros((self.N, self.num_devices), dtype=np.float32)  # 将字典转换为矩阵形式，方便索引
        for i, client in enumerate(self.Client_list):
            for j, es in enumerate(self.ES_list):
                self.devices_speed_matrix[i, j + 1] = self.devices_speed_map[es.rank]
            self.devices_speed_matrix[i, 0] = self.devices_speed_map[client.rank]
        self.client_flops = self.devices_speed_matrix[:, 0]  # Client的处理速度数组
        self.es_flops = self.devices_speed_matrix[0, 1:]  # ES的处理速度数组
        self.device_flops = np.concatenate((self.client_flops, self.es_flops))  # 所有设备的处理速度数组

        self.comp_loads = np.zeros(self.N + self.M, dtype=np.float32)  # 每个设备当前的计算负载
        # self.trans_loads = np.zeros(self.N + self.M, dtype=np.float32)  # 传输负载只算在Client端
        # 传输负载改为 (N, M) 矩阵，行是Client，列是ES。本地设备(0号)不产生传输负载，故只记录M个ES
        self.trans_loads = np.zeros((self.N, self.M), dtype=np.float32)

        # 数据库
        self.database = Database()
        # 估计路径算力开销
        self._precompute_path_costs()
        # 生成路径队列
        self._generate_path_queue()

        # 动作空间：离散，大小 = M + 1  ，新的逻辑下逐条路径进行分配
        # self.action_dim = self.num_devices  # 注意：to_device ≠ from_device
        self.action_dim = self.num_devices  # 注意：to_device ≠ from_device
        self.action_space = gym.spaces.Discrete(self.action_dim)
        # 状态空间：连续，flatten allocation
        state_dim = 0*self.N + 2*self.M + 4  # 当前路径开销+当前传输开销+采样后的时间负载（当前路径所属的client的时间负载和其余ES的时间负载）
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        self.current_step = 0  # 当前步骤
        self.allocation = np.zeros((self.N, self.num_devices), dtype=np.int32)
        self.final_makespan = 0.0

    def init_bandwiths(self, bandwidth=60.0):
        # 生成Client到ES的双向带宽。可以设置为随机值，或者固定值。实验中采用固定值保证可复现，通过调整不同带宽，测试决策器的鲁棒性
        bandwidths = {}
        # 所以先外层循环是Client，内层循环是ES
        for client in self.Client_list:
            for es in self.ES_list:
                bandwidths[(client.rank, es.rank)] = random.uniform(bandwidth, bandwidth)  # 单位MB/S

        return bandwidths

    def read_speed(self):
        # 用键值对的方式存储设备的速度，键是rank，值是处理速度
        devices_speed_map = {}
        # 先读取Client的速度
        for client in self.Client_list:
            devices_speed_map[client.rank] = client.MFLOPs
        # 再读取ES的速度
        for es in self.ES_list:
            devices_speed_map[es.rank] = es.MFLOPs
        return devices_speed_map

    def _precompute_path_costs(self):
        """预计算所有path的计算和传输开销，按照客户端顺序"""
        self.path_comp_costs = []
        self.path_trans_costs = []

        for client_id, s in enumerate(self.split_num_list):
            if s <= 0:
                continue

            b = self.batches_list[client_id]

            # Query database directly
            comp1 = self.database.get_compsize((s, 0), self.model_type)
            comp4 = self.database.get_compsize((s, 3), self.model_type)
            total_comp = (comp1 + comp4) * b
            # 传输量
            trans1 = self.database.get_transsize((s, 0), self.model_type)
            trans2 = self.database.get_transsize((s, 1), self.model_type)
            trans3 = self.database.get_transsize((s, 2), self.model_type)
            trans4 = self.database.get_transsize((s, 3), self.model_type)
            total_trans = (trans1 + trans2 + trans3 + trans4) * b

            self.path_comp_costs.append(total_comp)
            self.path_trans_costs.append(total_trans)

    def _generate_path_queue(self):
        """生成路径队列，按照计算开销从大到小排序"""
        path_data = []
        for client_id, s in enumerate(self.split_num_list):
            if s <= 0:
                continue
            comp_cost = self.path_comp_costs[client_id]
            trans_cost = self.path_trans_costs[client_id]
            for _ in range(s):
                path_data.append((client_id, comp_cost, trans_cost))

        path_data.sort(key=lambda x: x[1], reverse=True)

        self.path_queue = np.array([item[0] for item in path_data], dtype=np.int32)  # 路径id队列
        self.path_comp_array = np.array([item[1] for item in path_data], dtype=np.float32)  # 路径计算开销数组
        self.path_trans_array = np.array([item[2] for item in path_data], dtype=np.float32)  # 路径传输开销数组
        self.total_paths = len(self.path_queue)  # 总路径数量

    def reset(self, *, seed=None, options=None):
        # 处理随机种子
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # 当前分配情况
        # print(self.final_makespan)
        # print(self.allocation)
        self.final_makespan = 0.0
        self.allocation = np.zeros((self.N, self.num_devices), dtype=np.int32)  # 每个Client分配到每个设备的路径数量
        self.comp_loads = np.zeros(self.N + self.M, dtype=np.float32)
        self.trans_loads = np.zeros((self.N, self.M), dtype=np.float32) # 重置为 N*M 零矩阵
        self.time_loads = np.zeros(self.N + self.M, dtype=np.float32)
        self.current_step = 0

        self.total_rewards = 0.0
        # 获取初始观察
        observation = self._get_obs()
        info = {
            "total_paths": self.total_paths,
            "split_num_list": self.split_num_list.copy()
        }

        return observation, info


    def _get_obs(self):
        client_id = self.path_queue[self.current_step] if self.current_step < self.total_paths else 0  # 所属的client
        comp_time = self.comp_loads / self.device_flops  # 全局的计算时间负载 N + M
        current_client_trans_load = self.trans_loads[client_id]
        trans_time_es = current_client_trans_load / self.global_bandwidth
        es_comp_time = comp_time[self.N:]  # 取出 M 个 ES 的计算时间
        es_time_loads = es_comp_time + trans_time_es  # (M,)
        current_client_local_time = comp_time[client_id]  # (1,)
        current_time_loads = np.concatenate(([current_client_local_time], es_time_loads))
        # 4. 获取当前任务的开销
        if self.current_step >= self.total_paths:
            current_comp = 0.0
            current_trans = 0.0
        else:
            current_comp = self.path_comp_array[self.current_step]
            current_trans = self.path_trans_array[self.current_step]

        # 5. 获取当前设备的算力 (1 + M)
        current_client_flop = self.client_flops[client_id]
        current_flops = np.concatenate(([current_client_flop], self.es_flops))
        return np.concatenate([
            [current_comp],  # 1
            [current_trans],  # 1
            current_time_loads,  # 1 + M
            current_flops,  # 1+M
        ]).astype(np.float32)


    def _calculate_makespan(self):
        # 由于真实的全局时延时分阶段计算的，这里需要读取当前的分配矩阵，计算每个客户端的时间开销，再取最大值
        # 注意，这里不能用当前的计算量除以处理速度来估算，因为ES会同时处理其他客户端的任务，所以必须统一时间
        # 需要先获取ES在各个阶段的总计算负载
        client_time_list = []
        client_comp1_list = []
        client_comp4_list = []
        es_comp1_loads = np.zeros(self.M, dtype=np.float32)
        es_comp4_loads = np.zeros(self.M, dtype=np.float32)

        # 在计算ES的计算负载时，要把所有client卸载的路径都加和考虑进去，所以要先算出ES的总计算负载，最后再把各个client的本地负载拼上去
        for client_id, s in enumerate(self.split_num_list):
            comp1 = self.database.get_compsize((s, 0), self.model_type)
            es_comp1_loads = es_comp1_loads + comp1 * self.allocation[client_id, 1:]  # ES部分的计算负载
            # 将es_comp1_loads前面拼接一个client_comp1，得到当前client的comp1负载
            comp4 = self.database.get_compsize((s, 3), self.model_type)
            es_comp4_loads = es_comp4_loads + comp4 * self.allocation[client_id, 1:]  # ES部分的计算负载
        # 拼接client的负载
        for client_id, s in enumerate(self.split_num_list):
            comp1 = self.database.get_compsize((s, 0), self.model_type)
            client_comp1_loads = self.allocation[client_id, 0] * comp1
            full_comp1_loads = np.concatenate(([client_comp1_loads], es_comp1_loads))
            client_comp1_list.append(full_comp1_loads)
            comp4 = self.database.get_compsize((s, 3), self.model_type)
            client_comp4_loads = self.allocation[client_id, 0] * comp4
            full_comp4_loads = np.concatenate(([client_comp4_loads], es_comp4_loads))
            client_comp4_list.append(full_comp4_loads)

        for client_id, s in enumerate(self.split_num_list):
            client_dist = self.allocation[client_id]
            b = self.batches_list[client_id]
            # phase1
            # 传输量包括来回
            trans1 = self.database.get_transsize((s, 0), self.model_type) + self.database.get_transsize((s, 1),
                                                                                                        self.model_type)
            comp_loads_1 = client_comp1_list[client_id]
            # trans_loads_1 = trans1 * client_dist  # 由于异步传输，这里只计算一次即可
            # 由于异步传输，这里多条路径也只有一次传输，所以使用client_dist生成掩码，再乘trans1
            trans_mask1 = np.where(client_dist > 0, 1.0, 0.0)
            trans_loads_1 = trans1 * trans_mask1
            trans_loads_1[0] = 0.0  # 本地设备不计算传输
            time_phase1 = (comp_loads_1 / self.devices_speed_matrix[client_id]) + (
                        trans_loads_1 / self.bandwidth_matrix[client_id])
            time_phase1 = np.max(time_phase1)  # 并行取最大值，再乘以批次数量
            # phase2&3 local
            comp23 = self.database.get_compsize((s, 1), self.model_type) + self.database.get_compsize((s, 2),
                                                                                                      self.model_type)
            time_phase23 = comp23 / self.devices_speed_matrix[client_id, 0]
            # phase4
            trans4 = self.database.get_transsize((s, 2), self.model_type) + self.database.get_transsize((s, 3),
                                                                                                        self.model_type)
            comp_loads_4 = client_comp4_list[client_id]
            # trans_loads_4 = trans4 * client_dist  # 同理
            trans_mask2 = np.where(client_dist > 0, 1.0, 0.0)
            trans_loads_4 = trans4 * trans_mask2 # 同理
            trans_loads_4[0] = 0.0
            time_phase4 = (comp_loads_4 / self.devices_speed_matrix[client_id]) + (
                        trans_loads_4 / self.bandwidth_matrix[client_id])
            time_phase4 = np.max(time_phase4)
            total_time = (time_phase1 + time_phase23 + time_phase4) * b
            client_time_list.append(total_time)
        makespan = max(client_time_list)

        es_total_loads = es_comp1_loads + es_comp4_loads
        es_time_loads = es_total_loads / self.es_flops
        # 计算当前客户端的时间负载，如果是最终，则取最大的客户端
        if self.current_step >= self.total_paths:
            client_total_loads_list = client_comp1_list + client_comp4_list
            # 相加然后只取第一列（客户端）中的最大值
            client_time_loads = client_total_loads_list[:,0] / self.client_flops
            ctl = np.max(client_time_loads)
            time_load_list = np.concatenate(([ctl], es_time_loads))
        else:
            client_id = self.path_queue[self.current_step]
            client_total_loads = client_comp1_list[client_id][0] + client_comp4_list[client_id][0]
            ctl = client_total_loads / self.client_flops[client_id]
            time_load_list = np.concatenate(([ctl], es_time_loads))

        load_std = np.std(time_load_list)

        return makespan, client_time_list, load_std

    def calculate_makespan_for_allocation(self,
                                          allocation_matrix: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算给定全局分配矩阵下的最终 Makespan 和负载标准差。
        此方法不依赖于环境的 current_step 或内部 self.allocation 状态。

        Args:
            allocation_matrix: 全局分配矩阵 (N x (M+1))。
            batches_list: 每个客户端的批次大小列表 (N)。

        Returns:
            Tuple[float, float]: (makespan, load_std)
        """
        # 注意：此方法依赖于以下环境属性：
        # self.M, self.database, self.split_num_list, self.devices_speed_matrix,
        # self.bandwidth_matrix, self.es_flops, self.client_flops

        client_time_list = []
        client_comp1_list = []
        client_comp4_list = []
        es_comp1_loads = np.zeros(self.M, dtype=np.float32)
        es_comp4_loads = np.zeros(self.M, dtype=np.float32)

        # 1. 计算 ES 侧的总计算负载
        for client_id, s in enumerate(self.split_num_list):
            # 获取计算量 comp1 (s, 0)
            comp1 = self.database.get_compsize((s, 0), self.model_type)
            # allocation_matrix[client_id, 1:] 是 ES 侧的分配向量 (长度 M)
            es_comp1_loads += comp1 * allocation_matrix[client_id, 1:]  # ES部分的计算负载

            # 获取计算量 comp4 (s, 3)
            comp4 = self.database.get_compsize((s, 3), self.model_type)
            es_comp4_loads += comp4 * allocation_matrix[client_id, 1:]  # ES部分的计算负载

        # 2. 拼接客户端的负载（用于计算每个客户端的总负载）
        for client_id, s in enumerate(self.split_num_list):
            comp1 = self.database.get_compsize((s, 0), self.model_type)
            # 客户端 Comp1 负载 ( allocation[client_id, 0] 是本地设备是否分配)
            client_comp1_loads = allocation_matrix[client_id, 0] * comp1
            # 拼接本地负载和所有 ES 的 Comp1 负载（长度 M+1）
            full_comp1_loads = np.concatenate(([client_comp1_loads], es_comp1_loads))
            client_comp1_list.append(full_comp1_loads)

            comp4 = self.database.get_compsize((s, 3), self.model_type)
            client_comp4_loads = allocation_matrix[client_id, 0] * comp4
            full_comp4_loads = np.concatenate(([client_comp4_loads], es_comp4_loads))
            client_comp4_list.append(full_comp4_loads)

        # 3. 计算每个客户端的总时延 (Makespan)
        for client_id, s in enumerate(self.split_num_list):
            client_dist = allocation_matrix[client_id]  # (M+1) 分配向量
            b = self.batches_list[client_id]  # 批次数量

            # --- Phase 1 ---
            trans1 = self.database.get_transsize((s, 0), self.model_type) + self.database.get_transsize((s, 1),
                                                                                                        self.model_type)
            comp_loads_1 = client_comp1_list[client_id]
            trans_mask1 = np.where(client_dist > 0, 1.0, 0.0)
            trans_loads_1 = trans1 * trans_mask1
            trans_loads_1[0] = 0.0  # 本地设备不计算传输
            # trans_loads_1 = trans1 * client_dist
            # trans_loads_1[0] = 0.0  # 本地设备不计算传输

            # 时延 = 计算时延 + 传输时延 (元素级计算)
            time_phase1 = (comp_loads_1 / self.devices_speed_matrix[client_id]) + \
                          (trans_loads_1 / self.bandwidth_matrix[client_id])
            time_phase1 = np.max(time_phase1)  # 并行执行，取最大值

            # --- Phase 2 & 3 (Local only) ---
            comp23 = self.database.get_compsize((s, 1), self.model_type) + self.database.get_compsize((s, 2),
                                                                                                      self.model_type)
            time_phase23 = comp23 / self.devices_speed_matrix[client_id, 0]  # 仅使用客户端速度

            # --- Phase 4 ---
            trans4 = self.database.get_transsize((s, 2), self.model_type) + self.database.get_transsize((s, 3),
                                                                                                        self.model_type)
            comp_loads_4 = client_comp4_list[client_id]
            trans_mask2 = np.where(client_dist > 0, 1.0, 0.0)
            trans_loads_4 = trans4 * trans_mask2 # 同理
            trans_loads_4[0] = 0.0
            # trans_loads_4 = trans4 * client_dist
            # trans_loads_4[0] = 0.0  # 本地设备不计算传输

            # 时延 = 计算时延 + 传输时延 (元素级计算)
            time_phase4 = (comp_loads_4 / self.devices_speed_matrix[client_id]) + \
                          (trans_loads_4 / self.bandwidth_matrix[client_id])
            time_phase4 = np.max(time_phase4)

            # 客户端总时延
            total_time = (time_phase1 + time_phase23 + time_phase4) * b
            client_time_list.append(total_time)

        # 4. 最终 Makespan
        makespan = max(client_time_list)

        # 5. 计算负载标准差 (Load Standard Deviation)
        # es_total_loads = es_comp1_loads + es_comp4_loads
        # es_time_loads = es_total_loads / self.es_flops  # ES侧的时间负载

        # 客户端总负载（只取本地设备部分）
        # client_total_loads_list = [c1[0] + c4[0] for c1, c4 in zip(client_comp1_list, client_comp4_list)]
        # client_time_loads = np.array(client_total_loads_list) / self.client_flops

        # 负载标准差只关心客户端中最大的负载和ES的总负载
        # ctl = np.max(client_time_loads)
        # time_load_list = np.concatenate(([ctl], es_time_loads))

        # load_std = np.std(time_load_list)

        return makespan, client_time_list

    def _calculate_system_metrics(self, allocation_matrix, split_nums_override=None):  # 一个和外部调用兼容用的接口，实际上就是调用calculate_makespan_for_allocation
        makespan, client_time_list = self.calculate_makespan_for_allocation(allocation_matrix)
        return makespan, client_time_list

    def _get_reward(self, old_makespan, new_makespan, old_client_time_list, new_client_time_list, old_std, new_std, client_id):

        if self.current_step == 0:
            # reward = -new_makespan * 0.1
            reward = -new_makespan * 0.2
        else:
            # 1. 当前客户端改善（稠密信号）
            client_id = self.path_queue[self.current_step - 1]
            client_improve = -(new_client_time_list[client_id] - old_client_time_list[client_id])
            # 2. 全局 makespan 改善（稀疏信号）
            makespan_improve = -(new_makespan - old_makespan)
            # 3. 负载均衡奖励（鼓励 ES 均衡）
            load_balance_reward = (old_std - new_std) * 300  # 300
            # 4. 全局makespan绝对值（持续奖励，避免由首次分配导致的后续奖励稀疏问题）
            makespan_absolute_reward = -new_makespan * 0.2  # 目前设为0，不启用0.15

            reward = client_improve + makespan_improve + load_balance_reward + makespan_absolute_reward
        return reward



    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        if self.current_step >= self.total_paths:
            observation = self._get_obs()
            reward = 0.0
            terminated = True
            truncated = False
            info = {"makespan": self._calculate_makespan(), "allocation": self.allocation.copy()}
            return observation, reward, terminated, truncated, info

        # 读取当前路径的信息
        client_id = self.path_queue[self.current_step]
        comp_cost = self.path_comp_array[self.current_step]
        trans_cost = self.path_trans_array[self.current_step]

        # 时延要分段计算，先计算每个客户端的独立时间开销，再计算所有客户端的并行时间开销
        # old_makespan, old_client_time_list = self._calculate_makespan() if self.current_step > 0 else 0.0
        old_makespan, old_client_time_list, old_std = self._calculate_makespan()

        # 更新分配矩阵和计算负载
        if action == 0:  # Local device
            global_device_id = client_id
            self.allocation[client_id, 0] += 1
            self.comp_loads[global_device_id] += comp_cost
        else:  # ES device (action 1..M -> ES 0..M-1)
            es_id = action - 1
            global_device_id = self.N + es_id
            self.allocation[client_id, es_id + 1] += 1
            self.comp_loads[global_device_id] += comp_cost
            # 传输时延只计算首次分配的量
            # if self.trans_loads[global_device_id] == 0.0:  # 这里有问题，这个需要按照客户端区分，别的客户端你的传输时延不能相互算，只有计算时延可以这样算
            #     self.trans_loads[global_device_id] += trans_cost

            if self.trans_loads[client_id, es_id] == 0.0:  # 新的矩阵版本的传输负载更新
                self.trans_loads[client_id, es_id] += trans_cost

        # 只卸载es，不卸载client的策略
        # offload_to = action + 1
        # self.allocation[client_id, offload_to] += 1
        # self.comp_loads[self.N + action] += comp_cost
        # self.trans_loads[self.N + action] += trans_cost

        new_makespan, new_client_time_list, new_std = self._calculate_makespan()
        # 测试另一个计算时间是否一致
        # test_makespan, _ = self.calculate_makespan_for_allocation(self.allocation)

        # 计算奖励
        reward = self._get_reward(old_makespan, new_makespan, old_client_time_list, new_client_time_list, old_std, new_std, client_id)
        # reward = -new_makespan

        self.total_rewards += reward

        self.current_step += 1
        terminated = self.current_step >= self.total_paths
        truncated = False  # 如果有最大步数限制，这里设为 True
        obs = self._get_obs()

        if self.current_step >= self.total_paths:
            self.final_makespan = new_makespan

        return obs, reward, terminated, truncated, {
            "step": self.current_step,
            "makespan": new_makespan,
            "client_id": client_id,
            "action": action,
            "allocation": self.allocation.copy(),
            "client_time_list": new_client_time_list,
        }

    def render(self, mode='human'):
        print(f"Step {self.current_step}, Allocation:\n{self.allocation}")


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