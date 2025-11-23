# 本文件定义Client、ES和任务分配环境的Env，调用逻辑是
# 1. 先在main环境中通过torch.dist初始化并连接各个设备到网络中
# 2. 先依次创建各个ES的ESEnv，再创建各个Client的ClientEnv，创建时需要测试和各个ES的联通性和带宽
# 3. 最后创建TaskEnvironment，将ClientEnv和ESEnv注册进环境内，完成初始化
# 4. 环境定期更新，每次更新时，更新ClientEnv和ESEnv的状态

import random
import numpy as np
import torch

from utils_DQN import *


# 环境类，定义任务和设备，计算分配方案的总耗时
# 由于rank和生成任务出去的device_id并不是一个东西，所以每次生成任务出去前需要建立一个映射表
class TaskAssignmentEnv:
    def __init__(self, ES_list, Client_list):
        # 下面是环境的一些必要参数，由外部传入或者初始化得到
        self.split_num_list = None  # 这个是第一阶段得到的每个Client的路径切分数量
        self.batches_list = [client.num_batches for client in Client_list]  # 这个是各个Client本地数据量的列表
        self.ES_list = ES_list
        self.Client_list = Client_list
        self.num_devices = len(ES_list) + 1  # 这个其实不是总设备数，而是每一组Client-ESs的设备数，即ESs+1
        self.devices_speed_map = self.read_speed()  # 读取设备的速度，这个里面是按照rank来的
        self.bandwidths_map = self.init_bandwiths()  # 读取当前Env下的这台Client到域内各个ES的带宽
        self.client_power_list = np.array([self.devices_speed_map[device.rank] for device in self.Client_list])  # 客户端的功率列表
        self.es_power_list = np.array([self.devices_speed_map[device.rank] for device in self.ES_list])  # ES的功率列表
        self.bandwidths_list = [self.bandwidths_map[(client.rank, es.rank)] for client in self.Client_list for es in self.ES_list]
        # self.bandwidths_list = [0.0] + self.bandwidths_list  # 在Client本地也有一个带宽，所以前面加一个0.0

        # 下面是用于迭代过程的一些参数，由于需要在迭代过程中不断更新，所以定义在这里
        self.current_step = 0  # 当前迭代步
        self.previous_allocation = None  # 上一次的分配
        self.last_allocation = None  # 当前的分配
        self.state = None  # 当前的状态
        self.client_speeds = np.array([self.devices_speed_map[device.rank] for device in self.Client_list])
        self.es_speeds = np.array([self.devices_speed_map[device.rank] for device in self.ES_list])
        self.bandwiths_list = [None] * len(Client_list)  # 用于存储每个Client到各个ES的带宽
        self.client_time_list = [None] * len(Client_list)  # 用于存储每个Client的计算时间
        self.es_time_list = [None] * len(Client_list)  # 用于存储每个Client在各个ES上的计算时间
        self.fuse_time_list = [None] * len(Client_list)
        self.done = False

        self.database = Database()  # 数据库，用于存储数据集的传输和计算量
        self.model_type = "cifar100"  # 数据集类型
        # self.model_type = "cifar"  # 数据集类型
        # self.model_type = "fmnist"
        # self.model_type = "fmnist"

    def reset(self, pre_alloc=None, split_num_list=None, renew=False):
        """
        重置环境，初始化状态。
        """
        # 将prealloc按行相加即可得到split_num_list  # pre_alloc就是第一阶段的分配结果X，而split num list就是S
        split_num_list = np.sum(pre_alloc, axis=1) if pre_alloc is not None else split_num_list

        # if renew:
        #     self.renew_all_time(split_num_list)  # 新的算法下，时间必须等到状态确定，算力分配后才能更新，所以这个要去掉

        self.split_num_list = split_num_list  # 这个是第一阶段得到的每个Client的路径切分数量
        self.done = False
        # 平均初始化任务分配
        if pre_alloc is None:
            self.previous_allocation = self.generate_initial_allocation()
        else:
            self.previous_allocation = pre_alloc
        self.last_allocation = self.previous_allocation.copy()  # 当前轮的分配
        self.current_step = 0  # 迭代步，重新设置为0，迭代的停止条件还要设计
        self.state = self.generate_state()
        return self.state

# renew time系列，配合计算time 使用，都需要删掉

    def renew_all_time(self, split_num_list=None):
        # 每次调用前清空这些列表，避免重复添加
        # 初始化列表大小为固定长度，用 None 占位
        num_clients = len(self.Client_list)
        self.client_time_list = [None] * num_clients
        self.fuse_time_list = [None] * num_clients
        self.es_time_list = [None] * num_clients
        self.bandwiths_list = [None] * num_clients

        # 预计算 ES 和 Client 的算力
        self.es_speeds = np.array([self.devices_speed_map[device.rank] for device in self.ES_list])
        self.client_speeds = np.array([self.devices_speed_map[device.rank] for device in self.Client_list])

        # 逐个 client 更新
        for i in range(len(self.Client_list)):
            self.renew_one_time(i, split_num_list[i])

    def renew_one_time(self, client_index, split_num):
        """
        更新一个 client 的时延（包括计算、融合、传输）。
        client_index: int，该 client 在 Client_list 中的索引
        split_num: int，该 client 的路径切分数
        """
        # 获取 client 和相关参数
        client = self.Client_list[client_index]
        client_speed = self.devices_speed_map[client.rank]
        es_speeds = self.es_speeds
        bandwiths = [self.bandwidths_map[(client.rank, es.rank)] for es in self.ES_list]
        self.bandwiths_list.append(bandwiths)

        # client 本地两段的计算时间
        time_list = [
            self.database.get_compsize((split_num, 0), self.model_type) / client_speed,
            self.database.get_compsize((split_num, 3), self.model_type) / client_speed
        ]
        self.client_time_list[client_index] = time_list

        # fuse阶段两个部分的计算时间（固定切分为4）
        fuse_time = [
            self.database.get_compsize((4, 1), self.model_type) / client_speed,
            self.database.get_compsize((4, 2), self.model_type) / client_speed
        ]
        self.fuse_time_list[client_index] = fuse_time

        # 每个 ES 上的前向+反向总时间
        es_time = []
        for j, es in enumerate(self.ES_list):
            time_fb = []

            # 前向
            comptime_f = self.database.get_compsize((split_num, 0), self.model_type) / es_speeds[j]
            transsize_f = (self.database.get_transsize((split_num, 0), self.model_type) +
                           self.database.get_transsize((split_num, 1), self.model_type))
            transtime_f = transsize_f / bandwiths[j]
            time_fb.append(comptime_f + transtime_f)

            # 反向
            comptime_b = self.database.get_compsize((split_num, 3), self.model_type) / es_speeds[j]
            transsize_b = (self.database.get_transsize((split_num, 2), self.model_type) +
                           self.database.get_transsize((split_num, 3), self.model_type))
            transtime_b = transsize_b / bandwiths[j]
            time_fb.append(comptime_b + transtime_b)

            es_time.append(time_fb)

        self.es_time_list[client_index] = es_time

    def generate_state(self):
        """
        生成当前状态，包含静态和动态信息。
        :return: 当前状态 (numpy array)。
        """
        # 将last_allocation转换为张量作为state返回
        return torch.tensor(self.last_allocation, dtype=torch.float32).view(-1)

    def generate_initial_allocation(self, split_num_list=None):
        """
        这是之前在没有第一阶段作为初始分配时的分配算法，简单的平均分配，如果有一阶段决策，则不需要这个
        :return: 初始任务分配 (numpy array)。
        """
        # 生成平均分配列表，C*(M+1)的矩阵
        allocations = []
        for i in split_num_list:
            base_allocation = i // self.num_devices
            allocation = np.full(self.num_devices, base_allocation)
            left = i % self.num_devices
            if left > 0:
                allocation[-left:] += 1  # 平均分配剩余任务，从后往前
            # 将这个分配加入到分配列表中
            allocations.append(allocation)
        # 最后将整个列表转换成np数组
        allocations = np.array(allocations)
        return allocations

    def _final_time(self):
        """
        用于计算分配完成后，在该分配策略下的总耗时。这个函数是直接使用内部的last_allocation来计算的，不接受外部决策
        :param state: 当前状态。
        :return: 总耗时。
        """
        # 用last_allocation计算总耗时
        time_stage1_list = []
        time_stage4_list = []

        # stage1时间
        for i in range(len(self.last_allocation)):
            list1 = []
            list1.append(self.client_time_list[i][0] * self.last_allocation[i][0])
            # 后面列是各个ES的运算耗时
            for j in range(1, len(self.last_allocation[i])):
                list1.append(self.es_time_list[i][j - 1][0] * self.last_allocation[i][j])
            # 然后计算这一行的最大时间作为第一阶段的耗时
            time_stage1_list.append(np.max(list1))

        # stage2时间和stage3时间就是之前的fuse_time_list，不用重复算了

        # stage4时间
        for i in range(len(self.last_allocation)):
            list4 = []
            list4.append(self.client_time_list[i][1] * self.last_allocation[i][0])
            for j in range(1, len(self.last_allocation[i])):
                list4.append(self.es_time_list[i][j - 1][1] * self.last_allocation[i][j])
            # 计算第三阶段的时间
            time_stage4_list.append(np.max(list4))

        # 计算总时间
        fuse_time_list = np.array(self.fuse_time_list)
        global_time_list = np.array(time_stage1_list) + fuse_time_list[:, 0] + fuse_time_list[:, 1] + np.array(
            time_stage4_list)
        # 对应位置乘以batch数
        global_time_list = global_time_list * np.array(self.batches_list)

        return global_time_list

    def get_all_time(self, allocation):

        # 用last_allocation计算总耗时
        time_stage1_list = []
        # time_stage2_list = []
        # time_stage3_list = []  # 这两个时间就是之前的fuse_time_list
        time_stage4_list = []

        # stage1时间
        for i in range(len(allocation)):
            list1 = []
            list1.append(self.client_time_list[i][0] * allocation[i][0])
            # 后面列是各个ES的运算耗时
            for j in range(1, len(allocation[i])):
                list1.append(self.es_time_list[i][j - 1][0] * allocation[i][j])
            # 然后计算这一行的最大时间作为第一阶段的耗时
            time_stage1_list.append(np.max(list1))

        # stage2时间和stage3时间就是之前的fuse_time_list，不用重复算了

        # stage4时间
        for i in range(len(allocation)):
            list4 = []
            list4.append(self.client_time_list[i][1] * allocation[i][0])
            for j in range(1, len(allocation[i])):
                list4.append(self.es_time_list[i][j - 1][1] * allocation[i][j])
            # 计算第三阶段的时间
            time_stage4_list.append(np.max(list4))

        # 计算总时间
        fuse_time_list = np.array(self.fuse_time_list)
        global_time_list = np.array(time_stage1_list) + fuse_time_list[:, 0] + fuse_time_list[:, 1] + np.array(
            time_stage4_list)
        # 对应位置乘以batch数
        global_time_list = global_time_list * np.array(self.batches_list)

        return global_time_list

    def cal_total_time(self, allocation_list, split_num_list, power_allocation):
        # 在这里生成每个客户端时延计算所需的数据列表
        client_time_list = np.zeros(len(allocation_list))
        for i in range(len(allocation_list)):
            allocation = allocation_list[i]
            split_num = split_num_list[i]
            power = power_allocation[i]
            # bandwidth list是一个n*m长度的一维列表，所以对单个客户端来说，每次从中截取出自己的m个ES的带宽，需要按照i来截取
            bandwidth = self.bandwidths_list[i * len(self.ES_list): i * len(self.ES_list) + len(self.ES_list)]
            bandwidth = [0.0] + bandwidth  # 在Client本地也有一个带宽，所以前面加一个0.0
            client_time_list[i] = self.cal_client_time(allocation, split_num, power, bandwidth)
        # 计算总时间，这里是需要考虑batch数量
        total_time_list = client_time_list * self.batches_list
        return total_time_list



    def cal_client_time(self, allocation, split_num, power, bandwidth):
        """
        计算单个Client的总耗时
        :param allocation: 当前状态。
        :param split_num: 当前Client的路径切分数量。
        :param computation: 当前Client的计算量。
        :param transmission: 当前Client的传输量。
        :return: 总耗时。
        """
        # 阶段1耗时，客户端本地和ES的并行计算取较大值
        stage1_comp_size = self.database.get_compsize((split_num, 0), self.model_type)
        stage1_trans_size = self.database.get_transsize((split_num, 0), self.model_type) + self.database.get_transsize((split_num, 1), self.model_type)
        client_stage1 = allocation[0] * stage1_comp_size / power[0]
        # es的传输时延只要记录1条路径的，因为是边传边计算，只有第一次发送和最后一次传输需要被计算
        es_stage1_list = []
        for i in range(1, len(allocation)):
            if allocation[i] < 1:  # 如果分配量小于0.001，则不计算该ES的耗时
                es_stage1_list.append(0.0)
                continue
            es_stage1 = stage1_trans_size / bandwidth[i]
            es_stage1 += stage1_comp_size * allocation[i] / power[i] if power[i] > 0.001 else 0.0
            es_stage1_list.append(es_stage1)
        # 拼接
        stage1_list = [client_stage1] + es_stage1_list
        # 取最大值作为阶段1的耗时
        time_stage1 = np.max(stage1_list)
        # 阶段2+3耗时
        stage23_comp_size = self.database.get_compsize((split_num, 1), self.model_type) + self.database.get_compsize((split_num, 2), self.model_type)
        time_stage23 = stage23_comp_size / power[0]  # 客户端本地计算
        # 阶段4耗时，客户端本地和ES的并行计算取较大值
        stage4_comp_size = self.database.get_compsize((split_num, 3), self.model_type)
        stage4_trans_size = self.database.get_transsize((split_num, 2), self.model_type) + self.database.get_transsize((split_num, 3), self.model_type)
        client_stage4 = allocation[0] * stage4_comp_size / power[0]
        es_stage4_list = []
        for i in range(1, len(allocation)):
            if allocation[i] < 1:  # 如果分配量小于0.001，则不计算该ES的耗时
                es_stage1_list.append(0.0)
                continue
            es_stage4 = stage4_trans_size / bandwidth[i]
            es_stage4 += stage4_comp_size * allocation[i] / power[i] if power[i] > 0.001 else 0.0
            es_stage4_list.append(es_stage4)
        stage4_list = [client_stage4] + es_stage4_list
        time_stage4 = np.max(stage4_list)

        # 计算总时间
        global_time = time_stage1 + time_stage23 + time_stage4

        return global_time


    def get_all_load(self, allocation):
        """
        用于计算分配完成后，在该分配策略下的总耗时。这个函数是直接使用内部的last_allocation来计算的，不接受外部决策
        :param allocation: 当前状态。
        :return: 总耗时。
        """
        stage1_load_list = []
        stage4_load_list = []
        # stage1时间
        for i in range(len(allocation)):
            list1 = []
            list1.append(self.client_time_list[i][0] * allocation[i][0])
            # 后面列是各个ES的运算耗时
            for j in range(1, len(allocation[i])):
                list1.append(self.es_time_list[i][j - 1][0] * allocation[i][j])

            stage1_load_list.append(list1)

        # stage4时间
        for i in range(len(allocation)):
            list4 = []
            list4.append(self.client_time_list[i][1] * allocation[i][0])
            for j in range(1, len(allocation[i])):
                list4.append(self.es_time_list[i][j - 1][1] * allocation[i][j])
            stage4_load_list.append(list4)

        global_load = np.array(stage1_load_list) + np.array(stage4_load_list)
        global_load = global_load * np.array(self.batches_list)[:, np.newaxis]
        return global_load

    def get_alles_time(self, allocation):
        """
        用于计算分配完成后，在该分配策略下的es负载
        :param allocation: 当前状态。
        :return: 总耗时。
        """
        # 用last_allocation计算总耗时
        time_stage1_list = []
        # time_stage2_list = []
        # time_stage3_list = []  # 这两个时间就是之前的fuse_time_list
        time_stage4_list = []

        # stage1时间
        for i in range(len(allocation)):
            list1 = []
            # 后面列是各个ES的运算耗时
            for j in range(1, len(allocation[i])):
                list1.append(self.es_time_list[i][j - 1][0] * allocation[i][j])
            time_stage1_list.append(list1)

        # stage4时间
        for i in range(len(allocation)):
            list4 = []
            for j in range(1, len(allocation[i])):
                list4.append(self.es_time_list[i][j - 1][1] * allocation[i][j])
            time_stage4_list.append(list4)

        global_time_list = np.array(time_stage1_list) + np.array(time_stage4_list)
        # 对应位置乘以batch数
        global_time_list = global_time_list * np.array(self.batches_list)[:, np.newaxis]

        return global_time_list

    def get_client_time(self, allocation, index):
        """
        这个可以用于外部调用，根据外部传入的分配进行计算，不过只接受单一客户端，主要给Splitor使用
        :param state: 当前状态。
        :return: 总耗时。
        """
        # 阶段1耗时
        list1 = []
        list1.append(self.client_time_list[index][0] * allocation[0])
        # 后面列是各个ES的运算耗时
        for j in range(1, len(allocation)):
            list1.append(self.es_time_list[index][j - 1][0] * allocation[j])
        # 然后计算这一行的最大时间作为第一阶段的耗时
        time_stage1 = np.max(list1)
        # 阶段2耗时
        time_stage2 = self.fuse_time_list[index][0]
        # 阶段3耗时
        time_stage3 = self.fuse_time_list[index][1]
        # 阶段4耗时
        list4 = []
        list4.append(self.client_time_list[index][1] * allocation[0])
        for j in range(1, len(allocation)):
            list4.append(self.es_time_list[index][j - 1][1] * allocation[j])
        time_stage4 = np.max(list4)
        # 计算总时间
        global_time = (time_stage1 + time_stage2 + time_stage3 + time_stage4) * self.batches_list[index]

        return global_time

    @staticmethod
    def sigmoid_reward(time_diff, scale=0.1):
        return 2 / (1 + np.exp(-scale * time_diff)) - 1

    def get_computation_list(self, allocation, split_num_list):
        """
        ES是并行的执行所有client的，但是总算力是一定的，所以被分配的路径越多，计算速度是越慢的。
        每轮根据每个Client对其的卸载量更新一次算力列表，算力列表用于后续的算力计算。
        :param allocation: 当前状态。
        :return: None
        """
        # 生成一个和allocation一样大小的列表，用于存储每个Client在每个ES上的计算量
        computation_list = np.zeros_like(allocation, dtype=float)
        # 按列遍历allocation
        for i in range(len(allocation[0])):
            for j in range(len(allocation)):
                # 先统计当前列的总分配数量，用分配数乘以当前分割下的路径计算量
                computation_list[j][i] = allocation[j][i] * self.database.get_compsize((split_num_list[j], 0), self.model_type)
        # 然后根据算力量列表，将ES的算力分配给各个Client
        computation_allocation = np.zeros_like(allocation, dtype=float)
        # computation_list按列求和，得到每个ES将要计算的总计算量
        total_computation = np.sum(computation_list, axis=0)
        for i in range(len(allocation[0])):
            for j in range(len(allocation)):
                if i == 0:
                    # 客户端是本地计算，所以不需要分配算力，直接复制客户端的本地算力
                    computation_allocation[j][i] = self.client_speeds[j]
                else:
                    # 其他列是ES，根据每个客户端的分配量和总计算量来分配算力
                    ratio = computation_list[j][i] / total_computation[i] if total_computation[i] > 0.001 else 1.0
                    computation_allocation[j][i] = ratio * self.es_speeds[i - 1]

        return computation_list, computation_allocation

    def get_all_time_new(self, pre_alloc):
        split_num_list = np.sum(pre_alloc, axis=1)  # 计算每个Client的路径切分数量
        # 如果split_num_list有任何一个是0，则有误，打印检查
        if np.any(split_num_list == 0):
            print("Error: split_num_list contains zero values, check pre_alloc:", pre_alloc)
            return None
        # 首先根据当前的分配状态确定ES分配给各个Client的算力，假设带宽由Client的发射侧决定，接受不受影响，因此不考虑带宽的均分，只考虑计算
        comp_amount, computation_alloc = self.get_computation_list(pre_alloc, split_num_list)
        # 计算当前分配的总耗时
        total_time = self.cal_total_time(pre_alloc, split_num_list, computation_alloc)

        return total_time

    def step(self, action, done):
        self.current_step += 1
        if self.current_step >= 16:
            done = True
            self.current_step = 0

        # 解析动作
        # action 是一组调整动作的张量，先把每个元素整除num_devices，得到from_devices_list，然后模num_devices得到to_devices_list
        from_devices_list = action // self.num_devices
        to_devices_list = action % self.num_devices
        # 在上一次分配的基础上进行调整得到新的状态s'，但同时也要记录老的状态，不然不好计算前后的差值来得到奖励
        new_allocation = self.last_allocation.copy()
        for i in range(len(from_devices_list)):  # 状态转移
            new_allocation[i][from_devices_list[i]] -= 1
            new_allocation[i][to_devices_list[i]] += 1

        # 下面开始计算reward
        # 首先检查新状态的合法情况，对于每个Client，都要检查分配数量是否有负的
        if np.min(new_allocation) < 0:  # 防止意外，实际上不会触发
            return self.generate_state(), np.float64(-0.6), done

        # =========================奖励函数计算开始==========================

        # 首先根据当前的分配状态确定ES分配给各个Client的算力，假设带宽由Client的发射侧决定，接受不受影响，因此不考虑带宽的均分，只考虑计算
        old_comp_amount, old_computation_alloc = self.get_computation_list(self.last_allocation, self.split_num_list)
        new_comp_amount, new_computation_alloc = self.get_computation_list(new_allocation, self.split_num_list)

        # 计算当前分配的总耗时
        old_total_time = self.cal_total_time(self.last_allocation, self.split_num_list, old_computation_alloc)
        new_total_time = self.cal_total_time(new_allocation, self.split_num_list, new_computation_alloc)

        time_diff = old_total_time - new_total_time  # 计算时间差
        time_better_reward = np.tanh(time_diff / 10.0)

        reward = time_better_reward

        # # 1.客户端级的时间奖励
        # old_alloc_time = self.get_all_time(self.last_allocation)
        # new_alloc_time = self.get_all_time(new_allocation)
        # time_diff = old_alloc_time - new_alloc_time  # 修改为客户端级的时间奖励
        # time_better_reward = np.tanh(time_diff / 5.0)
        #
        # # 2.服务器负载均衡奖励
        # # 由于用时是分阶段的，所以ES的负载需要独立于Client被提取出来
        # # 提取ES时延，作为负载看待
        # old_server_time = self.get_alles_time(self.last_allocation)
        # new_server_time = self.get_alles_time(new_allocation)
        # # 每一列相加，代表单个服务器的总负载
        # old_load = np.sum(old_server_time, axis=0)
        # new_load = np.sum(new_server_time, axis=0)
        # # 求旧方差和新的方差
        # old_std = np.std(old_load)
        # new_std = np.std(new_load)
        # # 计算方差之差
        # std_diff = old_std - new_std
        # std_better_reward = np.tanh(std_diff / 10.0)

        # 3. 全局时间惩罚（可选）
        # old_max_time = np.max(old_alloc_time)
        # new_max_time = np.max(new_alloc_time)
        # max_time_diff = old_max_time - new_max_time
        # max_time_penalty = np.tanh(max_time_diff / 10.0)  # 更大范围，惩罚激增

        # reward = 0.2 * time_better_reward + 0.8 * std_better_reward
        # reward = std_better_reward

        # 奖励函数计算过程

        # 更新环境信息
        self.previous_allocation = self.last_allocation.copy()
        self.last_allocation = new_allocation.copy()
        self.state = self.generate_state()

        return self.state, reward, done

    # 任务生成
    def init_bandwiths(self):
        # 生成Client到ES的双向带宽。可以设置为随机值，或者固定值。实验中采用固定值保证可复现，通过调整不同带宽，测试决策器的鲁棒性
        bandwidths = {}
        # 所以先外层循环是Client，内层循环是ES
        for client in self.Client_list:
            for es in self.ES_list:
                bandwidths[(client.rank, es.rank)] = random.randint(20, 20)  # 单位MB/S
                # bandwidths[(client.rank, es.rank)] = random.uniform(1.25, 1.25)  # 单位MB/S
                # bandwidths[(client.rank, es.rank)] = random.uniform(2.5, 2.5)  # 单位MB/S
                # bandwidths[(client.rank, es.rank)] = random.uniform(3.75, 3.75)  # 单位MB/S
                # bandwidths[(client.rank, es.rank)] = random.uniform(5.0, 5.0)  # 单位MB/S
                # bandwidths[(client.rank, es.rank)] = random.uniform(6.25, 6.25)  # 单位MB/S
                # bandwidths[(client.rank, es.rank)] = random.uniform(7.5, 7.5)  # 单位MB/S
                # bandwidths[(client.rank, es.rank)] = random.uniform(8.75, 8.75)  # 单位MB/S
                # bandwidths[(client.rank, es.rank)] = random.uniform(10.0, 10.0)  # 单位MB/S
                # bandwidths[(client.rank, es.rank)] = random.uniform(11.25, 11.25)  # 单位MB/S
                # bandwidths[(client.rank, es.rank)] = random.uniform(12.5, 12.5)  # 单位MB/S
                # bandwidths[(client.rank, es.rank)] = random.uniform(13.75, 13.75)  # 单位MB/S
                # bandwidths[(client.rank, es.rank)] = random.uniform(15.0, 15.0)  # 单位MB/S
                # bandwidths[(client.rank, es.rank)] = random.uniform(16.25, 16.25)  # 单位MB/S
                # bandwidths[(client.rank, es.rank)] = random.uniform(17.5, 17.5)  # 单位MB/S
                # bandwidths[(client.rank, es.rank)] = random.uniform(18.75, 18.75)  #  单位MB/S
                # bandwidths[(client.rank, es.rank)] = random.uniform(20.0, 20.0)  # 单位MB/S

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

        # 这个用于计算没有分割的模型的传输和计算用时
        self.nonsplit_transmission_database = {}


    def get_transsize(self, key_tuple, database_type):
        if database_type == "cifar":
            return self.cifar_transmission_database[key_tuple]
        elif database_type == "fmnist":
            return self.fmnist_transmission_database[key_tuple]
        elif database_type == "cifar100":
            return self.cifar100_transmission_database[key_tuple]

    def get_compsize(self, key_tuple, database_type):
        if database_type == "cifar":
            return self.cifar_computation_database[key_tuple]
        elif database_type == "fmnist":
            return self.fmnist_computation_database[key_tuple]
        elif database_type == "cifar100":
            return self.cifar100_computation_database[key_tuple]


