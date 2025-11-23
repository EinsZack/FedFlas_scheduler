# 工具函数
import random

import numpy as np
import torch


# 测试client和es之间是否可以联通，如果可以，返回带宽值
def test_bandwith(client_rank, es_rank):
    # 由于是模拟，默认都返回200MB/s
    # TODO:后续可以改为真实的带宽测试函数，返回真实的带宽值
    return 200


def cal_transform_time(task_size, bandwidth):
    # 计算传输时间
    transform_time = task_size / bandwidth
    # 其中第一个由于是
    return


def cal_compute_time(task_size, device_speed):
    # 计算计算时间
    return task_size / device_speed


def generate_initial_allocation(N, M):
    # Step 1: Calculate base allocation
    base_allocation = N // M
    remainder = N % M

    # Step 2: Create allocation list
    allocation = np.full(M, base_allocation, dtype=int)  # Create a numpy array with base_allocation

    # Step 3: Distribute the remainder to the first 'remainder' devices
    allocation[:remainder] += 1

    return allocation


def allocate_tasks(ratio, N):
    ratio = np.array(ratio)  # 将输入的比例转换为numpy数组
    M = len(ratio)

    # 计算每个设备初步分配的任务数量
    tasks = np.round(ratio * N).astype(int)

    # 计算初步分配的任务总数和目标任务数 N 的差值
    total_assigned = np.sum(tasks)
    diff = N - total_assigned

    # 调整分配任务，确保总任务数等于 N
    if diff != 0:
        # 找到差值最大的设备进行调整
        idx = np.argsort(ratio)[::-1]  # 按比例从大到小排序设备索引

        # 对差值进行调整
        for i in range(abs(diff)):
            if diff > 0:
                # 如果差值是正的，增加任务数
                tasks[idx[i % M]] += 1
            elif diff < 0:
                # 如果差值是负的，减少任务数
                tasks[idx[i % M]] -= 1

    return tasks


def random_action_generator(num_devices):
    # 随机生成一组分配调整比例，长度为num_devices
    # 每个值可能为0、-1、+1三种情况

    return np.random.randint(-1, 2, num_devices)


def update_allocation_with_constraints(last_allocation, actions, total_tasks):
    """
    更新任务分配，使其满足以下约束：
    1. 分配非负
    2. 总分配数量等于 total_tasks

    参数：
    - last_allocation: 上一轮的任务分配数量
    - actions: 每个设备的动作 (-1, 0, +1)
    - total_tasks: 总任务数量

    返回：
    - new_allocation: 修正后的任务分配数量
    """
    num_devices = len(actions)

    new_allocation = last_allocation + actions  # 执行动作

    # 2. 修正非负性
    min_val = np.min(new_allocation)
    if min_val < 0:
        new_allocation -= min_val  # 平移至非负

    # 3. 调整总任务数
    current_sum = np.sum(new_allocation)
    if current_sum != total_tasks:
        new_allocation = new_allocation * (total_tasks / current_sum)  # 归一化调整

    # 4. 舍入到整数并确保总和仍为 total_tasks
    new_allocation = np.round(new_allocation).astype(int)
    # diff = total_tasks - np.sum(new_allocation)
    # while diff != 0:
    #     if diff > 0:
    #         new_allocation[np.argmin(new_allocation)] += 1  # 任务不足，增加
    #         diff -= 1
    #     elif diff < 0:
    #         new_allocation[np.argmax(new_allocation)] -= 1  # 任务超限，减少
    #         diff += 1

    return new_allocation


import torch
import heapq


def adjust_actions(q_values):
    """
    动态调整动作以满足和为0的约束，使用优先队列优化流程。

    :param q_values: torch.Tensor, 形状为 (M, 3)，每个机器的Q值。
    :return: torch.Tensor, 调整后的动作向量。
    """
    M = q_values.shape[0]
    # 动作空间 {+1, 0, -1}
    actions = torch.tensor([-1, 0, 1], device=q_values.device)

    # 初步选择 Q 值最大的动作
    initial_indices = torch.argmax(q_values, dim=1)
    action_vector = initial_indices - 1

    # 计算初始和
    S = torch.sum(action_vector).item()

    if S == 0:
        return action_vector  # 已满足约束，直接返回

    # 优先队列初始化
    # (差值, 动作索引, 当前动作值)
    priority_queue = []
    for i in range(M):
        if action_vector[i] == -1:
            diff = q_values[i, 1] - q_values[i, 0]
        elif action_vector[i] == 0:
            diff = q_values[i, 2] - q_values[i, 1]
        else:  # action_vector[i] == 1
            diff = float('inf')  # +1无法再增加，不需要加入队列
        heapq.heappush(priority_queue, (diff, i, action_vector[i]))

    # 调整逻辑
    while S != 0:
        # 取出当前差值最小的动作
        _, idx, current_action = heapq.heappop(priority_queue)

        if S > 0:  # 当前和过大，需要减少
            if current_action == 1:  # +1 → 0
                action_vector[idx] -= 1
                S -= 1
                # 重新计算差值，并将调整后的动作重新加入队列
                new_diff = q_values[idx, 1] - q_values[idx, 0]
                heapq.heappush(priority_queue, (new_diff, idx, 0))
            elif current_action == 0:  # 0 → -1
                action_vector[idx] -= 1
                S -= 1
        elif S < 0:  # 当前和过小，需要增加
            if current_action == -1:  # -1 → 0
                action_vector[idx] += 1
                S += 1
                # 重新计算差值，并将调整后的动作重新加入队列
                new_diff = q_values[idx, 2] - q_values[idx, 1]
                heapq.heappush(priority_queue, (new_diff, idx, 0))
            elif current_action == 0:  # 0 → +1
                action_vector[idx] += 1
                S += 1

    return action_vector


def generate_zero_sum_tensor(M):
    if M < 1:
        raise ValueError("M must be greater than 0")

    # 初始化数组
    result = [0] * M

    # 分配正负值
    half = M // 2
    result[:half] = [1] * half
    result[half:2 * half] = [-1] * half

    # 如果 M 为奇数，剩余一个元素保持为 0

    # 随机打乱数组
    random.shuffle(result)

    # 转换为张量
    tensor = torch.tensor(result, dtype=torch.int)

    return tensor
