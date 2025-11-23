# 本文件是FedServer的代码，也是整个DQN模型的Agent
import numpy
import torch
import replay_buffer as rb
import model
import torch.nn as nn
import random
import numpy as np

from scheduler_1115.utils_DQN import *


# TODO:把Fedserver做到Env内部
class FedServerAgent:  # 其实就是一个DQN实体
    def __init__(self, rank, state_size, action_size, path_num_list, client_num, es_num, env):
        self.rank = rank  # 设备id
        self.state_size = state_size
        self.action_size = action_size
        self.path_num_list = path_num_list
        self.client_num = client_num
        self.es_num = es_num
        self.env = env

        self.replay_buffer = rb.ReplayBuffer(1000)  # 初始化经验回放缓冲区
        # DQN超参数
        self.gamma = 0.9  # discount rate
        self.learning_rate = 0.001
        self.tau = 0.01
        self.tau_max = 1.0
        self.tau_min = 0.1
        self.tau_decay = (self.tau_max - self.tau_min) / 4000

        # 初始化DQN模型
        self.q_network = model.QNetwork(state_size, action_size)
        self.target_network = model.QNetwork(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        # 连接局域网内的各个设备

    def select_action_tau(self, state, step, istrain = True):

        last_allocation = state.reshape(self.client_num, self.es_num + 1)  # 每行代表一个客户端的分配情况
        # 计算所有设备负载（客户端 + ES）
        device_loads = self.env.get_all_load(last_allocation)

        # 构造掩码
        from_mask = torch.zeros((self.client_num, self.es_num + 1))  # [3, 6]
        for c in range(self.client_num):
            # 合法设备（allocation > 0）
            valid_devices = np.where(last_allocation[c].cpu().numpy() > 0)[0]
            if len(valid_devices) == 0:
                raise ValueError(f"No valid from_device for client {c}, allocation: {last_allocation[c]}")
            # 高负载设备（前 k）
            loads = device_loads[c][valid_devices]
            high_load_indices = valid_devices[np.argsort(loads)[-2:]]  # 最高 2 个
            from_mask[c, high_load_indices] = 1.0
        from_mask = from_mask.unsqueeze(2).repeat(1, 1, self.es_num + 1)  # [3, 6, 6]

        to_mask = torch.zeros((self.client_num, 1, self.es_num + 1))  # [3, 1, 6]
        es_t = self.env.get_alles_time(last_allocation)
        es_loads = np.sum(es_t, axis=0)  # [6]
        # m = []
        candidate_device_list = []
        for c in range(self.client_num):
            k = int(sum(last_allocation[c]).cpu())
            a = min(self.es_num % k, k % self.es_num)
            a = max(a, 2)
            candidate_es = np.argsort(es_t[c])[:a]  # 负载最小的设备
            candidate_device_list.append(candidate_es)

            # 对全局负载进行排序
            candidate_loads = [es_loads[j] for j in candidate_es]
            selected_indices = np.argsort(candidate_loads)[:2]
            selected_es = candidate_es[selected_indices] + 1  # 这里+1才能对应到ES，因为0号位是客户端
            to_mask[c, 0, selected_es] = 1.0
        to_mask = to_mask.repeat(1, self.es_num + 1, 1)  # [3, 6, 6]
        mask = from_mask * to_mask  # [3, 6, 6]

        # 计算 Q 值
        state = torch.FloatTensor(state)
        with torch.no_grad():
            q_values = self.q_network(state).view(self.client_num, self.es_num + 1, self.es_num + 1)  # [3, 6, 6]
        q_values = q_values * mask
        q_values = torch.where(mask == 0, torch.tensor(-1e10, dtype=torch.float32), q_values)

        if istrain:
            # Softmax 探索
            self.tau = max(self.tau_min, self.tau_max - step * self.tau_decay)
            probs = torch.softmax(q_values.view(self.client_num, -1) / self.tau, dim=1)  # [3, 36]
            actions = torch.multinomial(probs, 1).squeeze(1)  # [3]
        else:
            # 贪婪选择
            actions = torch.argmax(q_values.view(self.client_num, -1), dim=1)
        return actions

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

    # 设置从replay_buffer采样的batch_size，进行经验回放学习
    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        q_values = self.q_network(states)
        # 假设 actions 的形状是 [batch_size, 3]
        q_values = q_values.view(batch_size, self.client_num, -1)
        current_q_values = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)  # 从每个小动作的 36 个动作中选出对应的 Q 值
        # current_q_values = torch.mean(current_q_values, dim=1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.view(batch_size, self.client_num, -1)  # [batch_size, client_num, 36]
            # 动作掩码
            next_allocation = next_states.view(batch_size, self.client_num,
                                               self.es_num + 1)  # [batch_size, client_num, 6]
            from_mask = torch.where(next_allocation > 0, 1.0, 0.0)  # [64, 3, 6]
            from_mask = from_mask.unsqueeze(3).repeat(1, 1, 1, self.es_num + 1).view(batch_size, self.client_num,
                                                                                     -1)  # [64, 3, 36]
            next_q_values = next_q_values * from_mask
            next_q_values = torch.where(from_mask == 0, torch.tensor(-1e10, dtype=torch.float32),
                                        next_q_values)
            max_next_q_values = next_q_values.max(dim=2)[0]  # [64, 3]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones.unsqueeze(1))  # [64, 3]
            target_q_values = torch.clamp(target_q_values, -100, 100)

        loss = self.criterion(current_q_values, target_q_values)

        # 更新 Q 网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()

        return loss, self.tau

    def soft_update(self):
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def infer_task_allocation(self, env, pre_alloc, max_steps=16):
        """
        推理阶段任务分配，记录 16 步指标，选择最优分配。

        Args:
            self: 训练好的 FedServerAgent 实例
            env: Environment 实例
            pre_alloc: 初始分配，形状 [client_num, es_num + 1]，例如 [3, 6]
            max_steps: 最大步数（默认 16）

        Returns:
            final_allocation: 最优任务分配，形状 [3, 6]
            final_metrics: 最优步的指标（variance, max_time, es2_tasks）
        """
        state = env.reset(pre_alloc=pre_alloc, split_num_list=None, renew=True)
        done = False
        step = 0
        allocation_history = []
        allocation_history.append(pre_alloc)
        metrics_history = []
        init_max_time = np.max(env.get_all_time_new(pre_alloc))
        init_std = np.std(env.get_alles_time(pre_alloc))
        p = 1.1
        init_metrics = {
            "variance": init_std,
            "max_time": init_max_time,
        }
        metrics_history.append(init_metrics)

        # 运行 16 步或直到 done
        while not done and step < max_steps:
            actions = self.select_action_tau(state, step, istrain=False)
            next_state, reward, done = env.step(actions, done)
            allocation = env.last_allocation.copy()
            es_loads = env.get_alles_time(allocation)
            max_time = np.max(env.get_all_time_new(allocation))
            metrics = {
                "variance": np.std(es_loads),
                "max_time": max_time,
            }
            allocation_history.append(allocation)
            metrics_history.append(metrics)
            print(f"Step {step + 1}, Actions: {actions.cpu().numpy()}, "
                  f"Variance: \n{metrics['variance']:.8f}, Max Time: {metrics['max_time']:.8f}")
            state = next_state
            step += 1

        # 选择最优分配
        time_upbound = init_max_time * p
        valid_indices = [
            i for i in range(len(metrics_history))
            if metrics_history[i]["max_time"] <= time_upbound
        ]
        if valid_indices:
            best_idx = min(valid_indices,
                           key=lambda i: (metrics_history[i]["max_time"],
                                          metrics_history[i]["variance"]))
        else:
            best_idx = 0  # 初始分配
        final_allocation = allocation_history[best_idx]  # +1 因为初始状态在 history 中
        final_metrics = metrics_history[best_idx]

        print("\nFinal Allocation (Best Step):")
        print(final_allocation)
        print("Final Metrics:")
        print(f"Variance: {final_metrics['variance']:.8f}")
        print(f"Max Time: {final_metrics['max_time']:.8f}")

        return final_allocation, final_metrics