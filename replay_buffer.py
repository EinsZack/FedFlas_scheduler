# 经验回放缓冲区

import random
import numpy as np
import torch

"""
这是记忆缓冲，这里存储的每条记忆包括完整的决策包括其前后状态，奖励等信息
一条记忆的格式为 (state, action, reward, next_state, done)
"""


class ReplayBuffer:
    def __init__(self, capacity, done_ratio=0.1):
        self.done_ratio = done_ratio  # 已完成的经历的比例
        self.capacity = capacity  # 缓冲区的最大容量
        self.done_cap = int(capacity * done_ratio)  # 已完成的经历的最大容量
        self.not_done_cap = capacity - self.done_cap  # 未完成的经历的最大容量
        self.done_buffer = []  # 已经完成的经历
        self.nodone_buffer = []  # 未完成的经历
        self.done_position = 0  # 已经完成经历的存储位置
        self.nodone_position = 0  # 未完成经历的存储位置

    def add(self, experience):
        # 检查该记忆是done还是nodone
        if experience[4]:
            if len(self.done_buffer) < self.done_cap:
                self.done_buffer.append(experience)  # 如果缓冲区未满，直接添加
            else:
                self.done_buffer[self.done_position] = experience
            self.done_position = (self.done_position + 1) % self.done_cap  # 更新存储位置
        else:
            if len(self.nodone_buffer) < self.not_done_cap:
                self.nodone_buffer.append(experience)
            else:
                self.nodone_buffer[self.nodone_position] = experience
            self.nodone_position = (self.nodone_position + 1) % self.not_done_cap

    def sample(self, batch_size):
        """随机采样一批经历"""
        # 一开始可能会有done经验不足以采样的情况，此时尽可能采够done经验，不够的用nodone经验补充
        if len(self.done_buffer) < int(batch_size * self.done_ratio):
            done_batch = self.done_buffer
            undone_batch = random.sample(self.nodone_buffer, batch_size - len(self.done_buffer))
        else:
            done_batch = random.sample(self.done_buffer, int(batch_size * self.done_ratio))
            undone_batch = random.sample(self.nodone_buffer, batch_size - int(batch_size * self.done_ratio))
        batch = done_batch + undone_batch
        random.shuffle(batch)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        # 动作需要转换成整数张量
        actions = torch.stack(actions).to(torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """返回当前缓冲区的经历数量"""
        return len(self.done_buffer) + len(self.nodone_buffer)

    def get_done_num(self):
        return len(self.done_buffer)

    def get_nodone_num(self):
        return len(self.nodone_buffer)

    def is_done_num_enough(self, batch_size):
        return len(self.done_buffer) >= int(batch_size * self.done_ratio)


# 测试ReplayBuffer
if __name__ == "__main__":
    buffer = ReplayBuffer(capacity=1000)

    # 添加一些测试经历
    for i in range(120):
        experience = (i, i + 1, i + 2, i + 3, i % 2)  # 示例经历 (state, action, reward, next_state, done)
        buffer.add(experience)

    print(f"Buffer size: {len(buffer)}")  # 打印缓冲区的大小
    sample = buffer.sample(10)  # 随机采样10个经历
    print("Sampled experiences:", sample)  # 打印采样的经历
