# DQN模型部分，包含网络

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  # 第一层，64个神经元
        self.fc2 = nn.Linear(128, 64)  # 第二层，64个神经元
        self.fc3 = nn.Linear(64, action_size)  # 输出层，动作空间的大小

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU激活函数
        x = F.relu(self.fc2(x))  # ReLU激活函数
        return self.fc3(x)  # 返回输出


# 测试函数
if __name__ == '__main__':
    # 初始化模型
    model = QNetwork(10, 5)
    # 随机生成一个输入
    input = torch.randn(10)
    # 输出
    output = model(input)
    print(output)
    # 输出大小
    print(output.size())