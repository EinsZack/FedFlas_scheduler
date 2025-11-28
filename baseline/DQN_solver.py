import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os


class FixedSplitIterativeEnv(gym.Env):
    """
    适配 DQN 的固定切分迭代环境。
    修改点：移除了 split_num 的调整动作，Agent 只负责微调卸载分配（Allocation）。
    """

    def __init__(self, task_env, max_steps=200):
        super().__init__()
        self.task_env = task_env
        self.N = task_env.N
        self.M = task_env.M
        self.num_devices = task_env.num_devices  # M+1
        self.max_steps = max_steps

        self.split_idx_map = {0: 4, 1: 9, 2: 16}
        self.split_val_to_idx = {4: 0, 9: 1, 16: 2}

        # --- 1. 动作空间修改 ---
        # 以前: Type A (Split调整) + Type B (移动路径)
        # 现在: 仅保留移动路径 (从 Dev i 移到 Dev j)
        # 动作总数 = N * (M+1) * (M+1)
        self.ops_per_client = self.num_devices * self.num_devices
        self.action_space = spaces.Discrete(self.N * self.ops_per_client)

        # --- 2. 观测空间 (保持不变，作为 Context) ---
        # 包含：Split Num (静态Context) + 分配矩阵 (动态State) + 系统负载 (动态State)
        obs_dim = self.N + (self.N * self.num_devices) + self.num_devices
        self.observation_space = spaces.Box(low=-1, high=float('inf'), shape=(obs_dim,), dtype=np.float32)

        # 内部状态
        self.current_split_nums = np.zeros(self.N, dtype=np.int32)
        self.current_split_indices = np.zeros(self.N, dtype=np.int32)  # 仅用于Obs归一化
        self.current_allocation = np.zeros((self.N, self.num_devices), dtype=np.int32)

        self.current_step = 0
        self.last_makespan = 0.0
        self.last_std = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # --- 1. 确定 Split Nums (由外部给定或随机模拟) ---
        if options and 'fixed_split_nums' in options:
            # 推理模式：使用 GWO 算好的切分
            self.current_split_nums = np.array(options['fixed_split_nums'], dtype=np.int32)
            # 更新对应的 index 用于 observation
            self.current_split_indices = np.array([self.split_val_to_idx.get(n, 1) for n in self.current_split_nums])
        else:
            # 训练模式：随机采样切分方案，模拟 GWO 的各种可能输出，训练 Agent 的泛化能力
            self.current_split_indices = np.random.randint(0, 3, size=self.N)
            self.current_split_nums = np.array([self.split_idx_map[i] for i in self.current_split_indices])

        # --- 2. 初始化 Allocation (Warm Start) ---
        # 使用均匀分配作为起点，让 Agent 在此基础上微调，比纯随机分配收敛更快
        self.current_allocation = np.zeros((self.N, self.num_devices), dtype=np.int32)
        # for i in range(self.N):
        #     n = self.current_split_nums[i]
        #     base = n // self.num_devices
        #     rem = n % self.num_devices
        #     self.current_allocation[i] = base
        #     self.current_allocation[i, :rem] += 1

        for i in range(self.N):
            n = self.current_split_nums[i]
            base = n // self.num_devices
            rem = n % self.num_devices
            self.current_allocation[i] = base
            start_index = self.num_devices - rem
            if rem > 0:
                self.current_allocation[i, start_index:] += 1

        # --- 3. 计算初始指标 ---
        client_latencies, es_times = self.task_env._calculate_system_metrics(
            self.current_allocation,
            split_nums_override=self.current_split_nums
        )
        self.last_makespan = np.max(client_latencies)
        self.last_std = np.std(es_times)

        return self._get_obs(), {}

    def _get_obs(self):
        # 1. Split info (Static Context)
        split_obs = self.current_split_indices / 2.0

        # 2. Allocation info (Dynamic State)
        alloc_obs = self.current_allocation.flatten() / 16.0

        # 3. System Load (Dynamic State)
        _, es_times = self.task_env._calculate_system_metrics(
            self.current_allocation,
            split_nums_override=self.current_split_nums
        )
        # 补齐 Local 占位 (为了维度统一)
        load_obs = np.concatenate(([0], es_times))

        return np.concatenate([split_obs, alloc_obs, load_obs]).astype(np.float32)

    def step(self, action):
        # --- 解析动作 ---
        # Action ID -> (Client, From_Dev, To_Dev)
        client_id = action // self.ops_per_client
        move_code = action % self.ops_per_client

        from_dev = move_code // self.num_devices
        to_dev = move_code % self.num_devices

        reward = 0
        valid_action = False

        # --- 执行逻辑: 移动路径 ---
        # 只有当源设备有任务且源!=目标时，才执行移动
        if from_dev != to_dev and self.current_allocation[client_id, from_dev] > 0:
            self.current_allocation[client_id, from_dev] -= 1
            self.current_allocation[client_id, to_dev] += 1
            valid_action = True

        # --- 结算与奖励 ---
        if valid_action:
            client_latencies, new_es_times = self.task_env._calculate_system_metrics(
                self.current_allocation,
                split_nums_override=self.current_split_nums
            )
            new_makespan = np.max(client_latencies)
            new_std = np.std(new_es_times)

            # 改进的稠密奖励函数：
            # 1. Makespan 优化量 (权重高)
            delta_makespan = self.last_makespan - new_makespan
            # 2. 负载均衡优化量 (权重低，作为辅助信号)
            # 即使 Makespan 没变，如果负载更均衡了，也给一点奖励，防止 Agent 陷入平原
            delta_std = self.last_std - new_std

            reward = delta_makespan * 10.0 + delta_std * 2.0

            self.last_makespan = new_makespan
            self.last_std = new_std
        else:
            # 无效动作惩罚
            reward = -0.5
            new_makespan = self.last_makespan

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "makespan": new_makespan,
            "allocation": self.current_allocation.copy(),
            "split_nums": self.current_split_nums.copy()
        }

        return self._get_obs(), reward, terminated, truncated, info


def train_dqn_end2end(base_env, total_timesteps=70000):
    """
    训练 DQN 模型 (Fixed Split Version)
    """
    # 1. 使用新定义的固定切分环境
    env = FixedSplitIterativeEnv(base_env, max_steps=40)
    env = Monitor(env)

    # 2. 目录设置
    log_dir = "./logs/dqn_fixed"
    save_dir = "./models/dqn_fixed"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 3. 创建模型
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=2000,
        batch_size=128,
        gamma=0.95,  # 短视一点，关注当前几步的优化
        exploration_fraction=0.4,  # 增加探索时间
        exploration_final_eps=0.05,
        tensorboard_log=log_dir,
        device="auto"
    )

    # 4. 回调
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_dir, name_prefix="dqn_fixed")

    # 5. 训练
    print(f"--- Starting DQN Fixed-Split Training ({total_timesteps} steps) ---")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, tb_log_name="DQN_Fixed_Run")

    # 6. 保存
    final_path = os.path.join(save_dir, "final_dqn_fixed_model.zip")
    model.save(final_path)
    print(f"DQN Training Finished. Model saved to {final_path}")

    return model


def test_dqn_end2end(model_path, base_env, gwo_split_nums, num_episodes=20):
    """
    测试 DQN 模型 (需要传入 GWO 的切分结果)
    """
    env = FixedSplitIterativeEnv(base_env, max_steps=40)

    model = DQN.load(model_path, env=env)

    print("\n" + "=" * 40)
    print(" Testing DQN Fixed-Split Agent")
    print("=" * 40)

    best_global_makespan = float('inf')
    best_global_alloc = None

    for ep in range(num_episodes):
        # 关键：测试时必须传入固定的 split_nums
        obs, _ = env.reset(options={'fixed_split_nums': gwo_split_nums})
        done = False
        step = 0
        min_makespan = float('inf')
        best_alloc = None

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            current_makespan = info['makespan']
            if current_makespan < min_makespan:
                min_makespan = current_makespan
                best_alloc = info['allocation']

            step += 1

        if min_makespan < best_global_makespan:
            best_global_makespan = min_makespan
            best_global_alloc = best_alloc

        print(f"Episode {ep + 1}: Steps={step}, Best Makespan={min_makespan:.4f}")
        # print(best_alloc)

    print("-" * 40)
    print(f"Global Best Makespan: {best_global_makespan:.4f}")
    return best_global_makespan, best_global_alloc