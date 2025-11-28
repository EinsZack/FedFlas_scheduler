import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os


class FixedSplitContinuousEnv(gym.Env):
    """
    适配 SAC 的固定切分连续环境。
    Action: Box(3) -> [Client_Idx, From_Dev, To_Dev]
    逻辑: 接收连续动作 [-1, 1]，映射为离散的路径移动操作。
    """

    def __init__(self, task_env, max_steps=100):
        super().__init__()
        self.task_env = task_env
        self.N = task_env.N
        self.num_devices = task_env.num_devices  # M+1
        self.max_steps = max_steps

        self.split_idx_map = {0: 4, 1: 9, 2: 16}
        self.split_val_to_idx = {4: 0, 9: 1, 16: 2}

        # --- 动作空间 (SAC) ---
        # 3维连续向量: [Client_Idx, From_Dev, To_Dev]
        # 范围 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # --- 观测空间 (与 DQN 保持一致) ---
        # Split(N) + Alloc(N*(M+1)) + Load(M+1) + Progress(1)
        obs_dim = self.N + (self.N * self.num_devices) + self.num_devices + 1
        self.observation_space = spaces.Box(low=-1, high=float('inf'), shape=(obs_dim,), dtype=np.float32)

        # 内部状态
        self.current_split_nums = np.zeros(self.N, dtype=np.int32)
        self.current_split_indices = np.zeros(self.N, dtype=np.int32)
        self.current_allocation = np.zeros((self.N, self.num_devices), dtype=np.int32)

        self.current_step = 0
        self.last_makespan = 0.0
        self.last_std = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # 1. 确定 Split Nums
        if options and 'fixed_split_nums' in options:
            # 推理模式：使用 GWO 结果
            self.current_split_nums = np.array(options['fixed_split_nums'], dtype=np.int32)
            self.current_split_indices = np.array([self.split_val_to_idx.get(n, 1) for n in self.current_split_nums])
        else:
            # 训练模式：随机模拟
            self.current_split_indices = np.random.randint(0, 3, size=self.N)
            self.current_split_nums = np.array([self.split_idx_map[i] for i in self.current_split_indices])

        # 2. 初始化 Allocation (均匀分配作为起点)
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

        # 3. 计算初始指标
        client_latencies, es_times = self.task_env._calculate_system_metrics(
            self.current_allocation, split_nums_override=self.current_split_nums
        )
        self.last_makespan = np.max(client_latencies)
        self.last_std = np.std(es_times)

        return self._get_obs(), {}

    def _get_obs(self):
        # 1. Split info
        split_obs = self.current_split_indices / 2.0
        # 2. Allocation info
        alloc_obs = self.current_allocation.flatten() / 16.0
        # 3. Load info
        _, es_times = self.task_env._calculate_system_metrics(
            self.current_allocation, split_nums_override=self.current_split_nums
        )
        load_obs = np.concatenate(([0], es_times))
        # 4. Progress
        progress = np.array([self.current_step / self.max_steps])

        return np.concatenate([split_obs, alloc_obs, load_obs, progress]).astype(np.float32)

    def _map_action_to_discrete(self, action):
        """将连续动作 [-1, 1] 映射到离散逻辑"""
        # 1. Client Index
        # 映射到 [0, N-1]
        client_norm = (action[0] + 1) / 2.0
        client_id = int(client_norm * self.N)
        client_id = np.clip(client_id, 0, self.N - 1)

        # 2. From Device
        # 映射到 [0, M]
        from_norm = (action[1] + 1) / 2.0
        from_dev = int(from_norm * self.num_devices)
        from_dev = np.clip(from_dev, 0, self.num_devices - 1)

        # 3. To Device
        # 映射到 [0, M]
        to_norm = (action[2] + 1) / 2.0
        to_dev = int(to_norm * self.num_devices)
        to_dev = np.clip(to_dev, 0, self.num_devices - 1)

        return client_id, from_dev, to_dev

    def step(self, action):
        # 解析动作
        client_id, from_dev, to_dev = self._map_action_to_discrete(action)
        valid_action = False

        # 执行移动逻辑
        if from_dev != to_dev and self.current_allocation[client_id, from_dev] > 0:
            self.current_allocation[client_id, from_dev] -= 1
            self.current_allocation[client_id, to_dev] += 1
            valid_action = True

        # --- 奖励计算 ---
        reward = 0
        if valid_action:
            client_latencies, new_es_times = self.task_env._calculate_system_metrics(
                self.current_allocation, split_nums_override=self.current_split_nums
            )
            new_makespan = np.max(client_latencies)
            new_std = np.std(new_es_times)

            delta_makespan = self.last_makespan - new_makespan
            delta_std = self.last_std - new_std

            # SAC 奖励设计: 同样使用混合奖励，但由于 SAC 输出是连续值，
            # 可能会频繁触发无效动作，所以有效动作的奖励权重可以适当调大
            reward = delta_makespan * 20.0 + delta_std * 5.0

            self.last_makespan = new_makespan
            self.last_std = new_std
        else:
            # 无效动作惩罚
            reward = -0.5
            new_makespan = self.last_makespan

        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, False, {
            "makespan": new_makespan,
            "allocation": self.current_allocation.copy(),
            "split_nums": self.current_split_nums.copy()
        }


def train_sac_fixed_split(base_env, total_timesteps=70000):
    """
    训练 SAC 模型 (Fixed Split Version)
    """
    # 1. 包装环境
    env = FixedSplitContinuousEnv(base_env, max_steps=40)
    env = Monitor(env)

    # 2. 目录设置
    log_dir = "./logs/sac_fixed"
    save_dir = "./models/sac_fixed"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 3. 创建模型
    print(f"--- Starting SAC Fixed-Split Training ({total_timesteps} steps) ---")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=256,
        # 关键参数：自动调整熵系数，保证探索性
        ent_coef='auto',
        train_freq=1,
        gradient_steps=1,
        tau=0.005,
        gamma=0.98,  # 略微降低折扣因子，关注近期优化
        tensorboard_log=log_dir,
        device="auto"
    )

    # 4. 回调
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=save_dir, name_prefix="sac_fixed")

    # 5. 训练
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, tb_log_name="SAC_Fixed_Run")

    # 6. 保存
    final_path = os.path.join(save_dir, "final_sac_fixed_model.zip")
    model.save(final_path)
    print(f"SAC Training Finished. Model saved to {final_path}")

    return model


def test_sac_fixed_split(model_path, base_env, gwo_split_nums, num_episodes=20):
    """
    测试 SAC 模型 (需传入 GWO 切分方案)
    """
    env = FixedSplitContinuousEnv(base_env, max_steps=40)

    model = SAC.load(model_path, env=env)

    print("\n" + "=" * 40)
    print(" Testing SAC Fixed-Split Agent")
    print("=" * 40)

    best_global_makespan = float('inf')
    best_global_alloc = None

    for ep in range(num_episodes):
        # 关键：传入 GWO 结果
        obs, _ = env.reset(options={'fixed_split_nums': gwo_split_nums})

        done = False
        step = 0
        min_makespan = float('inf')
        best_alloc = None

        while not done:
            # SAC 默认 deterministic=True 输出均值
            # 在测试时，如果想探索更多解，可以设为 False (开启 stochastic 采样)
            # 对于迭代优化，建议开启随机性来寻找更优解
            action, _ = model.predict(obs, deterministic=False)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if info['makespan'] < min_makespan:
                min_makespan = info['makespan']
                best_alloc = info['allocation'].copy()

            step += 1

        if min_makespan < best_global_makespan:
            best_global_makespan = min_makespan
            best_global_alloc = best_alloc

        print(f"Episode {ep + 1}: Steps={step}, Best Makespan={min_makespan:.4f}")

    print("-" * 40)
    print(f"Global Best Makespan (SAC): {best_global_makespan:.4f}")
    return best_global_makespan, best_global_alloc