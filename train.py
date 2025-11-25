# 训练入口
import os

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch.utils.data import get_worker_info

from scheduler_1115.envs import gymEnv, Env
from scheduler_1115.envs import GWOEnv, NewGWOEnv
import FedServer_agent
from scheduler_1115.GWO.Splitor import Splitor
from scheduler_1115.GWO.Splitor_all import Splitor_All
from utils_DQN import *
# from Splitor_RS_PPO import Splitor_RS_PPO, PPOAgent
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


# def train_ppo_agent(ES_list, client_list, split_num_list, model_type="cifar100", total_timesteps=20000):
#     """
#     封装 PPO 算法的训练过程。
#
#     Args:
#         ES_list: Edge Server 列表。
#         client_list: Client 列表。
#         split_num_list: 路径切分数量列表。
#         model_type: 模型类型字符串。
#         total_timesteps: 训练的总时间步数。
#     """
#
#     # --- 1. 环境准备 ---
#     def make_ppo_env():
#         env = gymEnv.GymPPOEnv(ES_list, client_list, split_num_list, model_type=model_type)
#         # PPO 最好使用 Monitor 包装，方便 TensorBoard 记录
#         return Monitor(env)
#
#         # PPO 是 On-Policy 算法，通常使用 n_envs > 1 (如果有多个 CPU 核) 来加速数据收集
#
#     # 这里我们保持 n_envs=1，但可以使用 SubprocVecEnv 来加速 (需在 make_vec_env 中设置)
#     vec_env = make_vec_env(make_ppo_env, n_envs=1, vec_env_cls=DummyVecEnv)
#
#     # --- 2. 目录设置 ---
#     log_dir = "./logs/ppo_train"
#     save_dir = "./models/ppo_agent"
#     os.makedirs(log_dir, exist_ok=True)
#     os.makedirs(save_dir, exist_ok=True)
#
#     # --- 3. 模型实例化 ---
#     print(f"--- PPO Agent Training ({total_timesteps} steps) ---")
#     model = PPO(
#         "MlpPolicy",
#         vec_env,
#         verbose=1,
#         learning_rate=0.0003,
#         n_steps=200,  # 每次更新收集的步数
#         batch_size=64,
#         n_epochs=10,
#         gamma=0.98,
#         gae_lambda=0.94,
#         clip_range=0.2,
#         ent_coef=0.05,
#         vf_coef=0.5,
#         max_grad_norm=0.5,
#         tensorboard_log=log_dir,
#         device="auto"
#     )
#
#     # --- 4. 设置回调 (可选) ---
#     checkpoint_callback = CheckpointCallback(
#         save_freq=total_timesteps // 5,  # 每训练 1/5 的总步数保存一次
#         save_path=save_dir,
#         name_prefix="ppo_model"
#     )
#
#     # --- 5. 开始训练 ---
#     model.learn(
#         total_timesteps=total_timesteps,
#         callback=checkpoint_callback,
#         tb_log_name="PPO_run"
#     )
#
#     # --- 6. 保存最终模型 ---
#     final_model_path = os.path.join(save_dir, "final_ppo_model.zip")
#     model.save(final_model_path)
#     print(f"PPO 训练完成！最终模型保存到 {final_model_path}")
#
#     return final_model_path


def train_ppo_agent(ES_list, client_list, split_num_list, model_type="cifar100", total_timesteps=100000, bandwidth=60):
    """
    封装 PPO 算法的训练过程，添加 EvalCallback 进行评估和可视化。
    ... (Args 部分略) ...
    """

    # --- 1. 环境准备 ---
    # 训练环境 (Training Environment)
    def make_ppo_env():
        env = gymEnv.GymPPOEnv(ES_list, client_list, split_num_list, model_type=model_type,bandwidth=bandwidth)
        return Monitor(env)

    vec_env = make_vec_env(make_ppo_env, n_envs=1, vec_env_cls=DummyVecEnv)

    # 2. 包装 VecEnv，启用状态归一化
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,  # <--- 启用观测（状态）归一化
        norm_reward=False,  # 奖励归一化通常是可选的，这里先保持关闭
        clip_obs=10.0  # 防止归一化后的值过大
    )

    # 评估环境 (Evaluation Environment)
    def make_eval_env():
        env = gymEnv.GymPPOEnv(ES_list, client_list, split_num_list, model_type=model_type, bandwidth=bandwidth)
        return Monitor(env)

    # 评估环境通常也使用 n_envs=1
    eval_env = make_vec_env(make_eval_env, n_envs=1, vec_env_cls=DummyVecEnv)

    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        # 在训练结束后，你需要保存并加载训练环境的 stats，以供评估环境使用。
    )

    # --- 2. 目录设置 ---
    log_dir = "./logs/ppo_train"
    save_dir = "./models/ppo_agent"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 评估日志目录 (用于存储评估时的平均奖励、长度等统计信息)
    eval_log_dir = os.path.join(log_dir, "eval_results")
    os.makedirs(eval_log_dir, exist_ok=True)

    # --- 3. 模型实例化 ---
    # ... (模型实例化代码保持不变) ...
    print(f"--- PPO Agent Training ({total_timesteps} steps) ---")
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=256,  # 每次更新收集的步数
        batch_size=64,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.94,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        device="auto"
    )

    # --- 4. 设置回调 (可选) ---
    # 1. 保存检查点的回调
    checkpoint_callback = CheckpointCallback(
        save_freq=total_timesteps // 10,
        save_path=save_dir,
        name_prefix="ppo_model"
    )

    # 2. 评估回调 (EvalCallback)
    # 它会定期评估 Agent，并将性能最佳的模型保存到 best_model_save_path
    # eval_freq = max(total_timesteps // 10, 1000)  # 至少 1000 步评估一次，或总步数的 1/10
    eval_freq = 10000
    best_model_save_path = os.path.join(save_dir, "best_model")
    os.makedirs(best_model_save_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path=eval_log_dir,  # 评估结果日志路径
        eval_freq=eval_freq,
        deterministic=False,  # <--- 保持策略的随机性，避免收敛到次优的集中分配
        render=False,
        verbose=1
    )

    # 合并回调列表
    callback_list = [checkpoint_callback, eval_callback]

    # --- 5. 开始训练 ---
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        tb_log_name="PPO_run"
    )

    # --- 6. 保存最终模型 ---
    final_model_path = os.path.join(save_dir, "final_ppo_model.zip")
    model.save(final_model_path)
    print(f"PPO 训练完成！最终模型保存到 {final_model_path}")

    # 训练完成后，必须保存 VecNormalize 的统计信息！
    vec_env.save(os.path.join(save_dir, "vec_normalize.pkl"))

    # 返回最佳模型路径（EvalCallback 保存的）
    best_model_path = os.path.join(best_model_save_path, "best_model.zip")
    if os.path.exists(best_model_path):
        print(f"最佳模型已保存到: {best_model_path}")
        return best_model_path

    return final_model_path


def test_agent_allocation(
        model_path: str,
        algorithm_cls,
        ES_list: list,
        client_list: list,
        split_num_list: np.ndarray,
        model_type: str = "cifar100",
        test_episodes: int = 5,
        deterministic: bool = True,
        bandwidth=60
):
    """
    加载最佳模型并运行多个回合，输出每个回合的最终 ES 分配序列。

    Args:
        model_path: 最佳模型文件 (.zip) 的路径。
        algorithm_cls: 要加载的算法类 (如 PPO, DQN)。
        ES_list, client_list, split_num_list, model_type: 环境参数。
        test_episodes: 要运行的回合数。
        deterministic: 是否使用确定性策略（argmax）。
    """

    # 确定 VecNormalize 统计信息的路径
    # 假设 vec_normalize.pkl 与模型保存在同一父目录下
    base_dir = os.path.dirname(os.path.dirname(model_path))
    vec_stats_path = os.path.join(base_dir, "vec_normalize.pkl")

    # --- 1. 准备测试环境 ---
    def make_test_env():
        env = gymEnv.GymPPOEnv(ES_list, client_list, split_num_list, model_type=model_type,bandwidth=bandwidth)
        return Monitor(env)

    vec_env = make_vec_env(make_test_env, n_envs=1, vec_env_cls=DummyVecEnv)

    # --- 2. 加载 VecNormalize 统计信息 ---
    if os.path.exists(vec_stats_path):
        print(f"✅ 找到并加载 VecNormalize 统计信息: {vec_stats_path}")
        vec_env = VecNormalize.load(vec_stats_path, vec_env)
        # 必须禁用训练模式，防止评估时修改统计信息
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print("⚠️ 未找到 VecNormalize 统计信息，模型可能因输入未归一化而表现异常。")

    # --- 3. 加载模型 ---
    try:
        model = algorithm_cls.load(model_path, env=vec_env)
        print(f"✅ 成功加载模型: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # --- 4. 运行评估 ---
    print("\n" + "=" * 50)
    print(f"🚀 开始测试 {algorithm_cls.__name__} (Deterministic={deterministic}, Episodes={test_episodes})")
    print("=" * 50)

    all_allocations = []

    for episode in range(test_episodes):
        # obs, info = vec_env.reset()
        try:
            obs, info = vec_env.reset()
        except ValueError:
            # 假设只返回了 obs
            obs = vec_env.reset()
            info = [{}]  # 为兼容后续代码，添加一个包含空字典的列表（因为是 VecEnv）

        done = False
        step = 0

        # 用于记录本次回合的分配序列
        episode_allocations = []

        # 运行直到回合结束 (所有路径分配完毕)
        while not done:
            # 使用模型进行预测
            action, _ = model.predict(obs, deterministic=deterministic)

            # 记录本次分配的动作 (ES 索引)
            # action[0] 是因为 vec_env 包装了一层，即使 n_envs=1
            episode_allocations.append(action[0])

            # obs, reward, terminated, truncated, info = vec_env.step(action)

            # try:
            #     # 尝试接收 5 个值 (新版 API)
            #     obs, reward, terminated, truncated, info = vec_env.step(action)
            # except ValueError:
            # 如果报错，说明是旧版 API，只返回 4 个值 (obs, reward, done, info)
            # 此时，terminated 和 truncated 都包含在旧的 done 中
            obs, reward, done_old, info = vec_env.step(action)

            # 在 SB3/Monitor 环境中，当 done_old 为 True 时，
            # terminated 或 truncated 至少有一个是 True。
            # 为了兼容，我们直接使用 done_old 来定义循环跳出条件。
            terminated = done_old  # 假设所有结束都算是 terminated
            truncated = np.zeros_like(terminated)  # 假设没有专门的 truncated 信号

            done = terminated or truncated
            step += 1

        # 提取最终的 Makespan
        final_makespan = info[0].get('makespan', 'N/A')
        allocation = info[0].get('allocation', 'N/A')
        client_time_list = info[0].get('client_time_list', 'N/A')

        print(f"\n--- Episode {episode + 1} ---")
        print(f"  总步骤数 (Total Steps): {step}")
        print(f"  最终 Makespan (Final Makespan): {final_makespan:.4f}")
        print(f"  最终各个设备时延 (Device Makespans): {client_time_list}")
        print(f"  ES 分配序列 (Allocation Sequence):")
        # 打印序列，每10个换行，方便查看负载均衡情况

        allocation_str = " -> ".join(map(str, episode_allocations))
        print(f"    {allocation_str}")

        all_allocations.append(episode_allocations)
        print(f"  分配矩阵 (Allocation Matrix):")
        print(info[0].get('allocation', 'N/A'))

    print("\n" + "=" * 50)
    print("测试完成。")

    return all_allocations


def train_sac_agent(ES_list, client_list, split_num_list, model_type="cifar100", total_timesteps=100000):
    """
    封装 SAC 算法的训练过程。

    Args:
        ES_list, client_list, split_num_list, model_type: 同 PPO。
        total_timesteps: 训练的总时间步数 (SAC 通常需要更多)。
    """

    # --- 1. 环境准备 ---
    def make_sac_env():
        env = gymEnv.GymPPOEnv(ES_list, client_list, split_num_list, model_type=model_type)
        return Monitor(env)

    # SAC 是 Off-Policy 算法，使用 DummyVecEnv 且 n_envs=1 即可
    vec_env = make_vec_env(make_sac_env, n_envs=1, vec_env_cls=DummyVecEnv)

    # --- 2. 目录设置 ---
    log_dir = "./logs/sac_train"
    save_dir = "./models/sac_agent"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # --- 3. 模型实例化 ---
    print(f"--- SAC Agent Training ({total_timesteps} steps) ---")
    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        # === SAC/Off-Policy 参数 ===
        buffer_size=50000,  # 经验回放缓冲区大小
        learning_starts=1000,  # 收集 1000 步后开始训练
        batch_size=256,
        train_freq=(1, 'step'),  # 每收集 1 步数据训练 1 次
        gradient_steps=1,  # 每次训练执行 1 次梯度更新
        ent_coef='auto',  # 自动调整熵系数
        tau=0.005,  # 目标网络软更新率
        # === 通用参数 ===
        learning_rate=0.0003,
        gamma=0.98,
        tensorboard_log=log_dir,
        device="auto"
    )

    # --- 4. 设置回调 (可选) ---
    checkpoint_callback = CheckpointCallback(
        save_freq=total_timesteps // 5,
        save_path=save_dir,
        name_prefix="sac_model"
    )

    # --- 5. 开始训练 ---
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="SAC_run"
    )

    # --- 6. 保存最终模型 ---
    final_model_path = os.path.join(save_dir, "final_sac_model.zip")
    model.save(final_model_path)
    print(f"SAC 训练完成！最终模型保存到 {final_model_path}")

    return final_model_path


def train_dqn_agent(ES_list, client_list, split_num_list, model_type="cifar100", total_timesteps=100000):
    """
    封装 DQN 算法的训练过程。DQN 专为离散动作空间设计。

    Args:
        ES_list: Edge Server 列表。
        client_list: Client 列表。
        split_num_list: 路径切分数量列表。
        model_type: 模型类型字符串。
        total_timesteps: 训练的总时间步数 (DQN 通常需要更多)。
    """

    # --- 1. 环境准备 ---
    def make_dqn_env():
        env = gymEnv.GymPPOEnv(ES_list, client_list, split_num_list, model_type=model_type)
        return Monitor(env)

    # DQN 是 Off-Policy 算法，使用 DummyVecEnv 且 n_envs=1 即可
    vec_env = make_vec_env(make_dqn_env, n_envs=1, vec_env_cls=DummyVecEnv)

    # --- 2. 目录设置 ---
    log_dir = "./logs/dqn_train"
    save_dir = "./models/dqn_agent"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # --- 3. 模型实例化 ---
    print(f"--- DQN Agent Training ({total_timesteps} steps) ---")

    # DQN 特有超参数配置
    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=1,
        # === DQN 核心参数 ===
        buffer_size=50000,  # 经验回放缓冲区大小
        learning_starts=1000,  # 收集 1000 步后开始训练
        batch_size=128,  # 每次优化的批次大小
        train_freq=(4, 'step'),  # 每收集 4 步数据训练 1 次 (典型设置)
        target_update_interval=500,  # 目标网络更新频率（步数）
        # === 探索率衰减参数 ===
        exploration_fraction=0.1,  # 在前 10% 的时间步中衰减探索率
        exploration_final_eps=0.05,  # 最终的最小探索率

        # === 通用参数 ===
        learning_rate=0.0003,
        gamma=0.98,
        tensorboard_log=log_dir,
        device="auto"
    )

    # --- 4. 设置回调 (可选) ---
    checkpoint_callback = CheckpointCallback(
        save_freq=total_timesteps // 5,
        save_path=save_dir,
        name_prefix="dqn_model"
    )

    # --- 5. 开始训练 ---
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="DQN_run"
    )

    # --- 6. 保存最终模型 ---
    final_model_path = os.path.join(save_dir, "final_dqn_model.zip")
    model.save(final_model_path)
    print(f"DQN 训练完成！最终模型保存到 {final_model_path}")

    return final_model_path

if __name__ == '__main__':
    # 一些参数
    num_tasks = 16

    # 创建各个设备，包括ES，Client和FedServer
    # 各个ES和Client
    ES_list = []
    client_list = []
    # ================这一部分是MNIST的数据量设定================
    # scene1
    # client_list.append(Env.Client(1, 42864, 328))
    # client_list.append(Env.Client(2, 35813, 328)) #
    # client_list.append(Env.Client(3, 43798, 281))
    # ES_list.append(Env.ES(11, 621409))
    # ES_list.append(Env.ES(12, 735531))
    # ES_list.append(Env.ES(13, 405849))
    # ES_list.append(Env.ES(14, 460893))
    # ES_list.append(Env.ES(15, 534774))
    # scene2
    # client_list.append(Env.Client(1, 42864, 196))
    # client_list.append(Env.Client(2, 35813, 150))
    # client_list.append(Env.Client(3, 43798, 196))
    # client_list.append(Env.Client(4, 43290, 196))
    # client_list.append(Env.Client(5, 37754, 196))
    # ES_list.append(Env.ES(6, 621409))
    # ES_list.append(Env.ES(7, 735531))
    # ES_list.append(Env.ES(8, 405849))
    # scene3
    # client_list.append(Env.Client(1, 42864, 87))
    # client_list.append(Env.Client(2, 35813, 89))
    # client_list.append(Env.Client(3, 43798, 89))
    # client_list.append(Env.Client(4, 43290, 98))
    # client_list.append(Env.Client(5, 37754, 98))
    # client_list.append(Env.Client(6, 42590, 92))
    # client_list.append(Env.Client(7, 38999, 98))
    # client_list.append(Env.Client(8, 36477, 87))
    # client_list.append(Env.Client(9, 49122, 98))
    # client_list.append(Env.Client(10, 33303, 98))  #
    # ES_list.append(Env.ES(21, 621409))
    # ES_list.append(Env.ES(22, 735531))
    # ES_list.append(Env.ES(23, 405849))
    # ES_list.append(Env.ES(24, 460893))
    # ES_list.append(Env.ES(25, 534774))

    # ================这一半是CIFAR10的数据量设定================
    # scene1
    # client_list.append(Env.Client(1, 42864, 242))
    # client_list.append(Env.Client(2, 35813, 282)) #
    # client_list.append(Env.Client(3, 43798, 256))
    # ES_list.append(Env.ES(11, 621409))
    # ES_list.append(Env.ES(12, 735531))
    # ES_list.append(Env.ES(13, 405849))
    # ES_list.append(Env.ES(14, 460893))
    # ES_list.append(Env.ES(15, 534774))
    # scene2
    # client_list.append(Env.Client(1, 42864, 160))
    # client_list.append(Env.Client(2, 35813, 187))
    # client_list.append(Env.Client(3, 43798, 180))
    # client_list.append(Env.Client(4, 43290, 138))
    # client_list.append(Env.Client(5, 37754, 114))
    # ES_list.append(Env.ES(6, 621409))
    # ES_list.append(Env.ES(7, 735531))
    # ES_list.append(Env.ES(8, 405849))
    # scene3
    # client_list.append(Env.Client(1, 42864, 75))
    # client_list.append(Env.Client(2, 35813, 74))
    # client_list.append(Env.Client(3, 43798, 55))
    # client_list.append(Env.Client(4, 43290, 79))
    # client_list.append(Env.Client(5, 37754, 92))
    # client_list.append(Env.Client(6, 42590, 77))
    # client_list.append(Env.Client(7, 38999, 85))
    # client_list.append(Env.Client(8, 36477, 78))
    # client_list.append(Env.Client(9, 49122, 69))
    # client_list.append(Env.Client(10, 33303, 93)) #
    # ES_list.append(Env.ES(21, 621409))
    # ES_list.append(Env.ES(22, 735531))
    # ES_list.append(Env.ES(23, 405849))
    # ES_list.append(Env.ES(24, 460893))
    # ES_list.append(Env.ES(25, 534774))
    # ES_list.append(Env.ES(26, 716509))
    # ES_list.append(Env.ES(27, 510927))
    # ES_list.append(Env.ES(28, 685382))
    # ES_list.append(Env.ES(29, 761315))
    # ES_list.append(Env.ES(30, 408975))
    # ================这一半是FMNIST的数据量设定================
    # scene1
    # client_list.append(Env.Client(1, 42864, 311))
    # client_list.append(Env.Client(2, 35813, 288))  #
    # client_list.append(Env.Client(3, 43798, 337))
    # ES_list.append(Env.ES(11, 621409))
    # ES_list.append(Env.ES(12, 735531))
    # ES_list.append(Env.ES(13, 405849))
    # ES_list.append(Env.ES(14, 460893))
    # ES_list.append(Env.ES(15, 534774))
    # scene2
    # client_list.append(Env.Client(1, 42864, 171))
    # client_list.append(Env.Client(2, 35813, 177))
    # client_list.append(Env.Client(3, 43798, 208))
    # client_list.append(Env.Client(4, 43290, 208))
    # client_list.append(Env.Client(5, 37754, 171))
    # ES_list.append(Env.ES(6, 621409))
    # ES_list.append(Env.ES(7, 735531))
    # ES_list.append(Env.ES(8, 405849))
    # scene3
    # client_list.append(Env.Client(1, 42864, 84))
    # client_list.append(Env.Client(2, 35813, 84))
    # client_list.append(Env.Client(3, 43798, 104))
    # client_list.append(Env.Client(4, 43290, 98))
    # client_list.append(Env.Client(5, 37754, 104))
    # client_list.append(Env.Client(6, 42590, 84))
    # client_list.append(Env.Client(7, 38999, 84))
    # client_list.append(Env.Client(8, 36477, 104))
    # client_list.append(Env.Client(9, 49122, 84))
    # client_list.append(Env.Client(10, 33303, 104))  #
    # ES_list.append(Env.ES(21, 621409))
    # ES_list.append(Env.ES(22, 735531))
    # ES_list.append(Env.ES(23, 405849))
    # ES_list.append(Env.ES(24, 460893))
    # ES_list.append(Env.ES(25, 534774))
    # ================这一半是CIFAR100的数据量设定================
    # scene1
    client_list.append(Env.Client(1, 42864, 243))
    client_list.append(Env.Client(2, 35813, 264)) #
    client_list.append(Env.Client(3, 43798, 273))
    ES_list.append(Env.ES(11, 621409))
    ES_list.append(Env.ES(12, 735531))
    ES_list.append(Env.ES(13, 405849))
    ES_list.append(Env.ES(14, 460893))
    ES_list.append(Env.ES(15, 534774))
    # scene2
    # client_list.append(Env.Client(1, 42864, 160))
    # client_list.append(Env.Client(2, 35813, 161)) #
    # client_list.append(Env.Client(3, 43798, 140))
    # client_list.append(Env.Client(4, 43290, 155))
    # client_list.append(Env.Client(5, 37754, 165))
    # ES_list.append(Env.ES(6, 621409))
    # ES_list.append(Env.ES(7, 735531))
    # ES_list.append(Env.ES(8, 405849))
    # scene3
    # client_list.append(Env.Client(1, 42864, 74))
    # client_list.append(Env.Client(2, 35813, 76))
    # client_list.append(Env.Client(3, 43798, 78))
    # client_list.append(Env.Client(4, 43290, 78))
    # client_list.append(Env.Client(5, 37754, 80))
    # client_list.append(Env.Client(6, 42590, 71))
    # client_list.append(Env.Client(7, 38999, 79))
    # client_list.append(Env.Client(8, 36477, 82))
    # client_list.append(Env.Client(9, 49122, 82))
    # client_list.append(Env.Client(10, 33303, 76)) #
    # ES_list.append(Env.ES(21, 621409))
    # ES_list.append(Env.ES(22, 735531))
    # ES_list.append(Env.ES(23, 405849))
    # ES_list.append(Env.ES(24, 460893))
    # ES_list.append(Env.ES(25, 534774))
    # ES_list.append(Env.ES(26, 716509))
    # ES_list.append(Env.ES(27, 510927))
    # ES_list.append(Env.ES(28, 685382))
    # ES_list.append(Env.ES(29, 761315))
    # ES_list.append(Env.ES(30, 408975))

    model_type = "cifar100"
    bandwidth = 30


    # 创建Env，将设备信息传入给Env
    # gwoenv = GWOEnv.TaskAssignmentEnv(ES_list, client_list, model_type, bandwidth)  # 专门用于GWO的Env，后面再改
    gwoenv = NewGWOEnv.TaskAssignmentEnv(ES_list, client_list,None, model_type, bandwidth)  # 专门用于GWO的Env，后面再改

    # 创建Splitor，作为第一阶段决定路径切分数量
    # splitor = Splitor(ES_list, client_list, gwoenv)
    # split_num_list, init_dist = splitor.get_split_numlist()
    # split_num_list = split_num_list.astype(int)

    # splitor_all
    splitor_all = Splitor_All(ES_list, client_list, gwoenv)
    split_num_list, init_dist = splitor_all.get_split_numlist()
    split_num_list = split_num_list.astype(int)


    # 训练PPO智能体
    final_ppo_model_path = train_ppo_agent(
        ES_list,
        client_list,
        split_num_list,
        model_type=model_type,
        total_timesteps=150000,
        bandwidth=bandwidth
    )

    PPO_BASE_DIR = "./models/ppo_agent"
    BEST_PPO_MODEL_PATH = os.path.join(PPO_BASE_DIR, "best_model", "best_model.zip")


    if os.path.exists(BEST_PPO_MODEL_PATH):
        # --- 测试确定性模式 (检查是否集中分配) ---
        print("\n" + "#" * 60)
        print("### 模式一：Deterministic=True (集中分配检查) ###")
        print("#" * 60)
        test_agent_allocation(
            model_path=BEST_PPO_MODEL_PATH,
            algorithm_cls=PPO,
            ES_list=ES_list,
            client_list=client_list,
            split_num_list=split_num_list,
            model_type=model_type,
            test_episodes=5,
            deterministic=True,  # 检查只选一个ES的问题
            bandwidth=bandwidth
        )

        # --- 测试随机模式 (检查负载均衡) ---
        print("\n" + "#" * 60)
        print("### 模式二：Deterministic=False (均衡分配检查) ###")
        print("#" * 60)
        test_agent_allocation(
            model_path=BEST_PPO_MODEL_PATH,
            algorithm_cls=PPO,
            ES_list=ES_list,
            client_list=client_list,
            split_num_list=split_num_list,
            model_type=model_type,
            test_episodes=5,
            deterministic=False,  # 检查是否能分散分配
            bandwidth = bandwidth
        )
    else:
        print(f"❌ 最佳模型文件未找到: {BEST_PPO_MODEL_PATH}，请先运行训练。")

    print("对比初始GWO的分配方案：")
    print(init_dist)
    env = gymEnv.GymPPOEnv(ES_list, client_list, split_num_list, model_type=model_type, bandwidth=bandwidth)
    init_time, client_time_list = env.calculate_makespan_for_allocation(init_dist)
    print(f"初始分配方案的Makespan: {init_time}")
    print(client_time_list)


    # 训练 SAC (例如 100,000 步)
    # sac_path = train_sac_agent(
    #     ES_list,
    #     client_list,
    #     split_num_list,
    #     model_type="cifar100",
    #     total_timesteps=100000
    # )

    # dqn_path = train_dqn_agent(
    #     ES_list,
    #     client_list,
    #     split_num_list,
    #     model_type="cifar100",
    #     total_timesteps=100000
    # )
