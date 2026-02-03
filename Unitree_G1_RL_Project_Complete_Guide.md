# Unitree G1 机器人RL训练项目完整指南

## 项目概述

### 1.1 项目简介
本项目是一个基于Isaac Lab框架的Unitree G1机器人强化学习（RL）训练系统，专门用于训练29自由度的Unitree G1机器人在复杂地形上进行速度跟踪任务。项目采用PPO（Proximal Policy Optimization）算法，通过并行仿真4096个环境来高效训练机器人控制策略。

### 1.2 核心目标
- 训练Unitree G1机器人按照指定的线速度和角速度命令行走
- 在鹅卵石道路等复杂地形上保持稳定步态
- 实现高效的能量利用和自然的运动模式
- 通过课程学习逐步提高任务难度

### 1.3 技术栈
- **仿真平台**: Isaac Lab (基于NVIDIA Omniverse)
- **RL算法**: RSL-RL (PPO实现)
- **编程语言**: Python 3.8+
- **深度学习框架**: PyTorch
- **机器人模型**: Unitree G1 29自由度

## 项目架构设计

### 2.1 系统架构
```
┌─────────────────────────────────────────────────────────────┐
│                   训练脚本 (train.py)                        │
├─────────────────────────────────────────────────────────────┤
│                  Gymnasium 环境接口                          │
├─────────────────────────────────────────────────────────────┤
│              ManagerBasedRLEnv (Isaac Lab)                   │
├──────────────┬──────────────┬──────────────┬───────────────┤
│  场景管理器  │  观测管理器   │  奖励管理器   │  命令管理器    │
├──────────────┼──────────────┼──────────────┼───────────────┤
│  事件管理器  │  课程管理器   │  终止管理器   │  动作管理器    │
└──────────────┴──────────────┴──────────────┴───────────────┘
```

### 2.2 文件结构
```
phc_isaaclab/tasks/manager_based/phc_isaaclab/robot/g1/
├── __init__.py              # Gymnasium环境注册
├── velocity_env_cfg.py      # 主环境配置
├── agents/
│   ├── __init__.py
│   └── rsl_rl_ppo_cfg.py    # PPO算法配置
└── mdp/
    ├── __init__.py
    ├── rewards.py           # 自定义奖励函数
    ├── observations.py      # 自定义观测函数
    ├── curriculums.py       # 课程学习函数
    └── commands/
        ├── __init__.py
        └── velocity_command.py  # 速度命令扩展
```

## 环境配置详解

### 3.1 场景配置 (RobotSceneCfg)
```python
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """机器人场景配置"""
    # 地形: 鹅卵石道路，支持课程学习
    terrain = TerrainImporterCfg(...)
    
    # 机器人: Unitree G1 29自由度
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG
    
    # 传感器
    height_scanner = RayCasterCfg(...)      # 高度扫描仪
    contact_forces = ContactSensorCfg(...)  # 接触力传感器
    
    # 光照
    sky_light = AssetBaseCfg(...)
```

**关键参数**:
- `num_envs`: 4096个并行环境（训练时），32个（测试时）
- `env_spacing`: 2.5米环境间距
- 地形生成器: 9行×21列的鹅卵石道路网格

### 3.2 事件配置 (EventCfg)
事件系统用于环境随机化和域随机化：

| 事件类型 | 函数 | 目的 |
|---------|------|------|
| startup | randomize_rigid_body_material | 随机化摩擦系数 |
| startup | randomize_rigid_body_mass | 随机化基座质量 |
| reset | apply_external_force_torque | 施加外力和扭矩 |
| reset | reset_root_state_uniform | 重置基座位姿 |
| reset | reset_joints_by_scale | 重置关节状态 |
| interval | push_by_setting_velocity | 定期推动机器人 |

### 3.3 命令配置 (CommandsCfg)
```python
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),  # 每10秒重新采样命令
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),    # 初始X方向线速度范围
            lin_vel_y=(-0.1, 0.1),    # 初始Y方向线速度范围  
            ang_vel_z=(-0.1, 0.1)     # 初始Z轴角速度范围
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),    # 最大X方向线速度范围
            lin_vel_y=(-0.3, 0.3),    # 最大Y方向线速度范围
            ang_vel_z=(-0.2, 0.2)     # 最大Z轴角速度范围
        )
    )
```

### 3.4 动作配置 (ActionsCfg)
```python
@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"],      # 所有关节
        scale=0.25,              # 动作缩放因子
        use_default_offset=True  # 使用默认关节位置偏移
    )
```

### 3.5 观测配置 (ObservationsCfg)
观测分为策略网络观测和评论家网络观测：

**策略网络观测 (PolicyCfg)**:
- `base_ang_vel`: 基座角速度（带噪声）
- `projected_gravity`: 重力投影向量
- `velocity_commands`: 速度命令
- `joint_pos_rel`: 关节相对位置
- `joint_vel_rel`: 关节相对速度  
- `last_action`: 上一时间步动作
- 历史长度: 5步
- 启用数据损坏: 是

**评论家网络观测 (CriticCfg)**:
- 与策略网络类似，但包含更多特权信息
- 历史长度: 5步
- 无数据损坏

### 3.6 奖励配置 (RewardsCfg)
奖励函数设计复杂且全面：

#### 3.6.1 任务相关奖励
| 奖励项 | 权重 | 描述 |
|--------|------|------|
| track_lin_vel_xy | 1.0 | 跟踪XY平面线速度 |
| track_ang_vel_z | 0.5 | 跟踪Z轴角速度 |
| alive | 0.15 | 存活奖励 |

#### 3.6.2 基座稳定性奖励
| 奖励项 | 权重 | 描述 |
|--------|------|------|
| base_linear_velocity | -2.0 | 惩罚Z方向线速度 |
| base_angular_velocity | -0.05 | 惩罚XY平面角速度 |
| flat_orientation_l2 | -5.0 | 惩罚非水平姿态 |
| base_height | -10.0 | 惩罚偏离目标高度(0.78m) |

#### 3.6.3 关节控制奖励
| 奖励项 | 权重 | 描述 |
|--------|------|------|
| joint_vel | -0.001 | 惩罚关节速度 |
| joint_acc | -2.5e-7 | 惩罚关节加速度 |
| action_rate | -0.05 | 惩罚动作变化率 |
| dof_pos_limits | -5.0 | 惩罚关节位置超限 |
| energy | -2e-5 | 惩罚能量消耗 |

#### 3.6.4 腿部特定奖励
| 奖励项 | 权重 | 描述 |
|--------|------|------|
| joint_deviation_arms | -0.1 | 惩罚手臂关节偏移 |
| joint_deviation_waists | -1.0 | 惩罚腰部关节偏移 |
| joint_deviation_legs | -1.0 | 惩罚腿部关节偏移 |

#### 3.6.5 脚部步态奖励
| 奖励项 | 权重 | 描述 |
|--------|------|------|
| gait | 0.5 | 步态协调性奖励 |
| feet_slide | -0.2 | 惩罚脚部滑动 |
| feet_clearance | 1.0 | 脚部离地高度奖励 |

#### 3.6.6 其他奖励
| 奖励项 | 权重 | 描述 |
|--------|------|------|
| undesired_contacts | -1.0 | 惩罚非脚部接触 |

### 3.7 终止配置 (TerminationsCfg)
```python
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # 超时终止
    base_height = DoneTerm(func=mdp.root_height_below_minimum, 
                          params={"minimum_height": 0.2})  # 高度过低
    bad_orientation = DoneTerm(func=mdp.bad_orientation,
                              params={"limit_angle": 0.8})  # 姿态过差
```

### 3.8 课程学习配置 (CurriculumCfg)
```python
@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)      # 地形难度课程
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)       # 线速度命令课程
```

## MDP设计详解

### 4.1 状态空间 (State Space)
状态空间维度: ~200+ 维度

**观测组件**:
1. **基座状态** (6维):
   - 线速度 (3维)
   - 角速度 (3维)

2. **关节状态** (58维):
   - 29个关节的相对位置
   - 29个关节的相对速度

3. **外部输入** (3维):
   - 速度命令 (线速度x,y + 角速度z)

4. **环境信息** (3维):
   - 重力投影向量 (3维)

5. **历史信息**:
   - 5步历史观测 (约1000维)

### 4.2 动作空间 (Action Space)
动作空间维度: 29维

**动作类型**: 关节位置目标
- 范围: [-1, 1] 经过缩放
- 实际关节位置 = 默认位置 + 动作 × 缩放因子 × 力矩极限/刚度

### 4.3 奖励函数设计哲学

#### 4.3.1 分层奖励结构
```
主要任务奖励 (速度跟踪)
    ↓
次级任务奖励 (步态质量、能量效率)
    ↓
约束惩罚 (稳定性、安全限制)
```

#### 4.3.2 关键奖励函数实现

**速度跟踪奖励** (`track_lin_vel_xy_yaw_frame_exp`):
```python
def track_lin_vel_xy_yaw_frame_exp(env, command_name, std):
    # 计算命令速度与实际速度的误差
    command = env.command_manager.get_command(command_name)
    actual = env.scene["robot"].data.root_lin_vel_b[:, :2]
    error = torch.norm(command - actual, dim=1)
    # 使用指数核函数
    reward = torch.exp(-error**2 / (2 * std**2))
    return reward
```

**步态奖励** (`feet_gait`):
```python
def feet_gait(env, period, offset, sensor_cfg, threshold, command_name):
    # 计算全局相位
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period)
    # 为每只脚计算相位
    phases = [(global_phase + offset_) % 1.0 for offset_ in offset]
    # 检查接触状态与相位匹配
    reward = sum(~(is_stance ^ is_contact) for each foot)
    return reward
```

### 4.4 课程学习机制

#### 4.4.1 速度命令课程 (`lin_vel_cmd_levels`)
```python
def lin_vel_cmd_levels(env, env_ids, reward_term_name="track_lin_vel_xy"):
    # 获取当前奖励
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
    
    # 每 episode 检查一次
    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:  # 如果表现良好
            # 增加速度命令范围
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x) + [-0.1, 0.1],
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1]
            )
    return ranges.lin_vel_x[1]  # 返回当前最大速度
```

#### 4.4.2 地形难度课程
- 初始: 平坦地形
- 渐进: 增加坡度、不规则度
- 最终: 复杂鹅卵石道路

## RL算法配置

### 5.1 PPO超参数
```python
@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24          # 每个环境收集的步数
    max_iterations = 50000          # 最大训练迭代次数
    save_interval = 100             # 保存间隔
    empirical_normalization = False # 不使用经验归一化
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,         # 初始策略噪声
        actor_hidden_dims=[512, 256, 128],  # 策略网络结构
        critic_hidden_dims=[512, 256, 128], # 价值网络结构
        activation="elu"            # 激活函数
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,        # 价值损失系数
        use_clipped_value_loss=True,# 使用裁剪价值损失
        clip_param=0.2,             # PPO裁剪参数
        entropy_coef=0.01,          # 熵系数
        num_learning_epochs=5,      # 每个迭代的学习轮数
        num_mini_batches=4,         # 小批量数量
        learning_rate=1.0e-3,       # 学习率
        schedule="adaptive",        # 学习率调度
        gamma=0.99,                 # 折扣因子
        lam=0.95,                   # GAE参数
        desired_kl=0.01,            # 目标KL散度
        max_grad_norm=1.0           # 梯度裁剪
    )
```

### 5.2 网络架构
```
策略网络 (Actor):
输入: [观测维度] → 512 → 256 → 128 → 输出: [动作维度]

价值网络 (Critic):
输入: [观测维度] → 512 → 256 → 128 → 输出: [1]
```

## 训练流程

### 6.1 环境安装
```bash
# 1. 安装Isaac Lab
# 按照官方指南安装: https://isaac-sim.github.io/IsaacLab/

# 2. 克隆项目
git clone <repository-url>
cd phc_isaaclab

# 3. 安装项目依赖
python -m pip install -e source/phc_isaaclab

# 4. 验证安装
python scripts/list_envs.py  # 应该能看到 "Unitree-G1-29dof-Velocity-v0"
```

### 6.2 数据准备
```bash
# 确保Unitree模型文件存在
# 模型路径: unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd
# 如果没有，需要从Unitree官方获取
```

### 6.3 训练命令
```bash
# 基础训练（无头模式）
python scripts/rsl_rl/train.py \
  --task=Unitree-G1-29dof-Velocity-v0 \
  --headless \
  --max_iterations=50000

# 带视频录制
python scripts/rsl_rl/train.py \
  --task=Unitree-G1-29dof-Velocity-v0 \
  --headless \
  --video \
  --video_interval=2000 \
  --video_length=200

# 指定环境数量
python scripts/rsl_rl/train.py \
  --task=Unitree-G1-29dof-Velocity-v0 \
  --headless \
  --num_envs=2048

# 分布式训练（多GPU）
python scripts/rsl_rl/train.py \
  --task=Unitree-G1-29dof-Velocity-v0 \
  --headless \
  --distributed
```

### 6.4 测试命令
```bash
# 使用零动作代理测试环境
python scripts/zero_agent.py --task=Unitree-G1-29dof-Velocity-v0

# 使用随机动作代理测试
python scripts/random_agent.py --task=Unitree-G1-29dof-Velocity-v0
```

### 6.5 监控和调试
```bash
# 训练日志