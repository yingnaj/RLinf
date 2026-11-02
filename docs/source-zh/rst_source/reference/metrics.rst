训练指标
========

RLinf 通过 :doc:`MetricLogger <../guides/logger>` 在若干命名空间下记录指标——``train/``、``rollout/``、
``env/`` 与 ``time/``。本页统一给出它们的定义；示例页面直接链接到此处，而不再重复说明。

.. tip::

   对具身任务而言，最有用的单一指标是 **``env/success_once``** —— 未归一化的回合成功率。在稀疏
   奖励下，其他 ``env/*`` 指标往往难以直接解读（见下文）。

训练指标 —— ``train/``
----------------------

策略与价值优化的统计量，每次 actor 更新时记录。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 指标
     - 含义
   * - ``train/actor/approx_kl``
     - 新旧策略之间的近似 KL 散度。
   * - ``train/actor/clip_fraction``
     - 概率比被裁剪的更新比例。
   * - ``train/actor/clipped_ratio``
     - 被裁剪后概率比的均值。
   * - ``train/actor/grad_norm``
     - actor 的梯度范数。
   * - ``train/actor/lr``
     - 当前学习率。
   * - ``train/actor/policy_loss``
     - PPO / GRPO 策略损失。
   * - ``train/critic/value_loss``
     - 价值函数损失。
   * - ``train/critic/value_clip_ratio``
     - 价值目标更新被裁剪的比例。
   * - ``train/critic/explained_variance``
     - 价值预测的可解释方差（越接近 1 越好）。
   * - ``train/entropy_loss``
     - 策略熵。
   * - ``train/loss``
     - 训练总损失（actor + critic + 熵正则）。

Rollout 指标 —— ``rollout/``
----------------------------

rollout 阶段收集的优势与奖励统计量。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 指标
     - 含义
   * - ``rollout/advantages_max``
     - 该批次中优势的最大值。
   * - ``rollout/advantages_mean``
     - 该批次中优势的均值。
   * - ``rollout/advantages_min``
     - 该批次中优势的最小值。
   * - ``rollout/rewards``
     - 一个 rollout chunk 的奖励。

环境指标 —— ``env/``
--------------------

来自模拟器的任务级信号。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 指标
     - 含义
   * - ``env/success_once``
     - **推荐。** 未归一化的回合成功率——最能反映真实任务表现。
   * - ``env/episode_len``
     - 回合中实际经历的环境步数。
   * - ``env/return``
     - 回合总回报。稀疏奖励下在成功结束前几乎为 0，训练过程中参考价值有限。
   * - ``env/reward``
     - step 级奖励（中间步为 ``0``，成功时为 ``1``）。日志值按回合步数归一化，难以直接反映真实表现。

如何选择日志后端（TensorBoard、Weights & Biases、SwanLab）以及配置 ``runner.logger``，参见
:doc:`日志 <../guides/logger>` 教程。

Causal Cosmos3 FPO 指标
--------------------------------------------------

使用 ``examples/embodiment/train_causal_fpo.py`` 时，通过 ``metrics.log`` 查看 RLinf 指标表，通过 TensorBoard 查看曲线。causal 配方默认启用 ``runner.per_worker_log`` 和 TensorBoard。``runner.logger.log_path: null`` 表示采用 ``runner.output_dir``；设置显式路径后，指标表和后端日志写到该路径。原有 ``metrics.jsonl`` 始终保存在 ``runner.output_dir``，包含原始 Actor 结果、扁平指标 ``scalars`` 和各 rank 的 ``workers`` 数据。每完成一个行为版本才写一条记录，训练模式还要求 checkpoint 已提交。续训曲线沿用行为版本号，与 optimizer 更新次数分别记录。

启用各 worker 日志后，在 ``tensorboard/all`` 查看汇总曲线，在 ``worker_logs/<group>/rank_<rank>/tensorboard`` 比较各 rank。通过 :doc:`日志配置 <../guides/logger>` 设置 ``runner.logger.logger_backends``，可选 ``tensorboard``、``wandb`` 和 ``swanlab``；空列表仍保留指标表与 JSONL。

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 指标
     - 含义
   * - ``env/success_rate``、``env/reward``、``env/episode_length``
     - 训练采样组成功率、平均未折扣 episode 回报、平均实际控制步数。泛化效果使用独立评测集衡量。
   * - ``train/fpo/*``、``train/actor/grad_norm``、``train/actor/lr``
     - FPO loss、CFM surrogate 统计、optimizer 步数、梯度范数与学习率。CFM 比率是 surrogate，不是动作 log probability 或 KL。
   * - ``time/sync_weights``、``time/generate_rollouts``、``time/recv_trajectories``
     - Driver 端墙钟秒数，包含通信和等待；采样耗时包括全部补采样轮次与 shard 构建。
   * - ``time/validate_behavior``、``time/actor_training``、``time/save_checkpoint``、``time/step``
     - 行为对齐、训练更新、checkpoint 提交和完整版本耗时；完整版本计时不含指标输出及 profiler 窗口开启、关闭。
   * - ``time/env/*``、``time/rollout/*``、``time/actor/*``、``time/collector/*``
     - 复用 RLinf Worker 计时；同一版本重复工作累加，汇总取最慢 rank。部分计时区间重叠，不能相加作为版本耗时。
   * - ``rollout/{episodes,env_steps,chunks}_per_second``
     - 完成的工作总量除以采样墙钟时间，计入补采样和末块实际执行步数。
   * - ``env/steps_per_second``、``train/actor/chunks_per_second``
     - 整个版本每秒执行的控制步数，以及训练阶段每秒处理的 chunk 数（计入 update epoch）。零优势跳过更新时训练吞吐为零；冻结模式不输出训练吞吐。
   * - ``*/memory/rss_gib``、``*/memory/{memory_allocated,memory_reserved,max_memory_allocated,max_memory_reserved}_gib``
     - 进程 RSS 和 PyTorch 分配器显存，单位 GiB。Actor 峰值覆盖行为对齐与训练，Rollout 峰值覆盖一次采样轮次，Env 分配器峰值从最近一次 reset 请求开始统计；汇总保留各轮、各 rank 的最大观测值。
   * - ``*/memory/device_used_gib``、``*/memory/device_total_gib``
     - 整卡显存快照，包含模拟器及其他进程占用，不代表本进程峰值或设备利用率百分比。

启用 ``cluster.tracer.enable`` 后，Worker 计时器和 ``ScopedTimer`` 同时产生追踪事件。``cluster.profiling`` 复用 RLinf 的后端 profiler、worker group 选择和从零开始的 ``steps``\ （每次更新前的行为版本号）。后端支持范围见 :doc:`性能分析 <../guides/profile>`。算子级追踪依赖已安装后端，上述常规指标可独立使用。
