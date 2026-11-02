Training Metrics
================

RLinf reports metrics through the :doc:`MetricLogger <../guides/logger>` under a few namespaces —
``train/``, ``rollout/``, ``env/``, and ``time/``. This page defines them once; example
pages link here instead of repeating the definitions.

.. tip::

   For embodied tasks the single most useful signal is **``env/success_once``** — the
   unnormalized episodic success rate. Most other ``env/*`` values are hard to read
   directly under sparse rewards (see below).

Training metrics — ``train/``
-----------------------------

Policy- and value-optimization statistics, logged every actor update.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Metric
     - Meaning
   * - ``train/actor/approx_kl``
     - Approximate KL divergence between the old and new policies.
   * - ``train/actor/clip_fraction``
     - Fraction of updates where the probability ratio was clipped.
   * - ``train/actor/clipped_ratio``
     - Mean of the clipped probability ratios.
   * - ``train/actor/grad_norm``
     - Gradient norm of the actor.
   * - ``train/actor/lr``
     - Current learning rate.
   * - ``train/actor/policy_loss``
     - PPO / GRPO policy loss.
   * - ``train/critic/value_loss``
     - Value-function loss.
   * - ``train/critic/value_clip_ratio``
     - Fraction of value targets whose update was clipped.
   * - ``train/critic/explained_variance``
     - Explained variance of the value predictions (closer to 1 is better).
   * - ``train/entropy_loss``
     - Policy entropy.
   * - ``train/loss``
     - Total training loss (actor + critic + entropy regularization).

Rollout metrics — ``rollout/``
------------------------------

Statistics of the advantages and rewards collected during rollout.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Metric
     - Meaning
   * - ``rollout/advantages_max``
     - Maximum advantage in the batch.
   * - ``rollout/advantages_mean``
     - Mean advantage in the batch.
   * - ``rollout/advantages_min``
     - Minimum advantage in the batch.
   * - ``rollout/rewards``
     - Reward of a rollout chunk.

Environment metrics — ``env/``
------------------------------

Task-level signals from the simulator.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Metric
     - Meaning
   * - ``env/success_once``
     - **Recommended.** Unnormalized episodic success rate — the truest measure of task performance.
   * - ``env/episode_len``
     - Number of environment steps elapsed in the episode.
   * - ``env/return``
     - Episode return. Under sparse rewards this is near-zero until the terminal success step, so it is not very informative during training.
   * - ``env/reward``
     - Step-level reward (``0`` on intermediate steps, ``1`` on success). The logged value is normalized by episode length, which makes it hard to read as real performance.

See also the :doc:`Logger <../guides/logger>` tutorial for choosing backends (TensorBoard,
Weights & Biases, SwanLab) and configuring ``runner.logger``.

Causal Cosmos3 FPO Metrics
--------------------------------------------------

For ``examples/embodiment/train_causal_fpo.py``, read ``metrics.log`` for the
RLinf metric table and TensorBoard for curves. The causal recipes enable
``runner.per_worker_log`` and the TensorBoard backend by default.
``runner.logger.log_path: null`` uses ``runner.output_dir``; an explicit path
places the table and backend logs there. The original ``metrics.jsonl`` remains
in ``runner.output_dir`` with raw actor results, flattened ``scalars``, and
per-rank ``workers`` data. Each record follows a completed behavior version
(and a committed checkpoint in training mode). Resuming preserves this version
as the curve step; it is distinct from the optimizer step count.

With per-worker logging enabled, open ``tensorboard/all`` for aggregate curves
and ``worker_logs/<group>/rank_<rank>/tensorboard`` for individual ranks.
Choose ``runner.logger.logger_backends`` from ``tensorboard``, ``wandb``, and
``swanlab`` through :doc:`the logger configuration <../guides/logger>`.
An empty list keeps the table and JSONL outputs.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Meaning
   * - ``env/success_rate``, ``env/reward``, ``env/episode_length``
     - Training-group success rate, mean undiscounted episode return, and mean executed control steps. Use a separate evaluation set to measure generalization.
   * - ``train/fpo/*``, ``train/actor/grad_norm``, ``train/actor/lr``
     - FPO loss, CFM surrogate statistics, optimizer step counts, gradient norm, and learning rate. CFM ratios are surrogates, not action log probabilities or KL.
   * - ``time/sync_weights``, ``time/generate_rollouts``, ``time/recv_trajectories``
     - Driver wall time in seconds, including communication and waits. Rollout time includes every extra collection round and shard construction.
   * - ``time/validate_behavior``, ``time/actor_training``, ``time/save_checkpoint``, ``time/step``
     - Behavior alignment, update, checkpoint commit, and complete version wall time. The version timer excludes metric output and profiler-window setup/teardown.
   * - ``time/env/*``, ``time/rollout/*``, ``time/actor/*``, ``time/collector/*``
     - RLinf Worker timers. Repeated work accumulates within a version; the aggregate reports the slowest rank. Overlapping timers must not be summed into step time.
   * - ``rollout/{episodes,env_steps,chunks}_per_second``
     - Total completed work divided by rollout wall time, including retries and actual partial-tail control steps.
   * - ``env/steps_per_second``, ``train/actor/chunks_per_second``
     - Executed controls per complete version second, and trained chunks across update epochs per actor-training second. Skipped zero-advantage updates report zero training throughput; frozen runs omit it.
   * - ``*/memory/rss_gib``, ``*/memory/{memory_allocated,memory_reserved,max_memory_allocated,max_memory_reserved}_gib``
     - Process RSS and PyTorch allocator memory, in GiB. Actor peaks cover behavior validation plus training; rollout peaks cover one collection round; env allocator peaks start at the latest reset request. Aggregates keep the largest observed value across rounds/ranks.
   * - ``*/memory/device_used_gib``, ``*/memory/device_total_gib``
     - Device-wide memory snapshots, including simulator and other-process allocations. These are not per-process peaks or device utilization percentages.

Worker timers and ``ScopedTimer`` also emit events when
``cluster.tracer.enable`` is enabled. ``cluster.profiling`` uses RLinf's
backend-specific profiler support, selected worker groups, and zero-based
``steps`` (behavior versions before each update). See :doc:`Profiling <../guides/profile>`
for backend availability. Kernel tracing remains subject to the installed backend;
the metrics above are available without enabling it.
