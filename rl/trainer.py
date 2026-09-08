"""MAPPO 训练（注入 LLM 生成的奖励函数）。"""
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from LLM4RD import config as cfg
from LLM4RD.rl.networks import ActorCriticNetwork, RolloutStorage


def train_mappo_with_reward(env, compute_reward=None, training_steps=None, log_func=print):
    training_steps = training_steps or cfg.MAPPO_TRAINING_STEPS
    device = cfg.DEVICE
    state_dim = env.state_dim
    action_dim = env.action_dim
    n_vehicles = env.n_vehicles

    if compute_reward is None:
        env.external_reward_fn = None
        log_func("使用环境内置奖励函数进行训练")

    actor_critic = ActorCriticNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=cfg.LR, eps=1e-5)
    rollout = RolloutStorage(cfg.PPO_STEPS, n_vehicles, state_dim, device)

    current_state = env.reset()
    rollout.states[0].copy_(torch.FloatTensor(current_state).to(device))

    steps_per_rollout = cfg.PPO_STEPS * n_vehicles
    num_updates = max(10, int(training_steps // steps_per_rollout))
    log_func(
        f"MAPPO训练: steps={training_steps}, updates={num_updates}, "
        f"ppo_steps={cfg.PPO_STEPS}, vehicles={n_vehicles}"
    )

    training_stats = {
        'episode_rewards': [],
        'episode_benefits': [],
        'completed_orders': [],
    }
    current_episode_reward_sum = np.zeros(n_vehicles)
    total_steps = 0

    for update in range(num_updates):
        actor_critic.train()
        for step in range(cfg.PPO_STEPS):
            total_steps += n_vehicles
            with torch.no_grad():
                state_tensor = rollout.states[step]
                dist = Categorical(probs=actor_critic.get_action_probs(state_tensor))
                action = dist.sample()
                action_log_prob = dist.log_prob(action)
                value = actor_critic.get_value(state_tensor)

            actions_np = action.cpu().numpy()
            next_state, rewards, done = env.step(actions_np)
            rollout.insert(
                torch.FloatTensor(next_state).to(device),
                action.unsqueeze(-1),
                action_log_prob.unsqueeze(-1),
                value,
                torch.FloatTensor(rewards).unsqueeze(1).to(device),
                torch.FloatTensor([[1.0 - float(done)]] * n_vehicles).to(device),
            )
            current_episode_reward_sum += rewards
            if done:
                training_stats['episode_rewards'].append(float(current_episode_reward_sum.mean()))
                training_stats['completed_orders'].append(env.completed_orders_this_episode)
                training_stats['episode_benefits'].append({
                    'platform': getattr(env, 'total_platform_order_benefit', 0),
                    'grid': getattr(env, 'total_grid_power_loss_reduction_benefit', 0)
                    + getattr(env, 'total_grid_peak_shaving_benefit', 0),
                    'solar': getattr(env, 'total_solar_station_profit', 0),
                })
                current_state = env.reset()
                if step + 1 < cfg.PPO_STEPS:
                    rollout.states[step + 1].copy_(torch.FloatTensor(current_state).to(device))
                current_episode_reward_sum.fill(0.0)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollout.states[-1])
        rollout.compute_returns_and_advantages(next_value, cfg.GAMMA, cfg.GAE_LAMBDA)
        advantages = rollout.returns[:-1] - rollout.values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        num_minibatches = max(1, (cfg.PPO_STEPS * n_vehicles) // cfg.MINI_BATCH_SIZE)

        for _ in range(cfg.PPO_EPOCHS):
            for sample in rollout.feed_forward_generator(advantages, num_minibatches):
                states_batch, actions_batch, old_log_probs, returns_batch, adv_batch = sample
                values, action_log_probs, dist_entropy = actor_critic.evaluate_actions(states_batch, actions_batch)
                ratio = torch.exp(action_log_probs.unsqueeze(-1) - old_log_probs)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - cfg.CLIP_PARAM, 1.0 + cfg.CLIP_PARAM) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns_batch)
                loss = policy_loss + cfg.VF_COEFF * value_loss - cfg.ENT_COEFF * dist_entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), cfg.MAX_GRAD_NORM)
                optimizer.step()
        rollout.after_update()

    log_func(f"MAPPO训练完成: {len(training_stats['episode_rewards'])} 回合")
    return actor_critic, training_stats
