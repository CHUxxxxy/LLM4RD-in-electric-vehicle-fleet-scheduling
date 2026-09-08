"""策略评估与三方效益、组件统计收集。"""
from collections import defaultdict

import torch

from LLM4RD import config as cfg
from LLM4RD.analysis.correlation import analyze_component_benefit_correlation
from LLM4RD.analysis.statistical_reflection import analyze_episode_component_statistics


def evaluate_policy(env, actor_critic, num_episodes=None, compute_reward=None, log_func=print):
    num_episodes = num_episodes or cfg.EVALUATION_EPISODES
    actor_critic.eval()
    eval_stats = {
        'episode_benefits': [],
        'completed_orders': [],
        'component_records': [],
        'event_benefit_correlations': [],
        'episode_component_stats': [],
    }

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        ep_component_records = []
        ep_event_correlations = []
        ep_component_sums = defaultdict(float)
        ep_component_counts = defaultdict(int)
        prev_platform = prev_grid_loss = prev_grid_peak = prev_solar = 0

        while not done:
            with torch.no_grad():
                probs = actor_critic.get_action_probs(torch.FloatTensor(state).to(cfg.DEVICE))
                actions = probs.argmax(dim=-1).cpu().numpy()
            next_state, rewards, done = env.step(actions)

            cur_platform = getattr(env, 'total_platform_order_benefit', 0)
            cur_grid_loss = getattr(env, 'total_grid_power_loss_reduction_benefit', 0)
            cur_grid_peak = getattr(env, 'total_grid_peak_shaving_benefit', 0)
            cur_solar = getattr(env, 'total_solar_station_profit', 0)
            delta_platform = cur_platform - prev_platform
            delta_grid = (cur_grid_loss + cur_grid_peak) - (prev_grid_loss + prev_grid_peak)
            delta_solar = cur_solar - prev_solar

            if compute_reward is not None:
                env_metrics = env.get_environment_metrics()
                for v in env.vehicles:
                    try:
                        _, comps = compute_reward(v, env_metrics, env.current_cycle - 1, env.max_cycles)
                        if isinstance(comps, dict):
                            ep_component_records.append(comps.copy())
                            for k, val in comps.items():
                                try:
                                    ep_component_sums[k] += float(val)
                                    ep_component_counts[k] += 1
                                except Exception:
                                    pass
                            events = v.get('events', {})
                            if events.get('finished_order'):
                                ep_event_correlations.append({
                                    'event_type': 'order_completion',
                                    'components_at_event': comps.copy(),
                                    'actual_benefit': {
                                        'platform': max(0, delta_platform),
                                        'grid': 0,
                                        'solar': 0,
                                    },
                                })
                            charge_amt = events.get('charge_amount', 0)
                            if events.get('started_charging') or charge_amt > 0:
                                kwh = float(charge_amt) / 100.0 * 50
                                grid_gain = solar_gain = 0.0
                                if events.get('is_charging_solar') and kwh > 0:
                                    grid_gain += cfg.C_LOSS * kwh
                                    solar_gain += kwh * cfg.SOLAR_FARM_SUBSIDY
                                if events.get('is_peak_hour') and kwh > 0:
                                    grid_gain += cfg.PEAK_TO_BASE_COST_DIFFERENCE * kwh
                                ep_event_correlations.append({
                                    'event_type': 'charging',
                                    'components_at_event': comps.copy(),
                                    'actual_benefit': {'platform': 0, 'grid': grid_gain, 'solar': solar_gain},
                                })
                    except Exception:
                        pass

            prev_platform, prev_grid_loss, prev_grid_peak, prev_solar = (
                cur_platform, cur_grid_loss, cur_grid_peak, cur_solar
            )
            state = next_state

        ep_benefit = {
            'platform': getattr(env, 'total_platform_order_benefit', 0),
            'grid': getattr(env, 'total_grid_power_loss_reduction_benefit', 0)
            + getattr(env, 'total_grid_peak_shaving_benefit', 0),
            'solar': getattr(env, 'total_solar_station_profit', 0),
        }
        ep_benefit['total'] = ep_benefit['platform'] + ep_benefit['grid'] + ep_benefit['solar']
        eval_stats['episode_benefits'].append(ep_benefit)
        eval_stats['completed_orders'].append(env.completed_orders_this_episode)
        eval_stats['component_records'].extend(ep_component_records)
        eval_stats['event_benefit_correlations'].extend(ep_event_correlations)
        comp_avg = {
            k: ep_component_sums[k] / ep_component_counts[k]
            for k in ep_component_sums if ep_component_counts[k] > 0
        }
        eval_stats['episode_component_stats'].append({'component_avg': comp_avg, 'benefit': ep_benefit})

    n = num_episodes
    avg_benefit = {
        'platform': sum(b['platform'] for b in eval_stats['episode_benefits']) / n,
        'grid': sum(b['grid'] for b in eval_stats['episode_benefits']) / n,
        'solar': sum(b['solar'] for b in eval_stats['episode_benefits']) / n,
        'total': sum(b['total'] for b in eval_stats['episode_benefits']) / n,
    }
    eval_stats['avg_benefit'] = avg_benefit
    eval_stats['avg_completed_orders'] = sum(eval_stats['completed_orders']) / n

    if eval_stats['component_records']:
        component_summary = defaultdict(list)
        for rec in eval_stats['component_records']:
            for k, v in rec.items():
                try:
                    component_summary[k].append(float(v))
                except Exception:
                    pass
        eval_stats['component_analysis'] = {
            k: {
                'mean': sum(vals) / len(vals),
                'min': min(vals),
                'max': max(vals),
                'sum': sum(vals),
                'count': len(vals),
            }
            for k, vals in component_summary.items() if vals
        }

    eval_stats['component_benefit_correlation'] = analyze_component_benefit_correlation(
        eval_stats['event_benefit_correlations']
    )
    eval_stats['statistical_reflection'] = analyze_episode_component_statistics(
        eval_stats['episode_component_stats']
    )
    log_func(f"评估完成: 平均总效益={avg_benefit['total']:.2f}")
    return eval_stats
