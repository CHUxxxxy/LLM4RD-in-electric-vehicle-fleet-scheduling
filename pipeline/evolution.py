"""LLM4RD 主进化流程：LLM 生成奖励 -> MAPPO 训练 -> 统计反思 -> 定向变异。"""
import json
import math
import os
import random
import time
import traceback

import numpy as np
import torch

from LLM4RD import config as cfg
from LLM4RD.analysis.feedback import build_mutation_feedback
from LLM4RD.env import RLEnv
from LLM4RD.llm.reward_designer import LLM4RDRewardDesigner
from LLM4RD.rl.evaluator import evaluate_policy
from LLM4RD.rl.trainer import train_mappo_with_reward
from LLM4RD.utils.reward_utils import (
    is_code_likely_complete,
    make_safe_compute_reward,
    validate_reward_code_structure,
)


def _shared_exec_symbols():
    return {
        '__builtins__': __builtins__,
        'np': np,
        'torch': torch,
        'math': math,
        'random': random,
        'NEW_ORDER_ASSIGNMENT_LENGTH_MULTIPLIER': cfg.NEW_ORDER_ASSIGNMENT_LENGTH_MULTIPLIER,
        'GUROBI_DISTANCE_PENALTY_COEFF': cfg.GUROBI_DISTANCE_PENALTY_COEFF,
        'ORDER_COMPLETION_REWARD_COEFF': cfg.ORDER_COMPLETION_REWARD_COEFF,
        'CURRENT_ELECTRICITY_PRICE_PER_KWH': cfg.CURRENT_ELECTRICITY_PRICE_PER_KWH,
        'SOLAR_ELECTRICITY_PRICE_DISCOUNT': cfg.SOLAR_ELECTRICITY_PRICE_DISCOUNT,
        'C_LOSS': cfg.C_LOSS,
        'PEAK_TO_BASE_COST_DIFFERENCE': cfg.PEAK_TO_BASE_COST_DIFFERENCE,
        'SOLAR_FARM_SUBSIDY': cfg.SOLAR_FARM_SUBSIDY,
        'STORAGE_COST_PER_KWH': cfg.STORAGE_COST_PER_KWH,
        'STORAGE_COST_CALC_INTERVAL': cfg.STORAGE_COST_CALC_INTERVAL,
        'MAX_CYCLES': cfg.MAX_CYCLES,
    }


def _compile_reward(code: str, log_func):
    struct_ok, diags = validate_reward_code_structure(code)
    if not struct_ok:
        return None, '; '.join(diags) if diags else '结构校验失败'

    scope = dict(_shared_exec_symbols())
    exec(code, scope, scope)
    raw_fn = scope['compute_reward']
    return make_safe_compute_reward(raw_fn, log_func=log_func), None


def _validate_sample(compute_reward):
    sample_v = {
        'id': 0, 'status': 'idle', 'battery': 50.0,
        'events': {
            'finished_order': False, 'picked_up_order': False,
            'started_charging': False, 'invalid_action': False,
            'is_charging_solar': False, 'order_length': 0.0, 'charge_amount': 0.0,
        },
    }
    sample_metrics = {
        'is_peak_hour': True,
        'electricity_price': cfg.CURRENT_ELECTRICITY_PRICE_PER_KWH,
        'avg_battery': 50.0,
    }
    val = compute_reward(sample_v, sample_metrics, 0, cfg.MAX_CYCLES)
    if not (isinstance(val, tuple) and len(val) == 2):
        raise ValueError('compute_reward 必须返回 (reward, components)')


def run_llm4rd(excel_path: str):
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)

    designer = LLM4RDRewardDesigner()
    results_log = os.path.join(cfg.OUTPUT_DIR, 'llm4rd_results.txt')
    summary_path = os.path.join(cfg.OUTPUT_DIR, f'llm4rd_run_{int(time.time())}.jsonl')

    def write_log(msg):
        print(msg)
        with open(results_log, 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')

    def write_summary(entry: dict):
        with open(summary_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    write_log(f"\n=== LLM4RD 运行开始 {time.asctime()} ===")
    write_log(
        f"配置: iterations={cfg.EVOLUTION_ITERATIONS}, "
        f"candidates={cfg.CANDIDATES_PER_ITERATION}, "
        f"training_steps={cfg.MAPPO_TRAINING_STEPS}"
    )

    current_rewards = designer.generate_initial_rewards(cfg.CANDIDATES_PER_ITERATION)
    if not current_rewards:
        write_log("错误: 未生成任何初始奖励函数，请检查 LLM 服务。")
        return None

    best_code = None
    best_score = -float('inf')
    baseline_reported = False

    for iter_i in range(cfg.EVOLUTION_ITERATIONS):
        write_log(f"\n--- 迭代 {iter_i + 1}/{cfg.EVOLUTION_ITERATIONS} ---")
        performances = []

        if not baseline_reported:
            try:
                write_log("=== MAPPO 手工奖励基线 ===")
                env_train = RLEnv(excel_path, n_vehicles=cfg.N_VEHICLES, n_stations=cfg.N_STATIONS)
                ac, _ = train_mappo_with_reward(
                    env_train, compute_reward=None,
                    training_steps=cfg.MAPPO_TRAINING_STEPS, log_func=write_log,
                )
                env_eval = RLEnv(excel_path, n_vehicles=cfg.N_VEHICLES, n_stations=cfg.N_STATIONS)
                bl_stats = evaluate_policy(env_eval, ac, log_func=write_log)
                bl_benefit = bl_stats['avg_benefit']
                write_log(
                    f"基线三方效益: 总={bl_benefit['total']:.2f}, "
                    f"平台={bl_benefit['platform']:.2f}, 电网={bl_benefit['grid']:.2f}, "
                    f"光伏={bl_benefit['solar']:.2f}"
                )
                write_summary({'event': 'baseline', 'benefit': bl_benefit})
            except Exception as e:
                write_log(f"基线训练/评估失败: {e}\n{traceback.format_exc()}")
            baseline_reported = True

        for code_idx, code in enumerate(current_rewards):
            if not is_code_likely_complete(code):
                write_log(f"候选 {code_idx + 1} 代码不完整，跳过")
                fail_path = os.path.join(cfg.FAILED_DIR, f'incomplete_iter{iter_i}_idx{code_idx}.py')
                with open(fail_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                performances.append({'total': -1e9, 'stats': {}, 'code': code})
                continue

            compute_reward, err = _compile_reward(code, write_log)
            if compute_reward is None:
                write_log(f"候选 {code_idx + 1} 编译/校验失败: {err}")
                performances.append({'total': -1e9, 'stats': {}, 'code': code})
                continue

            try:
                _validate_sample(compute_reward)
            except Exception as e:
                write_log(f"候选 {code_idx + 1} 样本验证失败: {e}")
                performances.append({'total': -1e9, 'stats': {}, 'code': code})
                continue

            write_log(f"=== 训练候选 {code_idx + 1} ===")
            try:
                env_train = RLEnv(
                    excel_path, n_vehicles=cfg.N_VEHICLES, n_stations=cfg.N_STATIONS,
                    external_reward_fn=compute_reward,
                )
                actor_critic, train_stats = train_mappo_with_reward(
                    env_train, compute_reward,
                    training_steps=cfg.MAPPO_TRAINING_STEPS, log_func=write_log,
                )
                env_eval = RLEnv(excel_path, n_vehicles=cfg.N_VEHICLES, n_stations=cfg.N_STATIONS)
                eval_stats = evaluate_policy(
                    env_eval, actor_critic, compute_reward=compute_reward, log_func=write_log,
                )
            except Exception as e:
                write_log(f"候选 {code_idx + 1} 训练/评估异常: {e}\n{traceback.format_exc()}")
                performances.append({'total': -1e9, 'stats': {}, 'code': code})
                continue

            avg_benefit = eval_stats['avg_benefit']
            total_benefit = avg_benefit['total']
            avg_score = total_benefit / float(cfg.N_VEHICLES)
            perf_stats = {
                'platform': avg_benefit['platform'],
                'grid': avg_benefit['grid'],
                'solar': avg_benefit['solar'],
                'completed_orders': eval_stats.get('avg_completed_orders', 0),
            }
            write_log(
                f"候选 {code_idx + 1} 结果: 总效益={total_benefit:.2f}, "
                f"每车={avg_score:.6f}, 训练回合={len(train_stats.get('episode_rewards', []))}"
            )
            performances.append({
                'total': avg_score,
                'stats': perf_stats,
                'code': code,
                'eval_stats': eval_stats,
            })
            write_summary({
                'event': 'candidate_eval',
                'iteration': iter_i,
                'candidate': code_idx + 1,
                'score': avg_score,
                'benefit': avg_benefit,
            })

        if not performances:
            current_rewards = designer.generate_initial_rewards(cfg.CANDIDATES_PER_ITERATION)
            continue

        scores = [p['total'] for p in performances]
        best_idx = int(np.argmax(scores))
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_code = performances[best_idx]['code']
            write_log(f"发现更优奖励函数，得分={best_score:.6f}")

        if iter_i < cfg.EVOLUTION_ITERATIONS - 1 and best_code:
            best_perf = performances[best_idx]
            eval_stats = best_perf.get('eval_stats', {})
            feedback_obj = build_mutation_feedback(
                best_score=scores[best_idx],
                perf_stats=best_perf.get('stats', {}),
                statistical_analysis=eval_stats.get('statistical_reflection', {}),
                component_benefit_correlation=eval_stats.get('component_benefit_correlation', {}),
                component_analysis=eval_stats.get('component_analysis', {}),
                best_code=best_code,
            )
            feedback_json = json.dumps(feedback_obj, ensure_ascii=False, indent=2)
            write_log(f"统计反思反馈:\n{feedback_json}")
            write_summary({'event': 'reflection_feedback', 'iteration': iter_i, 'feedback': feedback_obj})
            new_rewards = designer.generate_mutated_rewards(
                best_code, feedback_json, cfg.CANDIDATES_PER_ITERATION,
            )
            current_rewards = new_rewards if new_rewards else [best_code]
        elif best_code:
            current_rewards = [best_code]

    if best_code:
        with open(cfg.BEST_REWARD_PATH, 'w', encoding='utf-8') as f:
            f.write(best_code)
        write_log(f"最佳奖励函数已保存: {cfg.BEST_REWARD_PATH}")
        write_summary({'event': 'run_end', 'best_score': best_score, 'path': cfg.BEST_REWARD_PATH})
    else:
        write_log("LLM4RD 未能找到有效奖励函数")

    write_log(f"=== LLM4RD 运行结束 {time.asctime()} ===")
    return best_code
