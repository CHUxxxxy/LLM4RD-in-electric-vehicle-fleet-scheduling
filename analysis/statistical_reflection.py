"""组件奖励与总效用的 Spearman 相关及线性回归分析。"""
import math
from typing import Any, Dict, List


def _rank_data(values: List[float]) -> List[float]:
    """计算平均秩（处理 ties）。"""
    n = len(values)
    if n == 0:
        return []
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(x: List[float], y: List[float]) -> float:
    """Pearson 相关作用于秩。"""
    if len(x) != len(y) or len(x) < 3:
        return 0.0
    rx = _rank_data(x)
    ry = _rank_data(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - my) ** 2 for b in ry))
    if den_x < 1e-12 or den_y < 1e-12:
        return 0.0
    return num / (den_x * den_y)


def _ols_with_intercept(X: List[List[float]], y: List[float]):
    """最小二乘 y = b0 + b1*x1 + ..."""
    n = len(y)
    if n == 0:
        return [], 0.0
    p = len(X[0]) if X else 0
    dim = p + 1
    ata = [[0.0] * dim for _ in range(dim)]
    aty = [0.0] * dim
    for i in range(n):
        row = [1.0] + X[i]
        for a in range(dim):
            aty[a] += row[a] * y[i]
            for b in range(dim):
                ata[a][b] += row[a] * row[b]
    aug = [ata[i][:] + [aty[i]] for i in range(dim)]
    for col in range(dim):
        pivot = col
        for r in range(col + 1, dim):
            if abs(aug[r][col]) > abs(aug[pivot][col]):
                pivot = r
        if abs(aug[pivot][col]) < 1e-12:
            return [0.0] * dim, 0.0
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        for j in range(col, dim + 1):
            aug[col][j] /= div
        for r in range(dim):
            if r == col:
                continue
            factor = aug[r][col]
            for j in range(col, dim + 1):
                aug[r][j] -= factor * aug[col][j]
    coef = [aug[i][dim] for i in range(dim)]
    y_mean = sum(y) / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    preds = [coef[0] + sum(coef[j + 1] * X[i][j] for j in range(p)) for i in range(n)]
    ss_res = sum((yi - pi) ** 2 for yi, pi in zip(y, preds))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return coef, r2


def _classify_impact(spearman_r: float, regression_coef: float) -> str:
    """影响类别：positive / negative / weak。"""
    sr = abs(spearman_r)
    if sr > 0.3:
        if regression_coef > 0:
            return 'positive'
        if regression_coef < 0:
            return 'negative'
    if sr > 0.1:
        return 'weak'
    return 'weak'


def _mutation_action(impact: str, spearman_r: float, regression_coef: float) -> str:
    """改写动作：retain / reweight / reformulate / remove。"""
    inconsistent = (spearman_r * regression_coef) < 0 and abs(spearman_r) > 0.1 and abs(regression_coef) > 1e-6
    if inconsistent:
        return 'reformulate'
    if impact == 'positive':
        return 'retain' if abs(regression_coef) >= 1.0 else 'reweight'
    if impact == 'negative':
        return 'remove' if abs(spearman_r) > 0.5 else 'reformulate'
    if impact == 'weak':
        return 'reweight'
    return 'reformulate'


def _build_suggestion(action: str, spearman_r: float, regression_coef: float, impact: str) -> str:
    return (
        f"{action}: Spearman={spearman_r:.4f}, linear_regression_coefficient={regression_coef:.4f}, "
        f"impact_classification={impact}"
    )


def analyze_episode_component_statistics(episode_component_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    基于 episode 级组件均值与总效益，计算 Spearman 与线性回归。
    输出字段供反思提示词中的 statistical_analysis 使用。
    """
    if not episode_component_stats or len(episode_component_stats) < 3:
        return {
            'spearman_and_regression': {},
            'component_statistics': {},
            'model_r2': 0.0,
            'note': 'insufficient_samples',
        }

    all_components = set()
    for ep in episode_component_stats:
        all_components.update(ep.get('component_avg', {}).keys())

    totals = [float(ep.get('benefit', {}).get('total', 0)) for ep in episode_component_stats]
    comp_names = sorted(all_components)
    X = []
    for ep in episode_component_stats:
        avg = ep.get('component_avg', {})
        X.append([float(avg.get(c, 0.0) or 0.0) for c in comp_names])

    coef, r2 = _ols_with_intercept(X, totals)
    spearman_and_regression = {}
    component_statistics = {}

    for j, comp_name in enumerate(comp_names):
        comp_vals = [row[j] for row in X]
        sp = spearman_correlation(comp_vals, totals)
        reg_coef = coef[j + 1] if len(coef) > j + 1 else 0.0
        impact = _classify_impact(sp, reg_coef)
        action = _mutation_action(impact, sp, reg_coef)
        entry = {
            'spearman_correlation': round(sp, 4),
            'linear_regression_coefficient': round(reg_coef, 4),
            'impact_classification': impact,
            'mutation_recommendation': action,
            'suggestion': _build_suggestion(action, sp, reg_coef, impact),
        }
        spearman_and_regression[comp_name] = entry
        component_statistics[comp_name] = entry

    return {
        'spearman_and_regression': spearman_and_regression,
        'component_statistics': component_statistics,
        'model_r2': round(r2, 4),
        'regression_intercept': round(coef[0], 4) if coef else 0.0,
        'num_episodes': len(episode_component_stats),
    }
