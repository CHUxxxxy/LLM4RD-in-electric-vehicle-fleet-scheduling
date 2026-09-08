"""构建反思阶段交给 LLM 的结构化反馈。"""
from typing import Any, Dict, Optional


def build_mutation_feedback(
    best_score: float,
    perf_stats: Dict[str, Any],
    statistical_analysis: Dict[str, Any],
    component_benefit_correlation: Dict[str, Any],
    component_analysis: Optional[Dict[str, Any]] = None,
    best_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    合并统计反思与事件关联分析，供 LLM 定向改写奖励组件。
    suggestion 取值为 retain / reweight / reformulate / remove。
    """
    paper_stats = statistical_analysis.get('spearman_and_regression') or statistical_analysis.get('component_statistics', {})
    component_effectiveness = component_benefit_correlation.get('component_effectiveness', {})
    component_real_impact = {}

    all_comp_names = set(component_effectiveness.keys()) | set(paper_stats.keys())
    if component_analysis:
        all_comp_names |= set(component_analysis.keys())

    for comp_name in sorted(all_comp_names):
        stat_data = paper_stats.get(comp_name, {})
        eff_data = component_effectiveness.get(comp_name, {})
        level_stats = (component_analysis or {}).get(comp_name, {})

        sp = stat_data.get('spearman_correlation')
        reg_coef = stat_data.get('linear_regression_coefficient')
        impact = stat_data.get('impact_classification', 'weak')
        action = stat_data.get('mutation_recommendation', 'reweight')

        # suggestion 使用 retain / reweight / reformulate / remove
        if stat_data.get('suggestion'):
            suggestion = stat_data['suggestion']
        else:
            suggestion = f"{action}: impact={impact}"

        breakdown = eff_data.get('benefit_breakdown', {})
        max_benefit_type = 'unknown'
        max_benefit_val = 0.0
        for btype, bval in breakdown.items():
            if bval > max_benefit_val:
                max_benefit_val = bval
                max_benefit_type = btype

        component_real_impact[comp_name] = {
            'spearman_correlation': sp,
            'linear_regression_coefficient': reg_coef,
            'impact_classification': impact,
            'mutation_recommendation': action,
            'suggestion': suggestion,
            'component_level_mean': level_stats.get('mean'),
            'component_level_sum': level_stats.get('sum'),
            'event_effectiveness': eff_data.get('effectiveness'),
            'total_benefit_contributed': round(eff_data.get('total_associated_benefit', 0), 2),
            'main_benefit_type': max_benefit_type,
            'benefit_breakdown': {k: round(v, 2) for k, v in breakdown.items()},
        }

    positive = [
        k for k, v in component_real_impact.items()
        if v.get('impact_classification') == 'positive'
        or (
            v.get('mutation_recommendation') in ('retain', 'reweight')
            and v.get('impact_classification') != 'negative'
        )
    ]
    negative = [
        k for k, v in component_real_impact.items()
        if v.get('impact_classification') == 'negative'
        or v.get('mutation_recommendation') == 'remove'
    ]
    weak = [
        k for k, v in component_real_impact.items()
        if v.get('impact_classification') == 'weak'
        and k not in positive and k not in negative
    ]

    platform = float(perf_stats.get('platform', 0))
    grid = float(perf_stats.get('grid', 0))
    solar = float(perf_stats.get('solar', 0))

    return {
        'best_score': float(best_score),
        'overall_utility': {
            'total': platform + grid + solar,
            'platform': platform,
            'grid': grid,
            'solar': solar,
        },
        'three_way_benefit': {
            'platform': platform,
            'grid': grid,
            'solar': solar,
        },
        'completed_orders': perf_stats.get('completed_orders', 0),
        'component_level_statistics': component_analysis or {},
        'statistical_analysis': statistical_analysis,
        'component_real_impact': component_real_impact,
        'optimization_guidance': {
            'positive_impact_components': positive,
            'negative_impact_components': negative,
            'weak_components': weak,
        },
        'current_reward_function_included': best_code is not None,
    }
