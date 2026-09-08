"""奖励组件与业务事件效益的关联分析。"""
from collections import defaultdict


def analyze_component_benefit_correlation(event_correlations):
    if not event_correlations:
        return {}

    events_by_type = defaultdict(list)
    for evt in event_correlations:
        events_by_type[evt.get('event_type', 'unknown')].append(evt)

    correlation_analysis = {'by_event_type': {}, 'component_effectiveness': {}}
    all_components = set()
    for evt in event_correlations:
        all_components.update(evt.get('components_at_event', {}).keys())

    component_benefit_accum = {comp: {
        'total_benefit_when_positive': 0,
        'total_benefit_when_negative': 0,
        'count_positive': 0,
        'count_negative': 0,
        'associated_platform': 0,
        'associated_grid': 0,
        'associated_solar': 0,
        'event_count': 0,
    } for comp in all_components}

    for evt_type, events in events_by_type.items():
        evt_analysis = {
            'count': len(events),
            'total_benefit': {'platform': 0, 'grid': 0, 'solar': 0, 'total': 0},
            'avg_benefit_per_event': {'platform': 0, 'grid': 0, 'solar': 0, 'total': 0},
            'component_stats_at_event': {},
        }
        comp_values_at_event = defaultdict(list)
        for evt in events:
            benefit = evt.get('actual_benefit', {})
            evt_analysis['total_benefit']['platform'] += benefit.get('platform', 0)
            evt_analysis['total_benefit']['grid'] += benefit.get('grid', 0)
            evt_analysis['total_benefit']['solar'] += benefit.get('solar', 0)
            comps = evt.get('components_at_event', {})
            for comp_name, comp_val in comps.items():
                try:
                    val = float(comp_val)
                    comp_values_at_event[comp_name].append(val)
                    total_evt_benefit = benefit.get('platform', 0) + benefit.get('grid', 0) + benefit.get('solar', 0)
                    acc = component_benefit_accum[comp_name]
                    if val > 0:
                        acc['total_benefit_when_positive'] += total_evt_benefit
                        acc['count_positive'] += 1
                    elif val < 0:
                        acc['total_benefit_when_negative'] += total_evt_benefit
                        acc['count_negative'] += 1
                    acc['associated_platform'] += benefit.get('platform', 0)
                    acc['associated_grid'] += benefit.get('grid', 0)
                    acc['associated_solar'] += benefit.get('solar', 0)
                    acc['event_count'] += 1
                except Exception:
                    pass
        evt_analysis['total_benefit']['total'] = (
            evt_analysis['total_benefit']['platform'] +
            evt_analysis['total_benefit']['grid'] +
            evt_analysis['total_benefit']['solar']
        )
        if events:
            for k in ['platform', 'grid', 'solar', 'total']:
                evt_analysis['avg_benefit_per_event'][k] = evt_analysis['total_benefit'][k] / len(events)
        for comp_name, vals in comp_values_at_event.items():
            if vals:
                evt_analysis['component_stats_at_event'][comp_name] = {
                    'mean_at_event': sum(vals) / len(vals),
                    'min_at_event': min(vals),
                    'max_at_event': max(vals),
                }
        correlation_analysis['by_event_type'][evt_type] = evt_analysis

    for comp_name, accum in component_benefit_accum.items():
        if accum['event_count'] == 0:
            continue
        avg_benefit_positive = (
            accum['total_benefit_when_positive'] / accum['count_positive']
            if accum['count_positive'] > 0 else 0
        )
        avg_benefit_negative = (
            accum['total_benefit_when_negative'] / accum['count_negative']
            if accum['count_negative'] > 0 else 0
        )
        total_associated = accum['associated_platform'] + accum['associated_grid'] + accum['associated_solar']
        effectiveness = 'unknown'
        effectiveness_score = 0
        if accum['count_positive'] > 0 and accum['count_negative'] > 0:
            benefit_diff = avg_benefit_positive - avg_benefit_negative
            if benefit_diff > 0.1:
                effectiveness = 'positive_correlation'
                effectiveness_score = benefit_diff
            elif benefit_diff < -0.1:
                effectiveness = 'negative_correlation'
                effectiveness_score = benefit_diff
            else:
                effectiveness = 'neutral'
        elif accum['count_positive'] > 5:
            effectiveness = 'positive_when_active' if avg_benefit_positive > 0.1 else 'weak_positive'
            effectiveness_score = avg_benefit_positive
        elif accum['count_negative'] > 5:
            effectiveness = 'effective_as_penalty' if avg_benefit_negative > 0.1 else 'weak_penalty'
            effectiveness_score = avg_benefit_negative

        correlation_analysis['component_effectiveness'][comp_name] = {
            'effectiveness': effectiveness,
            'effectiveness_score': effectiveness_score,
            'total_associated_benefit': total_associated,
            'benefit_breakdown': {
                'platform': accum['associated_platform'],
                'grid': accum['associated_grid'],
                'solar': accum['associated_solar'],
            },
            'positive_count': accum['count_positive'],
            'negative_count': accum['count_negative'],
        }
    return correlation_analysis
