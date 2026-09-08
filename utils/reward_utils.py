"""奖励函数校验与安全执行包装。"""
import ast
import inspect
import math
import traceback
from collections import Counter, defaultdict

FORBIDDEN_CALL_NAMES = {'open', 'exec', 'eval', 'compile', '__import__', 'input', 'os', 'sys'}
FORBIDDEN_ATTRIBUTE_PREFIXES = {'os.', 'sys.', 'subprocess.', 'pathlib.', 'shutil.', 'socket.', 'requests.'}


def make_safe_compute_reward(fn, log_func=print, components_recorder=None):
    """包装 LLM 生成的 compute_reward，保证不抛异常且返回 (float, dict)。"""
    try:
        sig = inspect.signature(fn)
    except Exception:
        sig = None

    def safe(v, env_metrics, current_cycle, max_cycles):
        try:
            if isinstance(v, dict):
                sanitized_v = v.copy()
            else:
                sanitized_v = {'id': getattr(v, 'id', None)}
            if not isinstance(sanitized_v, dict):
                sanitized_v = {'__orig__': v, 'id': getattr(v, 'id', None)}
            sanitized_v.setdefault('events', {})
            if not isinstance(sanitized_v['events'], dict):
                sanitized_v['events'] = {'__raw_events__': sanitized_v['events']}
            default_fields = {
                'status': 'idle', 'battery': 0.0, 'position': (0.0, 0.0),
                'current_order_id': None, 'target_station_id': None,
                'task_destination_pos': None, 'task_remaining_cycles': 0,
            }
            for field_name, default_value in default_fields.items():
                if sanitized_v.get(field_name) is None:
                    sanitized_v[field_name] = default_value
            ev = sanitized_v['events']
            for k, dv in [
                ('finished_order', False), ('picked_up_order', False),
                ('started_charging', False), ('invalid_action', False),
                ('is_charging_solar', False), ('order_length', 0), ('charge_amount', 0),
            ]:
                ev.setdefault(k, dv)

            if sig is not None:
                params = len(sig.parameters)
                if params >= 4:
                    out = fn(sanitized_v, env_metrics, current_cycle, max_cycles)
                elif params == 3:
                    out = fn(sanitized_v, env_metrics, current_cycle)
                elif params == 2:
                    out = fn(sanitized_v, env_metrics)
                else:
                    out = fn(sanitized_v)
            else:
                out = fn(sanitized_v, env_metrics, current_cycle, max_cycles)

            if out is None:
                return 0.0, {}
            if isinstance(out, tuple) and len(out) == 2:
                r, comps = out
                if r is None:
                    r = 0.0
                if comps is None:
                    comps = {}
                if components_recorder is not None:
                    try:
                        components_recorder(comps, sanitized_v, env_metrics, current_cycle)
                    except Exception:
                        pass
                return float(r), (comps if isinstance(comps, dict) else {})
            if isinstance(out, (int, float)):
                return float(out), {}
            log_func(f"Unexpected compute_reward output type: {type(out)}")
            return 0.0, {}
        except Exception as e:
            log_func(f"compute_reward crashed: {e}\n{traceback.format_exc()}")
            return 0.0, {}

    return safe


def is_code_likely_complete(code: str) -> bool:
    if not isinstance(code, str) or not code.strip():
        return False
    for a, b in [('(', ')'), ('[', ']'), ('{', '}')]:
        if code.count(a) != code.count(b):
            return False
    for q in ("'", '"'):
        if code.count(q) % 2 != 0:
            return False
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def validate_reward_code_structure(code_text):
    diagnostics = []
    try:
        tree = ast.parse(code_text)
    except SyntaxError as exc:
        diagnostics.append(f'SyntaxError: {exc.msg} (line {getattr(exc, "lineno", "?")})')
        return False, diagnostics
    except Exception as exc:
        diagnostics.append(f'无法解析代码: {exc}')
        return False, diagnostics

    compute_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_reward']
    if not compute_defs:
        diagnostics.append('缺少 compute_reward 定义')
        return False, diagnostics
    if len(compute_defs) > 1:
        diagnostics.append('存在多个 compute_reward 定义')
        return False, diagnostics

    func_def = compute_defs[0]
    expected_args = ['vehicle', 'env_metrics', 'current_cycle', 'max_cycles']
    actual_args = [arg.arg for arg in func_def.args.args]
    if actual_args != expected_args:
        diagnostics.append(f'compute_reward 参数必须按顺序 {expected_args}，当前为 {actual_args}')

    components_initialized = False
    for node in ast.walk(func_def):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            diagnostics.append('compute_reward 不允许声明 global/nonlocal')
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            diagnostics.append('compute_reward 内部禁止 import 语句')
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'components':
                    components_initialized = True
        if isinstance(node, ast.Call):
            call_name = None
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                parts = []
                attr = node.func
                while isinstance(attr, ast.Attribute):
                    parts.append(attr.attr)
                    attr = attr.value
                if isinstance(attr, ast.Name):
                    parts.append(attr.id)
                parts.reverse()
                call_name = '.'.join(parts)
            if call_name:
                if call_name in FORBIDDEN_CALL_NAMES:
                    diagnostics.append(f'检测到禁止调用: {call_name}')
                for prefix in FORBIDDEN_ATTRIBUTE_PREFIXES:
                    if call_name.startswith(prefix):
                        diagnostics.append(f'检测到禁止属性访问: {call_name}')

    if not components_initialized:
        diagnostics.append('compute_reward 必须初始化 components 字典')

    return_nodes = [n for n in ast.walk(func_def) if isinstance(n, ast.Return)]
    if not return_nodes:
        diagnostics.append('compute_reward 缺少 return 语句')
    else:
        for node in return_nodes:
            if not isinstance(node.value, ast.Tuple) or len(node.value.elts) != 2:
                diagnostics.append('所有 return 必须返回 (reward, components) 二元组')

    if diagnostics:
        return False, diagnostics
    return True, []


def summarize_component_records(component_records):
    if not component_records:
        return {'stats': {}, 'key_counts': {}, 'anomalies': []}
    values_by_key = defaultdict(list)
    key_counter = Counter()
    anomalies = []
    for rec in component_records:
        if not isinstance(rec, dict):
            anomalies.append('non_dict_record')
            continue
        comps = rec.get('comps')
        if not isinstance(comps, dict):
            anomalies.append('components_not_dict')
            continue
        for key, val in comps.items():
            if not isinstance(key, str):
                anomalies.append('non_string_component_key')
                continue
            key_counter[key] += 1
            try:
                numeric_val = float(val)
            except Exception:
                anomalies.append(f'non_numeric_value:{key}')
                continue
            if not math.isfinite(numeric_val):
                anomalies.append(f'non_finite_value:{key}')
                continue
            values_by_key[key].append(numeric_val)

    per_component_stats = {}
    for key, values in values_by_key.items():
        if not values:
            continue
        per_component_stats[key] = {
            'min': float(min(values)),
            'avg': float(sum(values) / len(values)),
            'max': float(max(values)),
            'abs_mean': float(sum(abs(v) for v in values) / len(values)),
            'samples': len(values),
        }
    return {'stats': per_component_stats, 'key_counts': dict(key_counter), 'anomalies': anomalies}


def compute_fitness(stats: dict) -> float:
    try:
        if 'platform' in stats and 'grid' in stats and 'solar' in stats:
            return float(stats.get('platform', 0) or 0) + float(stats.get('grid', 0) or 0) + float(stats.get('solar', 0) or 0)
        p = float(stats.get('total_platform_order_benefit', 0) or 0)
        g = float(stats.get('total_grid_power_loss_reduction_benefit', 0) or 0) + float(stats.get('total_grid_peak_shaving_benefit', 0) or 0)
        s = float(stats.get('total_solar_station_profit', 0) or 0)
        return p + g + s
    except Exception:
        return 0.0
