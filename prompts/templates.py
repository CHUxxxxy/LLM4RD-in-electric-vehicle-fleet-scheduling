"""LLM 提示词：初始奖励生成与统计反思改写。"""

# --- 初始生成 ---

PROMPT1_BACKGROUND_AND_OBJECTIVES = """
[Background and Objectives]
You are an expert in designing reward functions for multi-agent reinforcement learning. Please
design a reward function for the electric ride-hailing fleet dispatching problem. The key
considerations are as follows:
1. The objective is to account for the benefits of three stakeholders: the ride-hailing
platform, the photovoltaic-storage charging station, and the conventional power grid.
2. Each vehicle can take one of three actions: order serving, charging, or remaining idle.
Completing trips generates revenue for the platform, which depends on trip distance.
Charging behavior may contribute to peak shaving and valley filling, and charging at
photovoltaic-storage stations yields additional benefits for those stations. Idle driving
incurs operational costs. Detailed information, including trip revenue calculation, time-
of-use electricity pricing, and charging subsidies, will be provided in tabular form.
3. The environment uses a discrete action space. Therefore, all reward functions must be
compatible with discrete decisions such as order serving, charging, and idling, and should
not introduce reward terms designed for continuous control.
4. The reward function must explicitly include three stakeholder-level components: (i) ride-
hailing platform benefits, (ii) photovoltaic-storage charging station benefits, and (iii)
conventional grid benefits. These components should remain separately identifiable in the
generated code.
"""

PROMPT1_ENVIRONMENT_CONTEXT = """
[Environment Context]
Description of environment entities (data accessible in compute_reward):

- vehicle (dict): state dictionary of an individual vehicle
  o id (int): unique identifier of the vehicle
  o status (str): current status, one of ['idle', 'to_pickup', 'on_trip', 'to_charge', 'charging']
  o battery (float): battery level in [0, 100]
  o position (tuple[float, float]): location coordinates (latitude, longitude)
  o target_station_id (Optional[int]): ID of the target charging station
  o task_destination_pos (Optional[tuple]): task destination coordinates
  o task_remaining_cycles (int): remaining cycles for current task
  o events (dict): per-step event flags and derived order/charging information:
      - finished_order (bool): whether an order was completed this step
      - picked_up_order (bool): whether a new order was assigned this step
      - started_charging (bool): whether charging started this step
      - invalid_action (bool): whether the action was invalid
      - is_charging_solar (bool): whether charging at a PV-storage station
      - order_length (float): trip distance (km), maps to order.distance
      - charge_amount (float): charge amount this step, maps to charging energy

- order (dict): accessible indirectly via vehicle['events'] when order-related events fire
  o id: use vehicle.get('current_order_id')
  o origin / destination: use vehicle.get('task_destination_pos') and order flow status
  o distance (float): vehicle['events']['order_length']
  o fare (float): derive from order_length and platform coefficients in the parameter table

- charging_station (dict): accessible indirectly via vehicle fields and events
  o id: vehicle.get('target_station_id')
  o position: vehicle.get('task_destination_pos') when status is 'to_charge' or 'charging'
  o pv_output: indicated by vehicle['events']['is_charging_solar']
  o charging energy: vehicle['events']['charge_amount']

- grid (dict): accessible via env_metrics
  o is_peak_hour (bool): env_metrics['is_peak_hour']
  o electricity_price (float): env_metrics['electricity_price']
  o peak_to_base_cost_difference: use constant PEAK_TO_BASE_COST_DIFFERENCE from parameter table
  o avg_battery (float): env_metrics['avg_battery']

- env_metrics (dict): environment-level indicators passed as the second argument
- current_cycle (int): current time step index (0 to max_cycles-1)
- max_cycles (int): total number of cycles in one episode
"""

REWARD_PARAMETER_TABLE = """
[Reward Parameter Table]
| Symbol / Constant | Value | Stakeholder | Description |
|---|---:|---|---|
| ORDER_COMPLETION_REWARD_COEFF | 25.0 | Platform | Reward per km when an order is completed |
| NEW_ORDER_ASSIGNMENT_LENGTH_MULTIPLIER | 2.5 | Platform | Weight for newly assigned order length |
| GUROBI_DISTANCE_PENALTY_COEFF | 0.12 | Platform | Distance penalty when serving/charging |
| GUROBI_CHARGE_COEFF | 0.15 | Platform | Charging decision benefit coefficient |
| CURRENT_ELECTRICITY_PRICE_PER_KWH | 0.4 | Grid | Time-of-use electricity price (CNY/kWh) |
| PEAK_TO_BASE_COST_DIFFERENCE | 2.6 | Grid | Peak-shaving / valley-filling coefficient |
| C_LOSS | 1.31 | Grid | Grid loss-reduction benefit when using PV charging |
| SOLAR_ELECTRICITY_PRICE_DISCOUNT | 0.8 | PV-storage | Discount ratio for PV-station charging price |
| SOLAR_FARM_SUBSIDY | 0.25 | PV-storage | PV generation subsidy (CNY/kWh) |
| STORAGE_COST_PER_KWH | 0.05 | PV-storage | Storage cost (CNY/kWh) |
"""

IMPLEMENTATION_CONSTRAINTS = """
[Implementation Constraints - Required for code execution]
1. Function signature MUST be exactly:
   def compute_reward(vehicle, env_metrics, current_cycle, max_cycles):
2. Do NOT use parameter names: state, action, next_state, done, info, observation.
3. Initialize at the start: reward = 0.0 and components = {}
4. Return exactly: return reward, components
5. Record every stakeholder component in components with keys such as:
   platform_*, grid_*, solar_* (float values only)
6. Use vehicle.get('key', default) and events.get('key', default) to avoid KeyError
7. Do NOT import modules inside compute_reward
8. The last line of the function must be: # END_FUNCTION
9. Do NOT introduce variables or physical quantities not listed in [Environment Context]
"""

BASE_REWARD_TEMPLATE = """def compute_reward(vehicle, env_metrics, current_cycle, max_cycles):
    reward = 0.0
    components = {}
    events = vehicle.get('events', {}) or {}
    battery = float(vehicle.get('battery', 0.0) or 0.0)
    metrics = env_metrics or {}
    is_peak_hour = bool(metrics.get('is_peak_hour', False))
    electricity_price = float(metrics.get('electricity_price', 0.4) or 0.4)
    solar_discount = float(metrics.get('solar_discount', 0.8) or 0.8)
    avg_fleet_battery = float(metrics.get('avg_battery', battery) or battery)
    finished_order = bool(events.get('finished_order', False))
    order_length = float(events.get('order_length', 0.0) or 0.0)
    started_charging = bool(events.get('started_charging', False))
    charge_amount = float(events.get('charge_amount', 0.0) or 0.0)
    is_charging_solar = bool(events.get('is_charging_solar', False))
    invalid_action = bool(events.get('invalid_action', False))

    platform_reward = order_length * 25.0 if finished_order else 0.0
    components['platform_order_completion'] = platform_reward
    reward += platform_reward

    grid_penalty = -charge_amount * electricity_price * 10.0 if started_charging and is_peak_hour else 0.0
    components['grid_peak_shaving'] = grid_penalty
    reward += grid_penalty

    solar_reward = charge_amount * 1.31 * solar_discount if is_charging_solar else 0.0
    components['solar_incentive'] = solar_reward
    reward += solar_reward

    battery_penalty = -(avg_fleet_battery - battery) * 0.1 if battery < avg_fleet_battery else 0.0
    components['battery_safety'] = battery_penalty
    reward += battery_penalty

    invalid_penalty = -10.0 if invalid_action else 0.0
    components['invalid_action_penalty'] = invalid_penalty
    reward += invalid_penalty

    final_penalty = -20.0 if (current_cycle >= max_cycles - 1 and battery < 20) else 0.0
    components['final_cycle_penalty'] = final_penalty
    reward += final_penalty

    return reward, components
# END_FUNCTION"""

# --- 统计反思与定向改写 ---

PROMPT2_TRAINING_CONTEXT = """
After multiple rounds of training, the current reward function is associated with the following
information: the mathematical form of each reward component, the component-level reward values
recorded at intermediate policy checkpoints, the scalar values of the overall utility function, and
the statistical analysis results obtained from a separate analysis procedure, including Spearman
correlation coefficients and linear regression coefficients for each reward component.
"""

PROMPT2_STATISTICAL_REFLECTION = """
[Statistical Reflection Objective]
Please use the provided Spearman correlation and linear regression results to perform component-
wise reward reflection. Do not recalculate these statistical metrics. Instead, interpret the provided
results to determine the direction and relative contribution of each reward component and use this
evidence to guide targeted reward mutation.
For each reward component, identify:
1. Spearman correlation result;
2. Linear regression coefficient;
3. Impact classification (positive, negative, or weak);
4. Specific mutation recommendation.
Use the statistical results to determine whether each component should be retained, reweighted,
reformulated, or removed. Components showing consistently positive contributions should generally
be preserved, while weak or negative components should be selectively modified.
"""

PROMPT2_MUTATION_STRATEGIES = """
[Mutation Strategies]
1. For positive_impact_components: retain or slightly adjust weights;
2. For negative_impact_components: examine the logic and adjust weights, modify mathematical
   forms, or remove them;
3. For weak_components: increase weights or reformulate their mathematical expressions.
4. Targeted mutation: modify only the components identified by the statistical results rather
   than indiscriminately regenerating the entire reward function.
5. When Spearman correlation and regression results are inconsistent, interpret the two results
   jointly and avoid making aggressive structural changes without sufficient evidence.
6. Preserve the three stakeholder-level reward categories, discrete-action compatibility, and
   dense or staged feedback for delayed objectives during mutation.
"""

PROMPT2_OUTPUT_REQUIREMENTS = """
[Output Requirements]
- Directly output a complete Python code block (starting with def compute_reward);
- Ensure that all components are recorded in the components dictionary;
- Pay particular attention to the suggestion field in component_real_impact for each component.
- For each component, the suggestion field should state the recommended mutation action (retain,
  reweight, reformulate, or remove) based on the provided statistical results;
- Use the provided statistical analysis as the basis for component-wise mutation and do not
  independently perform another statistical analysis;
- Do not introduce variables or physical quantities that are not available in the provided
  environment context;
"""

PROMPT2_ENVIRONMENT_CONTEXT = PROMPT1_ENVIRONMENT_CONTEXT


def build_initial_prompt() -> str:
    """组装初始奖励函数生成提示。"""
    return f"""{PROMPT1_BACKGROUND_AND_OBJECTIVES}
{PROMPT1_ENVIRONMENT_CONTEXT}
{REWARD_PARAMETER_TABLE}
{IMPLEMENTATION_CONSTRAINTS}

[Reference Template]
```python
{BASE_REWARD_TEMPLATE}
```

Please provide the reward function:
"""


def build_reflection_prompt(best_code: str, feedback_json: str) -> str:
    """组装反思改写提示：当前代码 + 统计反馈 + 变异策略。"""
    return f"""{PROMPT2_TRAINING_CONTEXT}

[Current Best Reward Function]
```python
{best_code}
```

[Training and Statistical Feedback JSON]
The JSON below contains:
- three_way_benefit / overall_utility: scalar values of the overall utility function
- component_level_statistics: component-level reward values recorded during evaluation
- statistical_analysis: Spearman correlation and linear regression coefficients per component
- component_real_impact: per-component suggestion field (retain / reweight / reformulate / remove)
- optimization_guidance: positive_impact_components, negative_impact_components, weak_components

{feedback_json}

{PROMPT2_STATISTICAL_REFLECTION}
{PROMPT2_MUTATION_STRATEGIES}
{PROMPT2_ENVIRONMENT_CONTEXT}
{REWARD_PARAMETER_TABLE}
{IMPLEMENTATION_CONSTRAINTS}
{PROMPT2_OUTPUT_REQUIREMENTS}

Please provide the revised reward function:
"""


SYSTEM_PROMPT = (
    "You are an expert in designing reward functions for multi-agent reinforcement learning "
    "within the LLM4RD framework for electric ride-hailing fleet dispatching. "
    "You MUST implement: def compute_reward(vehicle, env_metrics, current_cycle, max_cycles) "
    "and return (reward, components). "
    "Keep platform, PV-storage, and grid components separately identifiable. "
    "Use only variables available in the provided environment context."
)
