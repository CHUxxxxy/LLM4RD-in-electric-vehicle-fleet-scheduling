
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import pandas as pd
from collections import deque, namedtuple
import os

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


from LLM4RD import config as cfg

TRAVEL_SPEED = cfg.TRAVEL_SPEED
MAX_CYCLES = cfg.MAX_CYCLES
N_VEHICLES = cfg.N_VEHICLES
N_STATIONS = cfg.N_STATIONS
BATTERY_THRESHOLD_FOR_ORDER = cfg.BATTERY_THRESHOLD_FOR_ORDER
BATTERY_CHARGE_PER_CYCLE = cfg.BATTERY_CHARGE_PER_CYCLE
BATTERY_DRAIN_MOVING = cfg.BATTERY_DRAIN_MOVING
BATTERY_TARGET_CHARGE_LEVEL = cfg.BATTERY_TARGET_CHARGE_LEVEL
MAX_VEHICLES_PER_STATION = cfg.MAX_VEHICLES_PER_STATION
MIN_CHARGE_CYCLES = cfg.MIN_CHARGE_CYCLES
GUROBI_CHARGE_COEFF = cfg.GUROBI_CHARGE_COEFF
NEW_ORDER_ASSIGNMENT_LENGTH_MULTIPLIER = cfg.NEW_ORDER_ASSIGNMENT_LENGTH_MULTIPLIER
GUROBI_DISTANCE_PENALTY_COEFF = cfg.GUROBI_DISTANCE_PENALTY_COEFF
ORDER_COMPLETION_REWARD_COEFF = cfg.ORDER_COMPLETION_REWARD_COEFF
SOLAR_ELECTRICITY_PRICE_DISCOUNT = cfg.SOLAR_ELECTRICITY_PRICE_DISCOUNT
C_LOSS = cfg.C_LOSS
PEAK_TO_BASE_COST_DIFFERENCE = cfg.PEAK_TO_BASE_COST_DIFFERENCE
SOLAR_FARM_SUBSIDY = cfg.SOLAR_FARM_SUBSIDY
STORAGE_COST_PER_KWH = cfg.STORAGE_COST_PER_KWH
STORAGE_COST_CALC_INTERVAL = cfg.STORAGE_COST_CALC_INTERVAL
CLIP_PARAM = cfg.CLIP_PARAM
GAMMA = cfg.GAMMA
GAE_LAMBDA = cfg.GAE_LAMBDA
VF_COEFF = cfg.VF_COEFF
ENT_COEFF = cfg.ENT_COEFF
MAX_GRAD_NORM = cfg.MAX_GRAD_NORM
HIDDEN_DIM = cfg.HIDDEN_DIM
LR = cfg.LR
PPO_STEPS = cfg.PPO_STEPS
PPO_EPOCHS = cfg.PPO_EPOCHS
MINI_BATCH_SIZE = cfg.MINI_BATCH_SIZE
MAX_TRAINING_STEPS = cfg.MAPPO_TRAINING_STEPS
EVAL_INTERVAL_STEPS = 50000
NUM_EVAL_EPISODES = cfg.EVALUATION_EPISODES

# --- Helper Functions ---
def haversine(c1, c2):
    if c1 is None or c2 is None: return float('inf')
    try:
        lat1, lon1 = map(math.radians, c1); lat2, lon2 = map(math.radians, c2)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        a = min(1.0, a); a = max(0.0, a)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        r = 6371
        return c * r
    except Exception as e:
        print(f"Haversine 计算错误: c1={c1}, c2={c2}, error={e}"); return float('inf')


# --- Environment ---
class RLEnv:
    def __init__(self, excel_path, n_vehicles=N_VEHICLES, n_stations=N_STATIONS, external_reward_fn=None):
        # external_reward_fn: optional callable compute_reward(vehicle, env_metrics, current_cycle, max_cycles)
        self.n_vehicles = n_vehicles; self.n_stations = n_stations
        self.travel_speed = TRAVEL_SPEED; self.max_cycles = MAX_CYCLES
        self.battery_threshold_order = BATTERY_THRESHOLD_FOR_ORDER
        self.battery_charge_per_cycle = BATTERY_CHARGE_PER_CYCLE
        self.battery_drain_moving = BATTERY_DRAIN_MOVING
        self.battery_target_charge = BATTERY_TARGET_CHARGE_LEVEL
        self.max_vehicles_per_station = MAX_VEHICLES_PER_STATION
        self.excel_path = excel_path
        self.max_orders_in_state_action = 20
        self.n_closest_stations_in_state = 10
        self._load_initial_order_data()
        # 空闲 + 订单 N + 最近充电站 M
        self.action_dim = 1 + self.max_orders_in_state_action + self.n_closest_stations_in_state
        self.state_dim = 1 + 1 + 6 + 2 + (3 * self.max_orders_in_state_action) + (3 * self.n_closest_stations_in_state)
        self._initialize_stations(); self.current_cycle = 0; self.active_orders_pool = []
        self.vehicles = []
        # 可选外部奖励：compute_reward(vehicle, env_metrics, current_cycle, max_cycles)
        self.external_reward_fn = external_reward_fn
        self.total_grid_power_loss_reduction_benefit = 0
        self.total_grid_peak_shaving_benefit = 0
        self.total_solar_station_profit = 0
        self.total_platform_order_benefit = 0
        self.total_platform_charge_benefit = 0
        self.completed_orders_this_episode = 0 

        print(f"RLEnv Initialized: state_dim={self.state_dim}, action_dim={self.action_dim}")

    def _load_initial_order_data(self):
        try:
            df = pd.read_excel(self.excel_path)
            # 选取 0-24 点的订单
            df_12h = df[(df['SHour'] >= 0) & (df['SHour'] < cfg.HORIZON_HOURS)].copy()
            if df_12h.empty:
                print("Warning: No orders found for SHour 0-23 in the Excel file.")
                self.min_lat, self.max_lat, self.min_lon, self.max_lon = 30.6, 30.8, 104.0, 104.1
                self.max_dist_norm = 50.0
                return
            cycles_per_hour = cfg.CYCLES_PER_HOUR
            cycle_duration_seconds = cfg.CYCLE_MINUTES * 60  # 每周期5分钟=300秒
            df_12h['length_km'] = df_12h['TD'] / 1000.0
            temp_orders_by_cycle = {i: [] for i in range(self.max_cycles)}
            temp_all_orders = []
            current_order_id_counter = 0
            for _, r in df_12h.iterrows():
                # 计算订单所属周期
                hour = int(r['SHour'])
                second_in_hour = r['STime'] % 3600
                cycle_in_hour = int(second_in_hour // cycle_duration_seconds)
                cyc = hour * cycles_per_hour + cycle_in_hour
                if cyc < self.max_cycles:
                    if not (math.isnan(r['SLat']) or math.isnan(r['SLon']) or math.isnan(r['ELat']) or math.isnan(r['ELon'])):
                        order_data = {'id': current_order_id_counter,'pickup': (r['SLat'], r['SLon']),'destination': (r['ELat'], r['ELon']),'length': r['length_km'],'status': 'unassigned','original_cycle': cyc,'duration_cycles': max(1, math.ceil(r['length_km'] / self.travel_speed)),'assigned_vehicle_id': None,'pickup_travel_remaining_cycles': 0,'trip_remaining_cycles': 0,'start_time': r['STime']}
                        temp_orders_by_cycle[cyc].append(order_data); temp_all_orders.append(order_data); current_order_id_counter += 1
            all_lats = [o['pickup'][0] for cyc_orders in temp_orders_by_cycle.values() for o in cyc_orders] + [o['destination'][0] for cyc_orders in temp_orders_by_cycle.values() for o in cyc_orders]
            all_lons = [o['pickup'][1] for cyc_orders in temp_orders_by_cycle.values() for o in cyc_orders] + [o['destination'][1] for cyc_orders in temp_orders_by_cycle.values() for o in cyc_orders]
            if not all_lats or not all_lons:
                    self.min_lat, self.max_lat, self.min_lon, self.max_lon = 30.6, 30.8, 104.0, 104.1
                    print("Warning: No valid order coordinates found, using default map boundaries.")
                    self.max_dist_norm = 50.0
            else:
                    self.min_lat, self.max_lat = min(all_lats), max(all_lats); self.min_lon, self.max_lon = min(all_lons), max(all_lons)
                    print(f"Map boundaries set: Lat ({self.min_lat:.4f}, {self.max_lat:.4f}), Lon ({self.min_lon:.4f}, {self.max_lon:.4f})")
                    self.max_dist_norm = haversine((self.min_lat, self.min_lon), (self.max_lat, self.max_lon))
                    if self.max_dist_norm < 1e-3: self.max_dist_norm = 50.0
            print(f"Max distance for normalization: {self.max_dist_norm:.2f} km")
        except FileNotFoundError:
            print(f"Error: Excel file not found at {self.excel_path}"); exit()
        except KeyError as e: print(f"Error: Missing expected column in Excel: {e}"); exit()

    def _initialize_stations(self):
        self.stations = []
        num_solar_stations = min(cfg.N_PSC_STATIONS, self.n_stations)
        solar_station_indices = random.sample(range(self.n_stations), num_solar_stations)

        if abs(self.max_lat - self.min_lat) < 1e-6 or abs(self.max_lon - self.min_lon) < 1e-6:
            print("Warning: Map boundaries have zero area. Placing stations at default fixed locations.")
            default_min_lat, default_max_lat = 30.6, 30.8; default_min_lon, default_max_lon = 104.0, 104.1
            for i in range(self.n_stations):
                lat_step = (default_max_lat - default_min_lat) / max(1, self.n_stations -1) if self.n_stations > 1 else 0
                lon_step = (default_max_lon - default_min_lon) / max(1, self.n_stations -1) if self.n_stations > 1 else 0
                lat = default_min_lat + i * lat_step; lon = default_min_lon + i * lon_step
                is_solar = True if i in solar_station_indices else False
                self.stations.append({'id': i, 'position': (lat, lon), 'charging_vehicle_ids': [], 'is_solar': is_solar, 'solar_power_output': 0.0, 'stored_energy': 0.0})
        else:
            for i in range(self.n_stations):
                is_solar = True if i in solar_station_indices else False
                self.stations.append({'id': i, 'position': (random.uniform(self.min_lat, self.max_lat), random.uniform(self.min_lon, self.max_lon)), 'charging_vehicle_ids': [], 'is_solar': is_solar, 'solar_power_output': 0.0, 'stored_energy': 0.0})
        print(f"Initialized {num_solar_stations} solar stations out of {self.n_stations} total stations.")

    def _normalize_lat(self, lat): return (lat - self.min_lat) / (self.max_lat - self.min_lat) if (self.max_lat - self.min_lat) > 1e-6 else 0.0
    def _normalize_lon(self, lon): return (lon - self.min_lon) / (self.max_lon - self.min_lon) if (self.max_lon - self.min_lon) > 1e-6 else 0.0

    def reset(self):
        self.current_cycle = 0; self.vehicles = []
        for i in range(self.n_vehicles):
            start_lat = random.uniform(self.min_lat, self.max_lat) if abs(self.max_lat - self.min_lat) > 1e-6 else self.min_lat
            start_lon = random.uniform(self.min_lon, self.max_lon) if abs(self.max_lon - self.min_lon) > 1e-6 else self.min_lon
            self.vehicles.append({'id': i,'position': (start_lat, start_lon), 'battery': random.uniform(0, 80),'status': 'idle','current_order_id': None,'target_station_id': None,'task_destination_pos': None,'task_remaining_cycles': 0, 'charging_cycles_remaining': 0})
        for station in self.stations:
            station['charging_vehicle_ids'] = []
            station['solar_power_output'] = 0.0
            station['stored_energy'] = 0.0
        self.orders_by_cycle = {i: [] for i in range(self.max_cycles)}; self.all_orders_master_list = []
        try:
            df = pd.read_excel(self.excel_path)
            # 选取 0-24 点的订单
            df_12h = df[(df['SHour'] >= 0) & (df['SHour'] < cfg.HORIZON_HOURS)].copy()
            if not df_12h.empty:
                cycles_per_hour = cfg.CYCLES_PER_HOUR  # 每小时12个周期
                cycle_duration_seconds = cfg.CYCLE_MINUTES * 60  # 每周期5分钟=300秒
                df_12h['length_km'] = df_12h['TD'] / 1000.0; current_order_id_counter = 0
                for _, r in df_12h.iterrows():
                    # 计算订单所属周期
                    hour = int(r['SHour'])
                    second_in_hour = r['STime'] % 3600
                    cycle_in_hour = int(second_in_hour // cycle_duration_seconds)
                    cyc = hour * cycles_per_hour + cycle_in_hour
                    if cyc < self.max_cycles:
                        if not (math.isnan(r['SLat']) or math.isnan(r['SLon']) or math.isnan(r['ELat']) or math.isnan(r['ELon'])):
                            order_data = {'id': current_order_id_counter,'pickup': (r['SLat'], r['SLon']),'destination': (r['ELat'], r['ELon']),'length': r['length_km'],'status': 'unassigned','original_cycle': cyc,'duration_cycles': max(1, math.ceil(r['length_km'] / self.travel_speed)),'assigned_vehicle_id': None,'pickup_travel_remaining_cycles': 0,'trip_remaining_cycles': 0,'start_time': r['STime']}
                            self.orders_by_cycle[cyc].append(order_data); self.all_orders_master_list.append(order_data); current_order_id_counter += 1
        except FileNotFoundError: print(f"Error (reset): Excel file not found at {self.excel_path}")
        except KeyError as e: print(f"Error (reset): Missing expected column in Excel: {e}"); exit()
        self.active_orders_pool = []
        if 0 in self.orders_by_cycle: self.active_orders_pool.extend([o for o in self.orders_by_cycle[0] if o['status'] == 'unassigned'])

        # Reset reward tracking for new episode
        self.total_grid_power_loss_reduction_benefit = 0
        self.total_grid_peak_shaving_benefit = 0
        self.total_solar_station_profit = 0
        self.total_platform_order_benefit = 0
        self.total_platform_charge_benefit = 0
        self.completed_orders_this_episode = 0 

        return self._get_state_all_vehicles()

    def get_environment_metrics(self):
        return {
            'is_peak_hour': cfg.is_peak_hour(self.current_cycle),
            'electricity_price': cfg.electricity_price(self.current_cycle),
            'solar_discount': SOLAR_ELECTRICITY_PRICE_DISCOUNT,
            'avg_battery': np.mean([v.get('battery', 0) for v in self.vehicles]) if self.vehicles else 0.0
        }

    def _nearest_stations(self, vehicle):
        """按到车辆当前位置的距离升序排列充电站，观测和充电动作共用这一顺序。"""
        stations_with_dist = []
        for station in self.stations:
            dist_to_station = haversine(vehicle['position'], station['position'])
            stations_with_dist.append({'station_obj': station, 'dist': dist_to_station})
        stations_with_dist.sort(key=lambda x: x['dist'])
        return stations_with_dist

    def _get_state_for_vehicle(self, vehicle_idx):
        vehicle = self.vehicles[vehicle_idx]
        battery_level = vehicle['battery'] / 100.0
        cycle_norm = self.current_cycle / self.max_cycles if self.max_cycles > 0 else 0.0
        pos_lat_norm = self._normalize_lat(vehicle['position'][0])
        pos_lon_norm = self._normalize_lon(vehicle['position'][1])
        status_list = ['idle', 'to_pickup', 'on_trip', 'to_charge', 'charging']
        status_one_hot = [0.0] * (len(status_list) + 1)
        try:
            if vehicle['status'] == 'idle' and vehicle['battery'] < self.battery_threshold_order: status_one_hot[5] = 1.0
            else: status_one_hot[status_list.index(vehicle['status'])] = 1.0
        except ValueError: status_one_hot[0] = 1.0
        orders_with_dist = []
        for order in [o for o in self.active_orders_pool if o['status'] == 'unassigned']:
            if isinstance(order.get('pickup'), tuple) and len(order['pickup']) == 2:
                dist_to_pickup = haversine(vehicle['position'], order['pickup'])
                orders_with_dist.append({'order_obj':order, 'dist': dist_to_pickup})
        orders_with_dist.sort(key=lambda x: x['dist'])
        order_features = []
        for i in range(self.max_orders_in_state_action):
            if i < len(orders_with_dist):
                order_info = orders_with_dist[i]; order = order_info['order_obj']
                dist_norm = min(1.0, order_info['dist'] / self.max_dist_norm if self.max_dist_norm > 0 else 1.0)
                length_norm = min(1.0, order['length'] / self.max_dist_norm if self.max_dist_norm > 0 else 0.0)
                potential_profit = (order['length'] * NEW_ORDER_ASSIGNMENT_LENGTH_MULTIPLIER - order_info['dist'] * GUROBI_DISTANCE_PENALTY_COEFF)
                normalized_profit = np.clip(potential_profit / 80.0, -1.0, 1.0)
                order_features.extend([dist_norm, length_norm, normalized_profit])
            else: order_features.extend([1.0, 0.0, -1.0])
        stations_with_dist = self._nearest_stations(vehicle)
        station_features = []
        for i in range(self.n_closest_stations_in_state):
            if i < len(stations_with_dist):
                station_info = stations_with_dist[i]; station = station_info['station_obj']
                dist_norm = min(1.0, station_info['dist'] / self.max_dist_norm if self.max_dist_norm > 0 else 1.0)
                is_full = 1.0 if len(station['charging_vehicle_ids']) >= self.max_vehicles_per_station else 0.0
                potential_charge_profit = (GUROBI_CHARGE_COEFF * max(0, 100 - vehicle['battery'])) - (station_info['dist'] * GUROBI_DISTANCE_PENALTY_COEFF)
                normalized_charge_profit = np.clip(potential_charge_profit / 80.0, -1.0, 1.0)
                station_features.extend([dist_norm, is_full, normalized_charge_profit])
            else: station_features.extend([1.0, 1.0, -1.0])
        state_vec = [battery_level, cycle_norm] + status_one_hot + [pos_lat_norm, pos_lon_norm] + order_features + station_features
        if len(state_vec) != self.state_dim:
            if len(state_vec) > self.state_dim: state_vec = state_vec[:self.state_dim]
            else: state_vec.extend([0.0] * (self.state_dim - len(state_vec)))
        return np.array(state_vec, dtype=np.float32)

    def _get_state_all_vehicles(self):
        all_states = np.zeros((self.n_vehicles, self.state_dim), dtype=np.float32)
        for i in range(self.n_vehicles):
            all_states[i] = self._get_state_for_vehicle(i)
        return all_states

    def _update_vehicle_position(self, vehicle, target_pos, travel_distance_this_cycle):
        if vehicle['position'] == target_pos: return True
        dist_to_target = haversine(vehicle['position'], target_pos)
        if dist_to_target < 1e-6: vehicle['position'] = target_pos; return True 
        if dist_to_target <= travel_distance_this_cycle: vehicle['position'] = target_pos; return True 
        else:
            fraction = travel_distance_this_cycle / dist_to_target
            current_lat, current_lon = vehicle['position']; target_lat, target_lon = target_pos
            new_lat = current_lat + (target_lat - current_lat) * fraction
            new_lon = current_lon + (target_lon - current_lon) * fraction
            new_lat = max(self.min_lat, min(self.max_lat, new_lat)) if abs(self.max_lat - self.min_lat) > 1e-6 else self.min_lat
            new_lon = max(self.min_lon, min(self.max_lon, new_lon)) if abs(self.max_lon - self.min_lon) > 1e-6 else self.min_lon
            vehicle['position'] = (new_lat, new_lon); return False

    def step(self, actions):
        current_rewards = np.zeros(self.n_vehicles, dtype=np.float32)
        travel_this_cycle = self.travel_speed

        is_peak_hour = cfg.is_peak_hour(self.current_cycle)
        grid_price = cfg.electricity_price(self.current_cycle)
        
        # 定义本周期内的累加器，这些累加器应在每个step的开始重置
        cycle_grid_power_loss_reduction_benefit_local = 0
        cycle_solar_station_profit_local_from_sales = 0 # 仅用于累积售电收入
        cycle_solar_station_profit_local_from_generation = 0 # 仅用于累加发电补贴
        total_charging_kwh_this_cycle_peak_hour_local = 0 # 用于计算削峰填谷

        # 光伏发电量生成与储能更新。单站装机 44 kW，储能上限 135 kWh
        for station in self.stations:
            if station['is_solar']:
                station['solar_power_output'] = cfg.psc_pv_energy_kwh(self.current_cycle)
                station['stored_energy'] = min(
                    cfg.STORAGE_CAPACITY_KWH,
                    station['stored_energy'] + station['solar_power_output'],
                )
                # 光伏场站补贴：在发电时就计入利润（这部分是全局且独立于售电的）
                cycle_solar_station_profit_local_from_generation += SOLAR_FARM_SUBSIDY * station['solar_power_output']

        # 初始化每辆车的事件标记，供外部 reward 使用（如果有）
        for v in self.vehicles:
            v.setdefault('events', {})
            v['events'].update({'finished_order': False, 'picked_up_order': False, 'started_charging': False, 'invalid_action': False, 'is_charging_solar': False, 'order_length': 0, 'charge_amount': 0})

        # 第一遍循环：处理车辆的动作决策和由此导致的平台收益，但不更新环境物理状态
        for i, action_idx in enumerate(actions):
            vehicle = self.vehicles[i]; can_take_new_action = False
            if vehicle['status'] == 'idle': can_take_new_action = True
            elif vehicle['status'] == 'charging' and vehicle['battery'] >= self.battery_target_charge:
                if vehicle['target_station_id'] is not None:
                    try:
                        station = self.stations[vehicle['target_station_id']]
                        if vehicle['id'] in station['charging_vehicle_ids']: station['charging_vehicle_ids'].remove(vehicle['id'])
                    except (IndexError, TypeError): pass
                vehicle['status'] = 'idle'; vehicle['target_station_id'] = None; vehicle['task_destination_pos'] = None; vehicle['task_remaining_cycles'] = 0; can_take_new_action = True
            if not can_take_new_action: continue
            vehicle_at_decision = vehicle.copy(); action_idx = int(action_idx)
            
            if action_idx == 0:
                vehicle['status'] = 'idle'
                current_rewards[i] -= 20.0
            elif 1 <= action_idx <= self.max_orders_in_state_action:
                order_choice_local_idx = action_idx - 1
                unassigned_orders_pool = [o for o in self.active_orders_pool if o['status'] == 'unassigned']; orders_with_dist_now = []
                for o_data in unassigned_orders_pool:
                    if isinstance(o_data.get('pickup'), tuple) and len(o_data['pickup']) == 2:
                        dist = haversine(vehicle_at_decision['position'], o_data['pickup']); orders_with_dist_now.append({'order_obj': o_data, 'dist': dist})
                orders_with_dist_now.sort(key=lambda x: x['dist'])
                if order_choice_local_idx < len(orders_with_dist_now):
                    chosen_order_info = orders_with_dist_now[order_choice_local_idx]
                    actual_order_obj = next((o for o in self.all_orders_master_list if o['id'] == chosen_order_info['order_obj']['id']), None)
                    if actual_order_obj and actual_order_obj['status'] == 'unassigned':
                        if vehicle_at_decision['battery'] >= self.battery_threshold_order:
                            dist_to_pickup = chosen_order_info['dist']
                            order_benefit = (actual_order_obj['length'] * NEW_ORDER_ASSIGNMENT_LENGTH_MULTIPLIER) - (dist_to_pickup * GUROBI_DISTANCE_PENALTY_COEFF)
                            current_rewards[i] += order_benefit
                            self.total_platform_order_benefit += order_benefit
                            actual_order_obj['status'] = 'assigned'; actual_order_obj['assigned_vehicle_id'] = vehicle['id']
                            vehicle['status'] = 'to_pickup'; vehicle['current_order_id'] = actual_order_obj['id']; vehicle['task_destination_pos'] = actual_order_obj['pickup']
                            pickup_cycles = max(1, math.ceil(dist_to_pickup / self.travel_speed)); vehicle['task_remaining_cycles'] = pickup_cycles; actual_order_obj['pickup_travel_remaining_cycles'] = pickup_cycles
                            # 标记事件：已接单
                            try:
                                vehicle['events']['picked_up_order'] = True
                                vehicle['events']['picked_up'] = True
                                vehicle['events']['order_length'] = actual_order_obj.get('length', 0)
                            except Exception:
                                pass
                            if chosen_order_info['order_obj'] in self.active_orders_pool: self.active_orders_pool.remove(chosen_order_info['order_obj'])
                        else: current_rewards[i] -= 50.0
                    else: current_rewards[i] -= 50.0
                else: current_rewards[i] -= 50.0
            elif self.max_orders_in_state_action < action_idx < self.action_dim:
                station_choice_idx = action_idx - (self.max_orders_in_state_action + 1)
                nearest_stations = self._nearest_stations(vehicle_at_decision)
                if 0 <= station_choice_idx < self.n_closest_stations_in_state and station_choice_idx < len(nearest_stations):
                    station_info = nearest_stations[station_choice_idx]
                    station_to_charge_at = station_info['station_obj']
                    if len(station_to_charge_at['charging_vehicle_ids']) < self.max_vehicles_per_station:
                        dist_to_station = station_info['dist']
                        
                        # --- 预计充电电量使用固定单周期充电量 ---
                        potential_charge_amount_kwh = self.battery_charge_per_cycle / 100.0 * cfg.BATTERY_CAPACITY_KWH
                        potential_charge_amount_kwh = max(0, potential_charge_amount_kwh) # 确保不为负
              
                        # 光伏站使用折扣电价，无单独补贴项
                        electricity_price = grid_price
                        if station_to_charge_at['is_solar']:
                            electricity_price *= SOLAR_ELECTRICITY_PRICE_DISCOUNT
                        charging_cost = electricity_price * potential_charge_amount_kwh
                        travel_cost_to_station = dist_to_station * GUROBI_DISTANCE_PENALTY_COEFF
                        charge_benefit = -charging_cost - travel_cost_to_station
                        
                        current_rewards[i] += charge_benefit
                        self.total_platform_charge_benefit += charge_benefit

                        vehicle['status'] = 'to_charge'; vehicle['target_station_id'] = station_to_charge_at['id']; vehicle['task_destination_pos'] = station_to_charge_at['position']
                        charge_travel_cycles = max(1, math.ceil(dist_to_station / self.travel_speed)); vehicle['task_remaining_cycles'] = charge_travel_cycles
                    else: current_rewards[i] -= 50
                else: current_rewards[i] -= 50.0
            else: current_rewards[i] -= 50.0

        # 第二遍循环：处理所有车辆的物理移动和充电状态更新
        # 并在该循环中累积 Grid Power Loss Reduction 和 Peak Shaving 的局部量
        for i in range(self.n_vehicles):
            vehicle = self.vehicles[i]; is_moving = False; original_status = vehicle['status']
            
            if vehicle['status'] == 'to_pickup':
                is_moving = True; order = next((o for o in self.all_orders_master_list if o['id'] == vehicle['current_order_id']), None)
                if order and vehicle['task_remaining_cycles'] > 0:
                    arrived = self._update_vehicle_position(vehicle, order['pickup'], travel_this_cycle); vehicle['task_remaining_cycles'] -= 1
                    if order.get('pickup_travel_remaining_cycles', 0) > 0: order['pickup_travel_remaining_cycles'] -=1
                    if vehicle['task_remaining_cycles'] <= 0 or haversine(vehicle['position'], order['pickup']) < self.travel_speed * 0.1: 
                        vehicle['position'] = order['pickup']; vehicle['status'] = 'on_trip'; vehicle['task_destination_pos'] = order['destination']; vehicle['task_remaining_cycles'] = order['duration_cycles']; order['trip_remaining_cycles'] = order['duration_cycles']
                elif order is None: vehicle['status'] = 'idle'; vehicle['current_order_id'] = None; vehicle['task_remaining_cycles'] = 0; current_rewards[i] -= 5.0
                elif vehicle['task_remaining_cycles'] <= 0:
                    if order: vehicle['position'] = order['pickup']; vehicle['status'] = 'on_trip'; vehicle['task_destination_pos'] = order['destination']; vehicle['task_remaining_cycles'] = order['duration_cycles']; order['trip_remaining_cycles'] = order['duration_cycles']
                    else: vehicle['status'] = 'idle'; vehicle['current_order_id'] = None; vehicle['task_remaining_cycles'] = 0; current_rewards[i] -= 5.0
            elif vehicle['status'] == 'on_trip':
                is_moving = True; order = next((o for o in self.all_orders_master_list if o['id'] == vehicle['current_order_id']), None)
                if order and vehicle['task_remaining_cycles'] > 0:
                    arrived = self._update_vehicle_position(vehicle, order['destination'], travel_this_cycle); vehicle['task_remaining_cycles'] -= 1
                    if order.get('trip_remaining_cycles', 0) > 0: order['trip_remaining_cycles'] -=1
                    if vehicle['task_remaining_cycles'] <= 0 or haversine(vehicle['position'], order['destination']) < self.travel_speed * 0.1: 
                        vehicle['position'] = order['destination']; vehicle['status'] = 'idle';
                        # 标记订单完成事件，供外部 reward 使用
                        try:
                            vehicle['events']['finished_order'] = True
                            vehicle['events']['order_length'] = order.get('length', 0)
                        except Exception:
                            pass
                        current_rewards[i] += order['length'] * ORDER_COMPLETION_REWARD_COEFF 
                        order['status'] = 'completed'; vehicle['current_order_id'] = None; vehicle['task_destination_pos'] = None
                        self.completed_orders_this_episode += 1 
                elif order is None: vehicle['status'] = 'idle'; vehicle['current_order_id'] = None; vehicle['task_remaining_cycles'] = 0; current_rewards[i] -= 5.0
                elif vehicle['task_remaining_cycles'] <= 0 and order and order['status'] != 'completed':
                    vehicle['position'] = order['destination']; vehicle['status'] = 'idle';
                    current_rewards[i] += order['length'] * ORDER_COMPLETION_REWARD_COEFF 
                    order['status'] = 'completed'; vehicle['current_order_id'] = None; vehicle['task_destination_pos'] = None
                    self.completed_orders_this_episode += 1 
            elif vehicle['status'] == 'to_charge':
                is_moving = True; station = None
                if vehicle['target_station_id'] is not None:
                    try: station = self.stations[vehicle['target_station_id']]
                    except (IndexError, TypeError): vehicle['status'] = 'idle'; vehicle['target_station_id'] = None; vehicle['task_remaining_cycles'] = 0; current_rewards[i] -= 1.0; station = None
                if station and vehicle['task_remaining_cycles'] > 0:
                    arrived = self._update_vehicle_position(vehicle, station['position'], travel_this_cycle); vehicle['task_remaining_cycles'] -= 1
                    if vehicle['task_remaining_cycles'] <= 0 or haversine(vehicle['position'], station['position']) < self.travel_speed * 0.1: 
                        vehicle['position'] = station['position']
                        if len(station['charging_vehicle_ids']) < self.max_vehicles_per_station: 
                            vehicle['status'] = 'charging'; station['charging_vehicle_ids'].append(vehicle['id']); vehicle['task_destination_pos'] = None
                            vehicle['charging_cycles_remaining'] = MIN_CHARGE_CYCLES  # 设置最少充电周期
                        else: vehicle['status'] = 'idle'; current_rewards[i] -= 0.5; vehicle['target_station_id'] = None; vehicle['task_destination_pos'] = None
                elif station is None and vehicle['target_station_id'] is not None: pass
                elif vehicle['task_remaining_cycles'] <= 0 and station:
                    vehicle['position'] = station['position']
                    if len(station['charging_vehicle_ids']) < self.max_vehicles_per_station: 
                        vehicle['status'] = 'charging'; station['charging_vehicle_ids'].append(vehicle['id']); vehicle['task_destination_pos'] = None
                        vehicle['charging_cycles_remaining'] = MIN_CHARGE_CYCLES  # 设置最少充电周期
                    else: vehicle['status'] = 'idle'; current_rewards[i] -= 0.5; vehicle['target_station_id'] = None; vehicle['task_destination_pos'] = None
            elif vehicle['status'] == 'charging':
                is_moving = False;
                charge_amount = self.battery_charge_per_cycle
                actual_charge_kwh = charge_amount / 100.0 * cfg.BATTERY_CAPACITY_KWH
                # 标记充电开始与充电量，供外部 reward 使用
                try:
                    vehicle['events']['started_charging'] = True
                    vehicle['events']['charge_amount'] = actual_charge_kwh
                except Exception:
                    pass
                station = self.stations[vehicle['target_station_id']]
                
                # 更新剩余强制充电周期
                if vehicle.get('charging_cycles_remaining', 0) > 0:
                    vehicle['charging_cycles_remaining'] -= 1
                
                # For Peak Shaving Benefit: Any charging at a station contributes if it's peak hour
                if is_peak_hour:
                    # 削峰填谷效益直接加到当前车辆的奖励中，使其即时感知
                    current_rewards[i] += PEAK_TO_BASE_COST_DIFFERENCE * actual_charge_kwh
                    total_charging_kwh_this_cycle_peak_hour_local += actual_charge_kwh # 累加到本周期的局部变量 (用于全局统计)

                # --- 光伏站充电优先级和收益计算 ---
                total_power_from_solar_kwh = 0 # 追踪所有来自光伏的电量 (储能 + 直冲)
                
                electricity_price_for_solar_station = grid_price * SOLAR_ELECTRICITY_PRICE_DISCOUNT
                if station['is_solar']:
                    charged_from_storage = 0
                    charged_from_direct_solar = 0
                    remaining_charge_needed_kwh = actual_charge_kwh

                    # 优先从储能取电
                    if station['stored_energy'] > 0:
                        charged_from_storage = min(remaining_charge_needed_kwh, station['stored_energy'])
                        station['stored_energy'] -= charged_from_storage
                        remaining_charge_needed_kwh -= charged_from_storage
                        total_power_from_solar_kwh += charged_from_storage
                        cycle_solar_station_profit_local_from_sales += (electricity_price_for_solar_station * charged_from_storage)

                    # 剩余需求由光伏直供
                    if remaining_charge_needed_kwh > 0:
                        charged_from_direct_solar = remaining_charge_needed_kwh
                        total_power_from_solar_kwh += charged_from_direct_solar
                        cycle_solar_station_profit_local_from_sales += (electricity_price_for_solar_station * charged_from_direct_solar)
                    current_rewards[i] += C_LOSS * total_power_from_solar_kwh
                    cycle_grid_power_loss_reduction_benefit_local += C_LOSS * total_power_from_solar_kwh
                
                vehicle['battery'] = min(100, vehicle['battery'] + charge_amount)
                
                # 检查是否可以离开充电站：已充满或完成最少充电周期
                min_cycles_done = vehicle.get('charging_cycles_remaining', 0) <= 0
                if vehicle['battery'] >= self.battery_target_charge and min_cycles_done:
                    # 离开充电站
                    if vehicle['id'] in station['charging_vehicle_ids']:
                        station['charging_vehicle_ids'].remove(vehicle['id'])
                    vehicle['status'] = 'idle'
                    vehicle['target_station_id'] = None

            if is_moving:
                vehicle['battery'] = max(0, vehicle['battery'] - self.battery_drain_moving)
                if vehicle['battery'] == 0 and original_status != 'charging':
                    current_rewards[i] -= 50.0
                    if vehicle['current_order_id'] is not None:
                        order = next((o for o in self.all_orders_master_list if o['id'] == vehicle['current_order_id']), None)
                        if order and (order['status'] == 'assigned' or order['status'] == 'pickup'):
                            order['status'] = 'unassigned'; order['assigned_vehicle_id'] = None
                            if order not in self.active_orders_pool: self.active_orders_pool.append(order)
                        vehicle['current_order_id'] = None
                    if vehicle['target_station_id'] is not None: vehicle['target_station_id'] = None
                    vehicle['status'] = 'idle'; vehicle['task_remaining_cycles'] = 0; vehicle['task_destination_pos'] = None
        
        # 将所有局部累加的全局效益，加到环境对象的总追踪器中
        self.total_grid_power_loss_reduction_benefit += cycle_grid_power_loss_reduction_benefit_local
        self.total_grid_peak_shaving_benefit += (PEAK_TO_BASE_COST_DIFFERENCE * total_charging_kwh_this_cycle_peak_hour_local) # 削峰填谷效益的累加
        
        # 累加光伏站的发电补贴和所有售电收入
        self.total_solar_station_profit += cycle_solar_station_profit_local_from_generation + cycle_solar_station_profit_local_from_sales

        # --- 光伏站储能成本 (每4个周期计算一次) ---
        if (self.current_cycle + 1) % STORAGE_COST_CALC_INTERVAL == 0: # +1 是因为 current_cycle 从0开始
            for station in self.stations:
                if station['is_solar'] and station['stored_energy'] > 0: 
                    self.total_solar_station_profit -= STORAGE_COST_PER_KWH * station['stored_energy']

        
        self.current_cycle += 1
        if self.current_cycle < self.max_cycles:
            new_orders_this_cycle = self.orders_by_cycle.get(self.current_cycle, [])
            for o_data in new_orders_this_cycle:
                is_existing_active = any(existing_o['id'] == o_data['id'] for existing_o in self.active_orders_pool)
                master_order = next((mo for mo in self.all_orders_master_list if mo['id'] == o_data['id']), None)
                if not is_existing_active and master_order and master_order['status'] == 'unassigned': self.active_orders_pool.append(master_order)
        valid_active_order_ids = {o['id'] for o in self.all_orders_master_list if o['status'] == 'unassigned'}
        self.active_orders_pool = [o for o in self.active_orders_pool if o['id'] in valid_active_order_ids]
        done = self.current_cycle >= self.max_cycles
        next_state_all_vehicles = self._get_state_all_vehicles()
        
        # 如果存在外部奖励函数，则优先使用外部奖励函数的返回值（覆盖内部 reward）
        if self.external_reward_fn is not None:
            try:
                env_metrics = self.get_environment_metrics()
                new_rewards = np.zeros(self.n_vehicles, dtype=np.float32)
                for idx, v in enumerate(self.vehicles):
                    try:
                        # 为外部奖励函数构造一个安全的 vehicle 视图，确保包含 events 等键，避免 KeyError
                        sanitized_v = v.copy() if isinstance(v, dict) else {'id': idx}
                        sanitized_v.setdefault('events', {})
                        ev = sanitized_v['events']
                        ev.setdefault('finished_order', False)
                        ev.setdefault('picked_up_order', False)
                        ev.setdefault('started_charging', False)
                        ev.setdefault('invalid_action', False)
                        ev.setdefault('is_charging_solar', False)
                        ev.setdefault('order_length', 0)
                        ev.setdefault('charge_amount', 0)

                        # 保护性地调用外部 reward 函数；它通常已由上层包装为安全版本
                        try:
                            r, comps = self.external_reward_fn(sanitized_v, env_metrics, max(0, self.current_cycle - 1), self.max_cycles)
                        except TypeError:
                            # 某些生成的函数可能声明参数更少，尝试降级调用
                            try:
                                r, comps = self.external_reward_fn(sanitized_v, env_metrics, max(0, self.current_cycle - 1))
                            except TypeError:
                                try:
                                    r, comps = self.external_reward_fn(sanitized_v, env_metrics)
                                except TypeError:
                                    r, comps = self.external_reward_fn(sanitized_v)

                        new_rewards[idx] = float(r) if r is not None else 0.0

                        # 可选：将组件贡献累加到环境统计（若组件以约定键名出现）
                        if isinstance(comps, dict):
                            try:
                                if any('grid' in k or 'peak' in k for k in comps.keys()):
                                    self.total_grid_power_loss_reduction_benefit += sum([float(x) for x in comps.values() if isinstance(x, (int, float))])
                            except Exception:
                                pass
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        print(f"外部奖励函数调用失败 (vehicle {idx}): {e}\n{tb}")
                        new_rewards[idx] = 0.0
                current_rewards = new_rewards
            except Exception as e:
                print(f"调用外部奖励函数失败: {e}")

        return next_state_all_vehicles, current_rewards, done

    def step_physics_only(self, actions):
        """
        Compatibility shim for the short-training probe.
        Applies actions and advances the environment one cycle by reusing `step`.
        Returns: done (bool)
        Note: this will also compute rewards internally; callers can ignore returned rewards.
        """
        # Reuse the existing step logic to keep physics consistent.
        next_state, rewards, done = self.step(actions)
        return done
