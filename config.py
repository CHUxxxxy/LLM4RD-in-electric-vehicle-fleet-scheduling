"""运行时配置。密钥与模型名仅从环境变量或 .env 读取，不在代码中硬编码。"""
import math
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import torch

# --- LLM（OpenAI 兼容接口）---
# 论文附录：Claude Sonnet 4.6，temperature 0.7，最大输出 4096
LLM_API_BASE = os.environ.get("LLM4RD_API_BASE", "https://api.openai.com/v1").rstrip("/")
LLM_API_KEY = os.environ.get("LLM4RD_API_KEY", "")
LLM_MODEL = os.environ.get("LLM4RD_MODEL", "claude-sonnet-4-6")
LLM_TEMPERATURE = float(os.environ.get("LLM4RD_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.environ.get("LLM4RD_MAX_TOKENS", "4096"))

# --- 进化搜索 ---
EVOLUTION_ITERATIONS = int(os.environ.get("LLM4RD_ITERATIONS", "10"))
CANDIDATES_PER_ITERATION = int(os.environ.get("LLM4RD_CANDIDATES", "8"))
EVALUATION_EPISODES = int(os.environ.get("LLM4RD_EVAL_EPISODES", "10"))
RANDOM_SEED = int(os.environ.get("LLM4RD_SEED", "42"))

# --- 仿真规模（与论文第 IV 节及附录一致）---
# 每步 5 分钟，一天 24 小时，共 288 步
CYCLE_MINUTES = 5
CYCLES_PER_HOUR = 60 // CYCLE_MINUTES
HORIZON_HOURS = 24
MAX_CYCLES = HORIZON_HOURS * CYCLES_PER_HOUR
HOURS_PER_CYCLE = CYCLE_MINUTES / 60.0
# 车速 50 km/h，换算为每个 5 分钟步长的行驶距离（km）
TRAVEL_SPEED = 50 / CYCLES_PER_HOUR
N_VEHICLES = int(os.environ.get("LLM4RD_N_VEHICLES", "1200"))
N_STATIONS = int(os.environ.get("LLM4RD_N_STATIONS", "220"))
# 其中 12 座为光储充站
N_PSC_STATIONS = int(os.environ.get("LLM4RD_N_PSC_STATIONS", "12"))
PV_CAPACITY_KW = 44.0
STORAGE_CAPACITY_KWH = 135.0
# 续航 500 km、电耗 5 km/kWh，对应电池容量 100 kWh
DRIVING_RANGE_KM = 500.0
KM_PER_KWH = 5.0
BATTERY_CAPACITY_KWH = DRIVING_RANGE_KM / KM_PER_KWH
# 150 kW 快充，充满 100 kWh 需要 40 分钟，即 8 个决策步
CHARGER_POWER_KW = 150.0
FULL_CHARGE_MINUTES = 40.0
BATTERY_THRESHOLD_FOR_ORDER = 20
BATTERY_CHARGE_PER_CYCLE = 100.0 * (CHARGER_POWER_KW * HOURS_PER_CYCLE) / BATTERY_CAPACITY_KWH
BATTERY_DRAIN_MOVING = 100.0 * (TRAVEL_SPEED / KM_PER_KWH) / BATTERY_CAPACITY_KWH
BATTERY_TARGET_CHARGE_LEVEL = 80
MAX_VEHICLES_PER_STATION = 10
MIN_CHARGE_CYCLES = 2

# 分时电价（元/kWh）：峰 10:00-12:00、15:00-21:00；平 7:00-10:00、12:00-15:00、21:00-23:00；谷 23:00-7:00
TOU_PEAK_PRICE = 0.7536
TOU_OFFPEAK_PRICE = 0.5464
TOU_VALLEY_PRICE = 0.2602


def hour_of_cycle(cycle):
    """当前决策步对应的小时，范围 [0, 24)。"""
    return (float(cycle) / CYCLES_PER_HOUR) % HORIZON_HOURS


def is_peak_hour(cycle):
    """是否处于论文中的峰时段。"""
    hour = hour_of_cycle(cycle)
    return (10.0 <= hour < 12.0) or (15.0 <= hour < 21.0)


def electricity_price(cycle):
    """按论文分时电价返回当前电价（元/kWh）。"""
    hour = hour_of_cycle(cycle)
    if hour >= 23.0 or hour < 7.0:
        return TOU_VALLEY_PRICE
    if is_peak_hour(cycle):
        return TOU_PEAK_PRICE
    return TOU_OFFPEAK_PRICE


def psc_pv_energy_kwh(cycle):
    """单座光储充站在当前 5 分钟步内的光伏发电量（kWh）。

    装机 44 kW。6:00-18:00 用余弦日变化，正午达到额定功率，夜间为 0。
    """
    hour = hour_of_cycle(cycle)
    if hour < 6.0 or hour > 18.0:
        return 0.0
    daylight_factor = max(0.0, math.cos((hour - 12.0) / 12.0 * math.pi))
    return PV_CAPACITY_KW * daylight_factor * HOURS_PER_CYCLE

# --- 奖励参考系数 ---
GUROBI_CHARGE_COEFF = 0.15
NEW_ORDER_ASSIGNMENT_LENGTH_MULTIPLIER = 2.5
GUROBI_DISTANCE_PENALTY_COEFF = 0.12
ORDER_COMPLETION_REWARD_COEFF = 25.0
SOLAR_STATION_RATIO = N_PSC_STATIONS / float(N_STATIONS)
# 保留名称供奖励代码读取；具体电价按决策步调用 electricity_price()
CURRENT_ELECTRICITY_PRICE_PER_KWH = TOU_OFFPEAK_PRICE
SOLAR_ELECTRICITY_PRICE_DISCOUNT = 0.8
C_LOSS = 1.31
PEAK_TO_BASE_COST_DIFFERENCE = 2.6
SOLAR_POWER_GENERATION_MAX_PER_HOUR = 200
SOLAR_FARM_SUBSIDY = 0.25
STORAGE_COST_PER_KWH = 0.05
STORAGE_COST_CALC_INTERVAL = 4

# --- MAPPO ---
_QUICK = os.environ.get("LLM4RD_QUICK", "0") == "1"
MAPPO_TRAINING_STEPS = float(os.environ.get(
    "LLM4RD_TRAINING_STEPS",
    "50000" if _QUICK else "3900000",
))
PPO_STEPS = 256 if not _QUICK else 64
PPO_EPOCHS = 4 if not _QUICK else 2
MINI_BATCH_SIZE = 4096 if not _QUICK else 1024
CLIP_PARAM = 0.15
GAMMA = 0.99
GAE_LAMBDA = 0.95
VF_COEFF = 0.5
ENT_COEFF = 0.01
MAX_GRAD_NORM = 0.45
HIDDEN_DIM = 256
LR = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PACKAGE_ROOT, "outputs")
FAILED_DIR = os.path.join(OUTPUT_DIR, "failed_candidates")
BEST_REWARD_PATH = os.path.join(OUTPUT_DIR, "best_reward.py")

for _d in (OUTPUT_DIR, FAILED_DIR):
    os.makedirs(_d, exist_ok=True)


def require_llm_credentials():
    """启动前检查 API 配置，避免把密钥写进代码。"""
    if not LLM_API_KEY.strip():
        raise RuntimeError(
            "未设置 LLM4RD_API_KEY。请复制 .env.example 为 .env 并填入密钥，"
            "或通过环境变量 / 命令行参数提供。"
        )
    if not LLM_MODEL.strip():
        raise RuntimeError("未设置 LLM4RD_MODEL。")
