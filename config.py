"""运行时配置。密钥与模型名仅从环境变量或 .env 读取，不在代码中硬编码。"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import torch

# --- LLM（OpenAI 兼容接口）---
LLM_API_BASE = os.environ.get("LLM4RD_API_BASE", "https://api.openai.com/v1").rstrip("/")
LLM_API_KEY = os.environ.get("LLM4RD_API_KEY", "")
LLM_MODEL = os.environ.get("LLM4RD_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.environ.get("LLM4RD_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.environ.get("LLM4RD_MAX_TOKENS", "4096"))

# --- 进化搜索 ---
EVOLUTION_ITERATIONS = int(os.environ.get("LLM4RD_ITERATIONS", "10"))
CANDIDATES_PER_ITERATION = int(os.environ.get("LLM4RD_CANDIDATES", "8"))
EVALUATION_EPISODES = int(os.environ.get("LLM4RD_EVAL_EPISODES", "10"))
RANDOM_SEED = int(os.environ.get("LLM4RD_SEED", "42"))

# --- 仿真规模 ---
TRAVEL_SPEED = 50 / 12
MAX_CYCLES = 144
N_VEHICLES = int(os.environ.get("LLM4RD_N_VEHICLES", "1000"))
N_STATIONS = int(os.environ.get("LLM4RD_N_STATIONS", "1000"))
BATTERY_THRESHOLD_FOR_ORDER = 20
BATTERY_CHARGE_PER_CYCLE = 10
BATTERY_DRAIN_MOVING = 2
BATTERY_TARGET_CHARGE_LEVEL = 80
MAX_VEHICLES_PER_STATION = 10
MIN_CHARGE_CYCLES = 2

# --- 奖励参考系数 ---
GUROBI_CHARGE_COEFF = 0.15
NEW_ORDER_ASSIGNMENT_LENGTH_MULTIPLIER = 2.5
GUROBI_DISTANCE_PENALTY_COEFF = 0.12
ORDER_COMPLETION_REWARD_COEFF = 25.0
SOLAR_STATION_RATIO = 0.2
CURRENT_ELECTRICITY_PRICE_PER_KWH = 0.4
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
CLIP_PARAM = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
VF_COEFF = 0.5
ENT_COEFF = 0.01
MAX_GRAD_NORM = 0.5
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
