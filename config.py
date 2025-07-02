# config.py
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = "gpt-3.5-turbo"  # Cost-effective model
MAX_TOKENS = 1500
TEMPERATURE = 0.7

# Dataset Configuration
MEDQA_PATH = "./data/medqa/"
NUM_SAMPLES = 50  # Start small for testing
RANDOM_SEED = 42

# Agent Configuration
MAX_ROUNDS = 5
MAX_TURNS = 5
NUM_AGENTS = {
    "low": 1,
    "moderate": 5,
    "high": 9  # 3 teams x 3 agents
}

# Logging Configuration
LOG_DIR = f"./logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(LOG_DIR, exist_ok=True)

# Weights & Biases Configuration
WANDB_PROJECT = "mdagents-medical"
WANDB_ENTITY = None  # Set your wandb username here if needed
