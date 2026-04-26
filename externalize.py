import yaml
import os
from dotenv import load_dotenv
import xgboost as xgb

# 1. Load Secrets and Config
load_dotenv()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

class MotionSensingAI:
    def __init__(self):
        # 2. Extract settings from config
        self.eta = config['ml_automation']['hyperparameters']['eta']
        self.max_depth = config['ml_automation']['hyperparameters']['max_depth']
        self.api_key = os.getenv("OPENAI_API_KEY")

    def train_model(self, X, y):
        # 3. Use externalized parameters
        params = {
            "eta": self.eta,
            "max_depth": self.max_depth,
            "objective": config['ml_automation']['hyperparameters']['objective']
        }
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(params, dtrain)
        print(f"Model trained with eta={self.eta}")

# This architecture allows hardware engineers to tune the model 
# just by editing the YAML file without touching the Python logic.