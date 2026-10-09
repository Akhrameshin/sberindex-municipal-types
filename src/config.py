import random
from pathlib import Path
import numpy as np
import yaml

CFG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "configs/config.yaml"))
SEED = CFG["seed"]

def seed_all(s=None):
    s = SEED if s is None else s
    random.seed(s); np.random.seed(s)
