import psutil
import os
import time
import sys
import math
from contextlib import contextmanager
import random
import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import f1_score

@contextmanager
def trace(title):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    yield
    m1 = p.memory_info()[0] / 2. ** 30
    delta = m1 - m0
    sign = '+' if delta >= 0 else '-'
    delta = math.fabs(delta)
    print(f"[{m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec] {title} ", file=sys.stderr)

@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(0)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    if torch.cuda.is_available(): 

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

# 最適な閾値を求める関数
def threshold_optimization(y_true, y_pred, metrics=None):
    def f1_opt(x):
        if metrics is not None:
            score = -metrics(y_true, y_pred >= x)
        else:
            raise NotImplementedError
        return score
    result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
    best_threshold = result['x'].item()
    return best_threshold

# 後で定義するモデルは確率で結果を出力するので、そこから最適なf1をgetする関数を定義
def optimized_f1(y_true, y_pred):
    if type(y_true) != np.ndarray:
        y_true = np.array(y_true)
    if type(y_pred) != np.ndarray:
        y_pred = np.array(y_pred)

    bt = threshold_optimization(y_true, y_pred, metrics=f1_score)
    score = f1_score(y_true, y_pred >= bt)
    return score, bt