import pandas as pd
import os
import os.path as osp

from Models.models import *
from Models.Tabnet import Tabnet_Model
from sklearn.metrics import mean_absolute_error
from utils import seed_everything

class Learning():
    def __init__(self, param):
        self.param = param
        self.ROOT = param["ROOT"]
        self.WORK_DIR = param["WORK_DIR"]
        self.val_pred_path = osp.join(self.WORK_DIR, "val_preds")
        self.weight_path = osp.join(self.WORK_DIR, "weight")

        self.flag = param["train_flag"]
        self.cv = param["cv"]
        self.seeds = param["seeds"]
        self.nfolds = param["nfolds"]
        self.y = param["y"]

        self.model = param["model"]
        self.model_param = param["model_param"] 

        if "val_preds" not in os.listdir(self.WORK_DIR): os.mkdir(self.val_pred_path)
        if "weight" not in os.listdir(self.WORK_DIR): os.mkdir(self.weight_path)

    def __call__(self):
        print("Training")
        print(os.path.join(os.path.abspath("../../../"), "test", "test.csv"))
        for seed in self.seeds:
            seed_everything(seed)
            self.train_by_seed(seed)
        
    
    def train_by_seed(self, seed):
        # calc cv score by seed

        if self.flag:
            for fold in range(self.nfolds):
                self.train_by_fold(seed, fold)


    def train_by_fold(self, seed, fold):
        # create oof_preds_seed_fold.csv
        # train and save model weight
        train_X = pd.read_csv(osp.join(self.WORK_DIR, "train", f"train_X_{seed}_{fold}.csv"))
        train_y = pd.read_csv(osp.join(self.WORK_DIR, "train", f"train_y_{seed}_{fold}.csv"))
        valid_X = pd.read_csv(osp.join(self.WORK_DIR, "valid", f"valid_X_{seed}_{fold}.csv"))
        valid_y = pd.read_csv(osp.join(self.WORK_DIR, "valid", f"valid_y_{seed}_{fold}.csv"))

        self.model_param["random_state"] = seed
        model = eval(self.model)(self.model_param)
        model.fit(train_X, train_y, valid_X, valid_y)

        val_pred = pd.DataFrame(model.predict(valid_X), columns=["pred"])
        val_pred.to_csv(osp.join(self.val_pred_path, f"preds_{seed}_{fold}.csv"), index=False)

        model.save_weight(osp.join(self.weight_path, f"{seed}_{fold}.pkl"))



