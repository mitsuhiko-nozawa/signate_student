import pandas as pd
import os
import os.path as osp

from Models.models import *
from Models.Tabnet import *
from Models.NN import *

class Predicting():
    def __init__(self, param):
        self.param = param
        self.ROOT = param["ROOT"]
        self.WORK_DIR = param["WORK_DIR"]
        self.pred_path = osp.join(self.WORK_DIR, "preds")
        self.weight_path = osp.join(self.WORK_DIR, "weight")

        self.cv = param["cv"]
        self.seeds = param["seeds"]
        self.nfolds = param["nfolds"]
        self.y = param["y"]
        self.flag = param["pred_flag"]

        self.model = param["model"]
        self.model_param = param["model_param"] 

        if "preds" not in os.listdir(self.WORK_DIR): os.mkdir(self.pred_path)

        

    def __call__(self):
        print("Predict")
        if self.flag:
            test_X = pd.read_csv(osp.join(self.WORK_DIR, "test", "test_X.csv"))
            preds = []
            for seed in self.seeds:
                for fold in range(self.nfolds):
                    self.model_param["random_state"] = seed
                    self.model_param["fold"] = fold
                    self.model_param["WORK_DIR"] = self.WORK_DIR
                    model = eval(self.model)(self.model_param)
                    weight_fname = osp.join(self.weight_path, f"{seed}_{fold}.pkl")
                    model.read_weight(weight_fname)
                    pred = model.predict(test_X)
                    pred_ = pd.DataFrame(pred, columns=["pred"])
                    pred_.to_csv(osp.join(self.pred_path, f"pred_{seed}_{fold}.csv"), index=False)
                    preds.append(pred)
            preds = np.mean(np.array(preds), axis=0)
            preds = pd.DataFrame(preds, columns=["pred"])
            preds.to_csv(osp.join(self.pred_path, "pred.csv"), index=False)


    