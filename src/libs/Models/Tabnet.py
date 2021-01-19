from .base import BaseModel
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_tabnet.tab_model import TabNetRegressor
from torch.nn import L1Loss, MSELoss

import pickle
import numpy as np

class Tabnet_Model(BaseModel):
    def get_model(self):
        return None

    def fit(self, train_X, train_y, valid_X, valid_y):
        cat_feats = self.model_param["cat_feats"]
        categorical_features_indices = [train_X.columns.to_list().index(feat) for feat in cat_feats]
        continuous_features_indices = [i for i in range(train_X.shape[1]) if i not in categorical_features_indices]
        self.model_param["params"]["cat_idxs"] = categorical_features_indices#カテゴリ変数
        self.model_param["params"]["cat_dims"] = list(train_X.iloc[:,categorical_features_indices].nunique() + 1000)
        self.model_param["params"]["cat_emb_dim"] = 20
        self.model_param["params"]["optimizer_fn"] = Adam
        self.model_param["params"]["optimizer_params"] = dict(lr = 2e-2, weight_decay = 1e-3)
        self.model_param["params"]["scheduler_fn"] = ReduceLROnPlateau
        self.model_param["params"]["scheduler_params"] = dict(mode = "min", patience = 3, min_lr = 1e-5, factor = 0.9)
        train_X.fillna(-999, inplace=True)
        valid_X.fillna(-999, inplace=True)
        train_X = train_X.astype(float)
        valid_X = valid_X.astype(float)
        device = torch.device("cuda:7") if torch.cuda.is_available() else torch.device("cpu")
        self.model = TabNetRegressor(**self.model_param["params"])
        self.model.device = device
        self.model.fit(
            X_train=train_X.values,
            y_train=train_y.values,
            eval_set=[(valid_X.values, valid_y.values)],
            eval_name = ["val"],
            eval_metric = ["mae"],
            max_epochs=3,
            patience=20, 
            batch_size=512, 
            virtual_batch_size=32,
            num_workers=1, 
            drop_last=False,
            loss_fn=L1Loss(),
        )

    def predict(self, X):
        X = X.fillna(-999).astype(float)
        preds = self.model.predict(X.values)
        return np.where(preds < 0, 0, preds)
    
    def save_weight(self, path):
        path = path.replace(".pkl", "")
        self.model.save_model(path)

    def read_weight(self, fname):
        self.model = TabNetRegressor()
        fname = fname.replace(".pkl", ".zip")
        self.model.load_model(fname)