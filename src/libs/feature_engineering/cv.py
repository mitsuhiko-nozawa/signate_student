import os.path as osp
import numpy as np
import pandas as pd
from .base import Feature
import datetime
from sklearn.model_selection import KFold, StratifiedKFold


class stratified_cv(Feature):
    def create_features(self):
        train_df, test_df = self.read_input()
        use_feats = []
        for seed in self.seeds:
            kf = StratifiedKFold(n_splits=self.nfolds, random_state=seed, shuffle=True)
            feat_name = f"{self.name}_{seed}"
            use_feats.append(feat_name)
            train_df[feat_name] = -1
            for fold, (tr_ind, val_ind) in enumerate(kf.split(train_df, train_df["state"])):
                train_df.loc[val_ind, feat_name] = fold

        return train_df[use_feats], None