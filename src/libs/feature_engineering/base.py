import re
from abc import ABCMeta, abstractmethod
from pathlib import Path

import pandas as pd
import sys, os
sys.path.append("../")
import os.path as osp
from utils import trace, timer


class Feature(metaclass=ABCMeta):
    def __init__(self, param):
        self.name = self.__class__.__name__
        self.ROOT = param["ROOT"] # */src
        self.input_path = osp.join(self.ROOT, param["input_dir"]) # */src/input
        self.out_path = osp.join(self.ROOT, param["output_dir"]) # */src/my_features
        self.out_train_path = osp.join(self.out_path, "train") # */src/my_features/train
        self.out_test_path = osp.join(self.out_path, "test") # */src/my_features/test
        if "train" not in os.listdir(self.out_path): os.mkdir(self.out_train_path)
        if "test" not in os.listdir(self.out_path): os.mkdir(self.out_test_path)
        self.fname = self.name + ".feather"
        self.seeds = param["seeds"]
        self.nfolds = param["nfolds"]

    def run(self):
        # create and save feature as feather
        # featuresは一つの特徴量とは限らず、複数に渡っているかもね(feat_1, feat_2, ...)
        # もしmy_featuresに欲しいものがあったら飛ばす
        if self.fname not in os.listdir(self.out_train_path) and self.fname not in os.listdir(self.out_test_path):
            print(f"create feature {self.name}")
            with trace(self.name):
                train_feats, test_feats = self.create_features()
                self.save(train_feats, "train")
                if test_feats is not None:
                    self.save(test_feats, "test")
            return self
    
    @abstractmethod
    def create_features(self, mode):
        # IOからサブクラスに託す
        raise NotImplementedError

    def save(self, features, mode=None):
        if mode == "train":
            save_fname = osp.join(self.out_train_path, self.fname)
        elif mode == "test":
            save_fname = osp.join(self.out_test_path, self.fname)
        else:
            save_fname = osp.join(self.out_path, self.fname)
        features.to_feather(save_fname)


    def create_default_features(self):
        train_feat = pd.read_csv(osp.join(self.input_path, "train.csv"))
        test_feat = pd.read_csv(osp.join(self.input_path, "test.csv"))
        return train_feat, test_feat



    def read_input(self):
        train_df = pd.read_csv(osp.join(self.input_path, "train.csv"))
        test_df = pd.read_csv(osp.join(self.input_path, "test.csv"))
        return train_df, test_df


    def read_feats(self, feats):
        df = [pd.read_feather( osp.join(self.out_path, f"{feat}.feather") ) for feat in feats]
        df = pd.concat(df, axis=1)
        return df



"""
基底クラスfeature
インターフェイス
train, inferで別の処理
def create_feature
    特徴量をpickle化してmy_featureに保存
def fit
    保存されたpickle特徴量を読み出してfit,
def transform
    fitされた情報を使って、引数のtestデータから求める特徴に変換
"""