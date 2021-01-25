from feature_engineering.features import *
from feature_engineering.cv import *
import os
import os.path as osp
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

class Preprocessing():
    def __init__(self, param):
        # 先にcv/date/にわけたインデックスを保存しておく
        # 普通に特徴作る
        # train valid 分けて作るものは分けて作る
        self.param = param
        self.cv = param["cv"]
        self.nfolds = param["nfolds"]
        self.seeds = param["seeds"]
        self.feats = param["features"]
        self.drop_feats = param["drop_features"]
        self.label_encode = param["label_encode"]
        self.y = param["y"]
        self.flag = param["prepro_flag"]
        self.scale_flag = param["scale_flag"]
        self.scale = param["scale"]
        self.scale_param = param["scale_param"]
        
        self.ROOT = param["ROOT"] # */src
        self.WORK_DIR = param["WORK_DIR"]
        self.outdir = param["output_dir"] # my_features
        self.out_train_path = osp.join(self.ROOT, self.outdir, "train") # */src/my_features/train
        self.out_test_path = osp.join(self.ROOT, self.outdir, "test") # */src/my_features/test

        if "train" not in os.listdir(self.WORK_DIR): os.mkdir(osp.join(self.WORK_DIR, "train")) 
        if "valid" not in os.listdir(self.WORK_DIR): os.mkdir(osp.join(self.WORK_DIR, "valid")) 
        if "test" not in os.listdir(self.WORK_DIR): os.mkdir(osp.join(self.WORK_DIR, "test")) 

    def __call__(self):
        print("Preprocessing")
        if self.flag: 
            feat_classes = [eval(feat)(self.param) for feat in self.feats]
            for f_class in feat_classes:
                f_class.run()

            train_df, test_df = self.read_feature()
            train_df.to_csv(osp.join(self.WORK_DIR, "train", "train.csv"), index=False)
            test_df.to_csv(osp.join(self.WORK_DIR, "test", "test.csv"), index=False)
            if self.scale_flag:
                for col in train_df.columns:
                    if (train_df[col].dtype == float or "Count" in col or "count" in col) and "country" not in col and "Vec" not in col:
                        print(col)
                        sc = eval(self.scale)().fit(train_df.append(test_df)[col].values.reshape(-1, 1))
                        train_df[col] = sc.transform(train_df[col].values.reshape(-1, 1))
                        test_df[col] = sc.transform(test_df[col].values.reshape(-1, 1))

            print("label encode")
            for feat in self.label_encode:
                lbl_enc = LabelEncoder().fit(pd.concat([train_df[feat], test_df[feat]]))
                train_df[feat] = lbl_enc.transform(train_df[feat])
                test_df[feat] = lbl_enc.transform(test_df[feat])

            
            print("save data")
            cv_feats = [feat for feat in train_df.columns if self.cv in feat]
            #cv_feats = [f"{self.cv}_{seed}" for seed in self.seeds]
            use_cols = train_df.columns.to_list()
            for feat in self.drop_feats+cv_feats+[self.y]:
                if feat in use_cols: use_cols.remove(feat)
            pickle.dump(use_cols, open(osp.join(self.WORK_DIR, "features.pkl"), "wb"))

            for seed, cv_feat in zip(self.seeds, cv_feats):
                for fold in range(self.nfolds):
                    train = train_df[~(train_df[cv_feat] == fold)]
                    valid = train_df[train_df[cv_feat] == fold]
                    train_X = train[use_cols]
                    valid_X = valid[use_cols]
                    train_y = train[[self.y]]
                    valid_y = valid[[self.y]]
                    train_X.to_csv(osp.join(self.WORK_DIR, "train", f"train_X_{seed}_{fold}.csv"), index=False)
                    train_y.to_csv(osp.join(self.WORK_DIR, "train", f"train_y_{seed}_{fold}.csv"), index=False)
                    valid_X.to_csv(osp.join(self.WORK_DIR, "valid", f"valid_X_{seed}_{fold}.csv"), index=False)
                    valid_y.to_csv(osp.join(self.WORK_DIR, "valid", f"valid_y_{seed}_{fold}.csv"), index=False)
            
            test_X = test_df[use_cols]
            test_X.to_csv(osp.join(self.WORK_DIR, "test", f"test_X.csv"), index=False)


    def read_feature(self):
        train_feat_fnames = [osp.join(self.out_train_path, f"{feat}.feather") for feat in self.feats]
        test_feat_fnames = [osp.join(self.out_test_path, f"{feat}.feather") for feat in self.feats if feat != self.cv]
        train_df = pd.concat([pd.read_feather(fname) for fname in train_feat_fnames], axis=1)
        test_df = pd.concat([pd.read_feather(fname) for fname in test_feat_fnames], axis=1)
        return train_df, test_df
        
        
