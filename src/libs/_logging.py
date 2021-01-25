import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import os.path as osp
import glob
from sklearn.metrics import f1_score
import pickle
from utils import optimized_f1

import mlflow
from google.cloud import storage
TH = 0.389

class Logging():
    def __init__(self, param):
        self.param = param
        self.ROOT = param["exp_param"]["ROOT"]
        self.WORK_DIR = param["exp_param"]["WORK_DIR"]
        self.val_pred_path = osp.join(self.WORK_DIR, "val_preds")
        self.weight_path = osp.join(self.WORK_DIR, "weight")

        self.feats = None
        self.model_param = param["train_param"]["model_param"]
        self.cv_score = None
        self.cv_scores = None
        self.feature_importances_fname = None
        self.submission_fname = osp.join(self.WORK_DIR, "submission.csv")
        self.gcs_flag = param["exp_param"]["gcs_flag"]

        self.seeds = param["exp_param"]["seeds"]
        self.nfolds = param["exp_param"]["nfolds"]
        self.y = param["exp_param"]["y"]
        self.cv = param["exp_param"]["cv"]
        self.flag = param["exp_param"]["log_flag"]
        self.exp_name = param["exp_param"]["exp_name"]
        self.bts = []

    def __call__(self):
        print("Logging")
        if self.flag:
            # cvの計算, seedごと、平均
            # 使った特徴量
            # feature_importanceの計算、描画
            # モデルのパラメータ
            # 結果の解釈
            # をmlflowにあげる
            self.feats = pickle.load(open(osp.join(self.WORK_DIR, "features.pkl"), 'rb'))
            self.cv_score, self.cv_scores = self.calc_cv()
            self.create_feature_importances()
            self.make_submission()


            mlflow.set_tracking_uri(osp.join(self.WORK_DIR, "mlruns"))
            self.create_mlflow()
            if self.gcs_flag: 
                self.upload_gcs()
        
        

    def calc_cv(self):
        preds = []
        cv_scores = []
        train_y = pd.read_csv(osp.join(self.WORK_DIR, "train", "train.csv"))
        
        for seed in self.seeds:
            cv_feat = f"{self.cv}_{seed}"
            mask = train_y[cv_feat] != -1
            train_y["pred"] = np.nan
            for fold in range(self.nfolds):
                val_preds = pd.read_csv(osp.join(self.val_pred_path, f"preds_{seed}_{fold}.csv"))
                train_y["pred"][train_y[cv_feat] == fold] = val_preds["pred"].values
            train_y[["pred"]].to_csv(osp.join(self.val_pred_path, f"oof_preds_{seed}.csv"), index=False)  
            cv_score, bt = optimized_f1(train_y[mask][self.y], train_y[mask]["pred"])
            self.bts.append(bt)
            cv_scores.append(cv_score)
            print(f"seed {seed}, cv : {cv_score}, beat threshold : {bt}")
            preds.append(train_y["pred"].values.copy()) # copy!!!!!
        preds = np.mean(np.array(preds), axis=0).reshape(-1,)
        preds = pd.DataFrame(preds, columns=["pred"])
        preds.to_csv(osp.join(self.val_pred_path, "oof_preds.csv"), index=False)
        try:
            cv_score, self.bt = optimized_f1(train_y[mask][self.y], preds["pred"])
        except:
            cv_score = np.mean(cv_scores)
            print("mean cv")
        self.bt = np.mean(np.array(self.bts))
            
        print(f"final cv : {cv_score}, best threshold : {self.bt}")
        return cv_score, cv_scores

        
    def create_feature_importances(self):
        try:
            models = []
            for seed in self.seeds:
                for fold in range(5):
                    p = osp.join(self.weight_path, f"{seed}_{fold}.pkl")
                    models.append(pickle.load(open(p, 'rb')))
            self.feature_importances_fname = osp.join(self.WORK_DIR, "feature_importances.png")
            self.visualize_importance(models, self.feats, self.feature_importances_fname)
        except:
            pass

    def visualize_importance(self, models, feats, save_fname):
        feature_importance_df = pd.DataFrame()
        for i, model in enumerate(models):
            _df = pd.DataFrame()
            try:
                _df['feature_importance'] = model.feature_importance()
            except:
                _df['feature_importance'] = model.get_feature_importance()
            _df['column'] = feats
            _df['fold'] = i + 1
            feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

        order = feature_importance_df.groupby('column')\
            .sum()[['feature_importance']]\
            .sort_values('feature_importance', ascending=False).index

        fig, ax = plt.subplots(figsize=(max(6, len(order) * .4), 7))
        sns.boxenplot(data=feature_importance_df, x='column', y='feature_importance', order=order, ax=ax, palette='viridis')
        ax.tick_params(axis='x', rotation=90)
        ax.grid()
        fig.tight_layout()
        plt.savefig(save_fname)
        return fig, ax
    
    def create_mlflow(self):
        with mlflow.start_run():
            mlflow.log_param("exp_name", self.exp_name)
            mlflow.log_param("model_param", self.model_param)
            mlflow.log_param("features", self.feats)
            mlflow.log_param("seeds", self.seeds)
            mlflow.log_param("nfolds", self.nfolds)
            mlflow.log_param("cv_type", self.cv)
            mlflow.log_param("cv_scores", self.cv_scores)

            mlflow.log_metric("cv_score", self.cv_score)
            #log_metric("cv_scores", self.cv_scores)

            mlflow.log_artifact(self.feature_importances_fname)
            mlflow.log_artifact(self.submission_fname)

    def upload_gcs(self):
        PROJECT_ID = 'signateJR2020'
        BUCKET_NAME = 'signate-jr2020'

        os.environ["GCLOUD_PROJECT"] = PROJECT_ID
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = osp.join(self.ROOT, "signateJR2020-d11d5341dabe.json")
        storage_client = storage.Client(project=PROJECT_ID)
        
        blobs = storage_client.list_blobs(BUCKET_NAME) # storageのディレクトリ

        files = [f for f in glob.glob(osp.join(self.WORK_DIR, "mlruns", "**"), recursive=True)]
        files_in_bucket = [f.name for f in storage_client.list_blobs(BUCKET_NAME)]
        bucket = storage_client.get_bucket(BUCKET_NAME)
        for f in files:
            try:
                if f not in files_in_bucket:
                    print(f.replace(self.WORK_DIR+"/", ""))
                    blob = bucket.blob(f.replace(self.WORK_DIR+"/", ""))
                    blob.upload_from_filename(f)
            except:
                pass
    
    def make_submission(self):
        test = pd.read_csv(osp.join(self.ROOT, "input", "test.csv"), usecols=["id"])
        pred = pd.read_csv(osp.join(self.WORK_DIR, "preds", "pred.csv"))
        print(test.shape)
        print(pred.shape)
        sub_df = pd.concat([test, pred], axis=1)
        sub_df["pred1"] = np.where(sub_df["pred"].values.copy() < self.bt, 0, 1)
        sub_df["pred2"] = np.where(sub_df["pred"].values.copy() < TH, 0, 1)
        sub_df[["id", "pred1"]].to_csv(osp.join(self.WORK_DIR, "submission.csv"), index=False, header=False)
        sub_df[["id", "pred2"]].to_csv(osp.join(self.WORK_DIR, "submission2.csv"), index=False, header=False)
