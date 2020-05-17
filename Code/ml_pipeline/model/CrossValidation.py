import numpy as np
import pandas as pd
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeavePOut

import MLPipeline
import PostPreprocessing as ppp
import Evaluation

import pickle

DATA_FLD_NAME = "step2"
DATA_FILE_NAME_PRFX = "PP_"

MODEL_FLD = "step5"

RESULTS_FLD_NAME = "cross-validation"

ALL_MODELS = {}


class CrossValidation:

    def __init__(self, ml_pipeline: MLPipeline):
        print("Inside CrossValidation initialization")
        self.ml_pipeline = ml_pipeline

        # TODO read after column pruning
        padel_data_fp = os.path.join(ml_pipeline.job_data['job_data_path'], "step1", "FG_Padel.csv")

        data, data_labels = self.read_data(padel_data_fp)
        self.ml_pipeline.data = data
        self.ml_pipeline.data_labels = data_labels

        self.perform_column_pruning()

        self.apply_cv()

    def read_data(self, filepath):  # Input data should have one 1 column "Activation_Status" 1 column named "Ligand"
        data = pd.read_csv(filepath)
        data_labels = data["Activation Status"]
        Ligand_names = data["Ligand"]
        data = data.drop("Activation Status", axis=1)
        data = data.drop("Ligand", axis=1)
        try:
            data = data.drop("Smiles", axis=1)
        except:
            print("Don't have Smile's Column")
        data = self.coerce_df_columns_to_numeric(data, data.columns)
        return data, data_labels

    def coerce_df_columns_to_numeric(self, df, column_list):
        df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')
        return df

    def perform_column_pruning(self):
        if self.ml_pipeline.config.pp_mv_col_pruning_flg:
            th = self.ml_pipeline.config.pp_mv_col_pruning_th

            data = self.ml_pipeline.data

            data = data.replace(r'\s+', np.nan, regex=True)
            data[data == np.inf] = np.nan
            data = data.replace(r'^\s*$', np.nan, regex=True)

            NAN_fld_path = self.ml_pipeline.job_data['job_data_path']
            NAN_fld_path = os.path.join(NAN_fld_path, DATA_FLD_NAME)

            NAN_file_path = os.path.join(NAN_fld_path, DATA_FILE_NAME_PRFX + "NAN_values1.csv")

            NAN_data = pd.read_csv(NAN_file_path, header=None)
            dropped = []
            for i in range(len(NAN_data)):
                if NAN_data.iloc[i][1] >= th:
                    dropped.append(NAN_data.iloc[i][0])
            data = data.drop(dropped, axis=1)

            print("Dropped columns: ", len(dropped))
            print("Data shape after pruning NAN values: ", data.shape)

            self.ml_pipeline.data = data

    def apply_cv(self):
        self.apply_3fold_cv()
        self.apply_5fold_cv()
        self.apply_loocv()

    def get_data_splits(self, cv_method):
        cv_data_splits = []

        x = self.ml_pipeline.data.values
        y = self.ml_pipeline.data_labels.values

        i = 0
        for train_index, test_index in cv_method.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            x_train_pd = pd.DataFrame(x_train)
            x_test_pd = pd.DataFrame(x_test)

            ppp_ml_pipeline = MLPipeline.MLPipeline(self.ml_pipeline.job_id)
            ppp_ml_pipeline.x_train = x_train_pd
            ppp_ml_pipeline.y_train = y_train
            ppp_ml_pipeline.x_test = x_test_pd
            ppp_ml_pipeline.y_test = y_test

            ppp.PostPreprocessing(ppp_ml_pipeline)

            print(ppp_ml_pipeline.x_train.shape)
            print(ppp_ml_pipeline.x_test.shape)
            print(ppp_ml_pipeline.y_train.shape)
            print(ppp_ml_pipeline.y_test.shape)

            cv_data_splits.append(
                (ppp_ml_pipeline.x_train, ppp_ml_pipeline.x_test, ppp_ml_pipeline.y_train, ppp_ml_pipeline.y_test))

            print("Split ", i)
            i += 1

        return cv_data_splits

    def evaluate_kfold_splits(self, k, cv_data_splits):
        # Fetch all classifiers first
        if len(ALL_MODELS) == 0:
            self.get_all_classifiers()

        evaluation = Evaluation.Evaluation(self.ml_pipeline)

        for model_name, clf in ALL_MODELS.items():
            res_list = []
            iters = []
            res_dict_keys = {}

            i = 0
            for x_train, x_test, y_train, y_test in cv_data_splits:
                clf.fit(x_train, y_train)

                ypred = clf.predict(x_test)

                yproba = clf.predict_proba(x_test)

                res = evaluation.evaluate_model(ypred, y_test, yproba)

                res_list.append(list(res.values()))
                res_dict_keys = list(res.keys())
                iters.append(i + 1)

                i += 1
                print(i)
                print(res)

            results = pd.DataFrame(res_list, columns=res_dict_keys)

            fld_path = os.path.join(
                *[self.ml_pipeline.job_data['job_results_path'], RESULTS_FLD_NAME, str(k) + "Fold"])
            os.makedirs(fld_path, exist_ok=True)

            file_name = model_name + "_" + str(k) + "_fold.csv"

            cv_result_fp = os.path.join(fld_path, file_name)

            results.to_csv(cv_result_fp, float_format='%.4f')

    def evaluate_loocv_splits(self, loocv_data_splits):
        if len(ALL_MODELS) == 0:
            self.get_all_classifiers()

        evaluation = Evaluation.Evaluation(self.ml_pipeline)

        res_list = []
        res_dict_keys = {}

        for model_name, clf in ALL_MODELS.items():

            all_ypreds = []
            all_ytrues = []
            all_yprobas = []

            for x_train, x_test, y_train, y_test in loocv_data_splits:
                clf.fit(x_train, y_train)

                ypred = clf.predict(x_test)
                all_ypreds.append(ypred[0])
                all_ytrues.append(y_test[0])

                yproba = clf.predict_proba(x_test)
                all_yprobas.append(yproba[0])

            np_ypreds = np.array(all_ypreds)
            np_ytrues = np.array(all_ytrues)
            np_yprobas = np.array(all_yprobas)

            # print(np_ypreds)
            # print(np_ytrues)
            # print(np_yprobas)

            # df = pd.DataFrame([], columns=['Prediction', 'Ground Truth', 'Prob 0', 'Prob 1'])
            # df['Prediction'] = np_ypreds
            # df['Ground Truth'] = np_ytrues
            # df['Prob 0'] = np_yprobas[:, 0]
            # df['Prob 1'] = np_yprobas[:, 1]

            evaluation.print_confusion_matrix(np_ytrues, np_ypreds, "Confusion Matrix - " + model_name)

            gt = np_ytrues.tolist()
            probs = np_yprobas[:, 1].tolist()
            fpr, tpr = evaluation.get_smoothened_fpr_tpr_from_pROC(gt, probs)
            evaluation.plot_r_smoothened_curve(fpr, tpr, "ROC Curve - " + model_name)

            evaluation.print_roc_curve(np_ytrues, np_ypreds, np_yprobas, "ROC Curve - " + model_name)
            res = evaluation.evaluate_model(np_ypreds, np_ytrues, np_yprobas)

            res_list.append(list(res.values()))
            res_dict_keys = list(res.keys())

            print(model_name, res)

        results = pd.DataFrame(res_list, columns=res_dict_keys, index=ALL_MODELS.keys())

        fld_path = os.path.join(
            *[self.ml_pipeline.job_data['job_results_path'], RESULTS_FLD_NAME, "LOOCV"])
        os.makedirs(fld_path, exist_ok=True)

        loocv_result_fp = os.path.join(fld_path, "LOOCV_ALL_CLASSIFIERS.csv")

        results.to_csv(loocv_result_fp, float_format='%.4f')

        loocv_pdf_fp = os.path.join(fld_path, "LOOCV.pdf")

        evaluation.plot_all_figures(loocv_pdf_fp)

    def apply_3fold_cv(self):
        if self.ml_pipeline.config.cv_3fold_flg:
            skf = StratifiedKFold(n_splits=3, random_state=42)
            cv_data_splits = self.get_data_splits(skf)

            self.evaluate_kfold_splits(3, cv_data_splits)
            print("Completed 3-Fold")

    def apply_5fold_cv(self):
        if self.ml_pipeline.config.cv_5fold_flg:
            skf = StratifiedKFold(n_splits=5, random_state=42)
            cv_data_splits = self.get_data_splits(skf)

            self.evaluate_kfold_splits(5, cv_data_splits)
            print("Completed 5-Fold")

    def apply_loocv(self):
        if self.ml_pipeline.config.cv_loocv_flg:
            loo = LeavePOut(1)
            cv_data_splits = self.get_data_splits(loo)

            # loocv_pkl = os.path.join(
            #     *[self.ml_pipeline.job_data['job_results_path'], RESULTS_FLD_NAME, "loocv_preprocess.pkl"])
            #
            # cv_data_splits = None
            # with open(loocv_pkl, 'rb') as f:
            #     cv_data_splits = pickle.load(f)

            self.evaluate_loocv_splits(cv_data_splits)

    def get_all_classifiers(self):
        self.apply_gbm()
        self.apply_svm()
        self.apply_rf()
        self.apply_lr()
        self.apply_gnb()
        self.apply_et()
        self.apply_mlp()

    def fetch_and_store_model(self, model_name):
        model_pkl_path = os.path.join(
            *[self.ml_pipeline.job_data['job_data_path'], MODEL_FLD, model_name, "clf_" + model_name + ".pkl"])

        # TODO handle error incase file is not found
        with open(model_pkl_path, 'rb') as f:
            model = pickle.load(f)
            ALL_MODELS[model_name] = model

    def apply_gnb(self):
        if self.ml_pipeline.config.clf_gnb_flg:
            self.fetch_and_store_model("gnb")

    def apply_lr(self):
        if self.ml_pipeline.config.clf_lr_flg:
            self.fetch_and_store_model("lr")

    def apply_svm(self):
        if self.ml_pipeline.config.clf_svm_flg:
            self.fetch_and_store_model("svm")

    def apply_rf(self):
        if self.ml_pipeline.config.clf_rf_flg:
            self.fetch_and_store_model("rf")

    def apply_mlp(self):
        if self.ml_pipeline.config.clf_mlp_flg:
            self.fetch_and_store_model("mlp")

    def apply_et(self):
        if self.ml_pipeline.config.clf_et_flg:
            self.fetch_and_store_model("et")

    def apply_gbm(self):
        if self.ml_pipeline.config.clf_gbm_flg:
            self.fetch_and_store_model("gbm")

    #TODO add bagging models also here
