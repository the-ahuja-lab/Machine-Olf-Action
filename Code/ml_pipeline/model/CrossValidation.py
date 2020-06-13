import os

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeavePOut

import pickle

import MLPipeline
# import PostPreprocessing as ppp

import Preprocessing as ppp
import FeatureSelection as pfs
import FeatureExtraction as pfe

import AppConfig as app_config
import Evaluation

DATA_FLD_NAME = app_config.PP_FLD_NAME
DATA_FILE_NAME_PRFX = app_config.PP_FLD_PREFIX
MODEL_FLD = app_config.CLF_FLD_NAME
RESULTS_FLD_NAME = app_config.CV_RESULTS_FLD_NAME

ALL_MODELS = {}


class CrossValidation:

    def __init__(self, ml_pipeline: MLPipeline):
        self.ml_pipeline = ml_pipeline
        self.jlogger = self.ml_pipeline.jlogger

        self.jlogger.info(
            "Inside CrossValidation initialization with status {}".format(self.ml_pipeline.status))

        if self.ml_pipeline.status == app_config.STEP5_STATUS:  # resuming at step 5
            self.apply_on_all_fg()

    def apply_on_all_fg(self):

        if self.ml_pipeline.config.fg_padelpy_flg:
            self.jlogger.info("Started cross validation of preprocessed PaDEL features")
            pp_padel_fld_path = os.path.join(
                *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, app_config.FG_PADEL_FLD_NAME])

            self.fg_fld_name = app_config.FG_PADEL_FLD_NAME

            if self.ml_pipeline.config.pp_mv_col_pruning_flg:
                padel_cols_pruned_fp = os.path.join(pp_padel_fld_path, DATA_FILE_NAME_PRFX + "cols_pruned.csv")
                padel_labels_data_fp = os.path.join(pp_padel_fld_path, DATA_FILE_NAME_PRFX + "init_labels.csv")

                self.ml_pipeline.data = pd.read_csv(padel_cols_pruned_fp)
                self.ml_pipeline.data_labels = pd.read_csv(padel_labels_data_fp)

            else:
                padel_init_data_fp = os.path.join(pp_padel_fld_path, DATA_FILE_NAME_PRFX + "init_data.csv")
                padel_labels_data_fp = os.path.join(pp_padel_fld_path, DATA_FILE_NAME_PRFX + "init_labels.csv")

                self.ml_pipeline.data = pd.read_csv(padel_init_data_fp)
                self.ml_pipeline.data_labels = pd.read_csv(padel_labels_data_fp)

            self.apply_cv()

        if self.ml_pipeline.config.fg_mordered_flg:
            self.jlogger.info("Started cross validation of preprocessed mordred features")
            pp_mordred_fld_path = os.path.join(
                *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, app_config.FG_MORDRED_FLD_NAME])

            self.fg_fld_name = app_config.FG_MORDRED_FLD_NAME

            if self.ml_pipeline.config.pp_mv_col_pruning_flg:
                mordred_cols_pruned_fp = os.path.join(pp_mordred_fld_path, DATA_FILE_NAME_PRFX + "cols_pruned.csv")
                mordred_labels_data_fp = os.path.join(pp_mordred_fld_path, DATA_FILE_NAME_PRFX + "init_labels.csv")

                self.ml_pipeline.data = pd.read_csv(mordred_cols_pruned_fp)
                self.ml_pipeline.data_labels = pd.read_csv(mordred_labels_data_fp)
            else:
                mordred_init_data_fp = os.path.join(pp_mordred_fld_path, DATA_FILE_NAME_PRFX + "init_data.csv")
                mordred_labels_data_fp = os.path.join(pp_mordred_fld_path, DATA_FILE_NAME_PRFX + "init_labels.csv")

                self.ml_pipeline.data = pd.read_csv(mordred_init_data_fp)
                self.ml_pipeline.data_labels = pd.read_csv(mordred_labels_data_fp)

            self.apply_cv()

        self.jlogger.info("Cross-validation completed successfully")

    def apply_cv(self):
        self.apply_3fold_cv()
        self.apply_5fold_cv()
        self.apply_loocv()

    def get_data_splits(self, cv_method):
        cv_data_splits = []

        x = self.ml_pipeline.data.values
        y = self.ml_pipeline.data_labels.values.ravel()

        i = 1
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

            pp = ppp.Preprocessing(ppp_ml_pipeline, is_train=False)
            pp.preprocess_data()

            fs = pfs.FeatureSelection(ppp_ml_pipeline, is_train=False)
            fs.perform_feature_selection()

            fe = pfe.FeatureExtraction(ppp_ml_pipeline, is_train=False)
            fe.perform_feature_extraction()

            self.jlogger.info("Cross validation split number {}".format(i))
            self.jlogger.info("XTrain Shape: {}".format(ppp_ml_pipeline.x_train.shape))
            self.jlogger.info("XTest Shape: {}".format(ppp_ml_pipeline.x_test.shape))
            self.jlogger.info("YTrain Shape: {}".format(ppp_ml_pipeline.y_train.shape))
            self.jlogger.info("YTest Shape: {}".format(ppp_ml_pipeline.y_test.shape))

            cv_data_splits.append(
                (ppp_ml_pipeline.x_train, ppp_ml_pipeline.x_test, ppp_ml_pipeline.y_train, ppp_ml_pipeline.y_test))

            i += 1

        return cv_data_splits

    def evaluate_kfold_splits(self, k, cv_data_splits):
        # Fetch all classifiers first
        if len(ALL_MODELS) == 0:
            self.get_all_classifiers()

        evaluation = Evaluation.Evaluation(self.ml_pipeline)

        for model_name, clf in ALL_MODELS.items():
            # TODO check if can be moved to evaluation file
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

                self.jlogger.info("Cross validation split {} has evaluation results {}".format(i + 1, res))

            results = pd.DataFrame(res_list, columns=res_dict_keys)

            fld_path = os.path.join(
                *[self.ml_pipeline.job_data['job_results_path'], self.fg_fld_name, RESULTS_FLD_NAME,
                  str(k) + "Fold"])
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
            # TODO check if can be moved to evaluation file
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
            # fpr, tpr = evaluation.get_smoothened_fpr_tpr_from_pROC(gt, probs)
            # evaluation.plot_r_smoothened_curve(fpr, tpr, "ROC Curve - " + model_name)

            evaluation.print_roc_curve(np_ytrues, np_ypreds, np_yprobas, "ROC Curve - " + model_name)
            res = evaluation.evaluate_model(np_ypreds, np_ytrues, np_yprobas)

            res_list.append(list(res.values()))
            res_dict_keys = list(res.keys())

            self.jlogger.info("LOOCV result for model {} is {}".format(model_name, res))

        results = pd.DataFrame(res_list, columns=res_dict_keys, index=ALL_MODELS.keys())

        fld_path = os.path.join(
            *[self.ml_pipeline.job_data['job_results_path'], self.fg_fld_name, RESULTS_FLD_NAME, "LOOCV"])
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
            self.jlogger.info("Completed 3-Fold")

    def apply_5fold_cv(self):
        if self.ml_pipeline.config.cv_5fold_flg:
            skf = StratifiedKFold(n_splits=5, random_state=42)
            cv_data_splits = self.get_data_splits(skf)

            self.evaluate_kfold_splits(5, cv_data_splits)
            self.jlogger.info("Completed 5-Fold")

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
            self.jlogger.info("Completed LOOCV")

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
            *[self.ml_pipeline.job_data['job_data_path'], MODEL_FLD, self.fg_fld_name, model_name,
              "clf_" + model_name + ".pkl"])

        # TODO handle error incase file is not found
        with open(model_pkl_path, 'rb') as f:
            model = pickle.load(f)
            ALL_MODELS[model_name] = model

    def apply_gnb(self):
        if self.ml_pipeline.config.clf_gnb_flg:
            self.fetch_and_store_model("gnb")

            if self.ml_pipeline.config.clf_bagging_gnb:
                self.fetch_and_store_model("gnb_bagging")

    def apply_lr(self):
        if self.ml_pipeline.config.clf_lr_flg:
            self.fetch_and_store_model("lr")

            if self.ml_pipeline.config.clf_bagging_lr:
                self.fetch_and_store_model("lr_bagging")

    def apply_svm(self):
        if self.ml_pipeline.config.clf_svm_flg:
            self.fetch_and_store_model("svm")

            if self.ml_pipeline.config.clf_bagging_svm:
                self.fetch_and_store_model("svm_bagging")

    def apply_rf(self):
        if self.ml_pipeline.config.clf_rf_flg:
            self.fetch_and_store_model("rf")

            if self.ml_pipeline.config.clf_bagging_rf:
                self.fetch_and_store_model("rf_bagging")

    def apply_mlp(self):
        if self.ml_pipeline.config.clf_mlp_flg:
            self.fetch_and_store_model("mlp")

            if self.ml_pipeline.config.clf_bagging_mlp:
                self.fetch_and_store_model("mlp_bagging")

    def apply_et(self):
        if self.ml_pipeline.config.clf_et_flg:
            self.fetch_and_store_model("et")

            if self.ml_pipeline.config.clf_bagging_et:
                self.fetch_and_store_model("et_bagging")

    def apply_gbm(self):
        if self.ml_pipeline.config.clf_gbm_flg:
            self.fetch_and_store_model("gbm")

            if self.ml_pipeline.config.clf_bagging_gbm:
                self.fetch_and_store_model("gbm_bagging")
