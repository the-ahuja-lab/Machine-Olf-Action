import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from joblib import dump, load

import MLPipeline

import LIMEExplanation

from CompoundSimilarity import CompoundSimilarity
from ml_pipeline.settings import APP_STATIC

import pickle

DATA_FLD_NAME = "step6"
TEST_FLD_NAME = "test"
PADEL_FLD_NAME = "fg_padel"
PADEL_FLD_PP_NAME = "preprocessed"
TEST_CMPNDS_FLD_NAME = "test_compounds"
TEST_CMPNDS_FILE_NAME = "test_compounds.csv"

RESULTS_FLD_NAME = "novel_predictions"

MODEL_FLD = "step5"

ALL_TEST_DF = {}
ALL_TEST_COMPOUNDS = {}


class TestSetPrediction:

    def __init__(self, ml_pipeline: MLPipeline):
        print("Inside TestSetPrediction initialization")

        self.ml_pipeline = ml_pipeline

        self.lime_exp = None

        # TODO change status from test_set_generation to test_set_prediction
        if self.ml_pipeline.status == "test_set_generation":  # resuming at step 1
            self.initialize_lime_explanation()
            self.apply_classification_models()

    def initialize_lime_explanation(self):
        lime_ml_pipeline = MLPipeline.MLPipeline(self.ml_pipeline.job_id)
        lime_ml_pipeline.status = "test_set_generation"
        self.lime_exp = LIMEExplanation.LIMEExplanation(lime_ml_pipeline)

    def apply_classification_models(self):
        self.apply_gbm()
        self.apply_svm()
        self.apply_rf()
        self.apply_lr()
        self.apply_gnb()
        self.apply_et()
        self.apply_mlp()

    def load_all_test_files(self):
        if len(ALL_TEST_DF) == 0:

            padel_pp_fld_path = os.path.join(
                *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, PADEL_FLD_NAME, PADEL_FLD_PP_NAME])

            test_cmpnd_fld_path = os.path.join(
                *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, PADEL_FLD_NAME, TEST_CMPNDS_FLD_NAME])

            # all_test_compounds_df = pd.read_csv(test_cmpnd_fld_path)
            # ALL_TEST_COMPOUNDS = all_test_compounds_df[all_test_compounds_df.columns[0]]

            for file in os.listdir(padel_pp_fld_path):
                print(file)

                if file.endswith(".csv"):  # checking only csv files
                    padel_test_fp = os.path.join(padel_pp_fld_path, file)
                    df = pd.read_csv(padel_test_fp)
                    ALL_TEST_DF[file] = df

            for file in os.listdir(test_cmpnd_fld_path):
                print(file)

                if file.endswith(".csv"):  # checking only csv files
                    cmpnd_name_fp = os.path.join(test_cmpnd_fld_path, file)
                    df = pd.read_csv(cmpnd_name_fp)
                    ALL_TEST_COMPOUNDS[file] = df[df.columns[0]]

        return ALL_TEST_DF, ALL_TEST_COMPOUNDS

    def apply_model_for_predictions(self, model, df, all_test_compounds):

        padel_test_df = df.copy()

        pred_labels = model.predict(padel_test_df)
        try:
            prob = model.predict_proba(padel_test_df)
        except:
            print("This model does not support probability scores")
            prob = []
        padel_test_df['Ligand'] = all_test_compounds

        novel_compounds_predictions = pd.DataFrame(list(zip(padel_test_df['Ligand'], pred_labels, prob)),
                                                   columns=['Compound Name', 'Predicted Label',
                                                            'Predicted probability'])
        return novel_compounds_predictions

    def fetch_model_save_predictions(self, model_name):
        model_pkl_path = os.path.join(
            *[self.ml_pipeline.job_data['job_data_path'], MODEL_FLD, model_name, "clf_" + model_name + ".pkl"])

        with open(model_pkl_path, 'rb') as f:
            model = pickle.load(f)

        all_test_df, all_test_compounds = self.load_all_test_files()

        self.lime_exp.lime_explainer = None

        for padel_fname, test_df in all_test_df.items():
            test_compounds = all_test_compounds[padel_fname]
            novel_compounds_predictions = self.apply_model_for_predictions(model, test_df, test_compounds)

            novel_pred_fld_p = os.path.join(
                *[self.ml_pipeline.job_data['job_results_path'], RESULTS_FLD_NAME, model_name])
            os.makedirs(novel_pred_fld_p, exist_ok=True)

            # TODO Add model name in prediction file
            pred_f_name = "pred_" + model_name + "_" + padel_fname
            novel_pred_fp = os.path.join(novel_pred_fld_p, pred_f_name)

            novel_compounds_predictions.to_csv(novel_pred_fp, index=False)

            lime_exp_f_name = "lime_exp_" + model_name + "_" + self.change_ext(padel_fname, ".csv", ".pdf")
            lime_exp_pdf_fp = os.path.join(novel_pred_fld_p, lime_exp_f_name)

            self.lime_exp.exp_preds_using_lime(model, test_compounds, padel_fname, lime_exp_pdf_fp)

    def change_ext(self, fname, ext_init, ext_fin):
        if fname.endswith(ext_init):
            return fname.replace(ext_init, ext_fin)
        else:
            return fname + ext_fin

    def apply_gnb(self):
        if self.ml_pipeline.config.clf_gnb_flg:
            self.fetch_model_save_predictions("gnb")

        if self.ml_pipeline.config.clf_bagging_gnb:
            self.fetch_model_save_predictions("gnb_bagging")

    def apply_gbm(self):
        if self.ml_pipeline.config.clf_gbm_flg:
            self.fetch_model_save_predictions("gbm")

        if self.ml_pipeline.config.clf_bagging_gbm:
            self.fetch_model_save_predictions("gbm_bagging")

    def apply_svm(self):
        if self.ml_pipeline.config.clf_svm_flg:
            self.fetch_model_save_predictions("svm")

        if self.ml_pipeline.config.clf_bagging_svm:
            self.fetch_model_save_predictions("svm_bagging")

    def apply_lr(self):
        if self.ml_pipeline.config.clf_lr_flg:
            self.fetch_model_save_predictions("lr")

        if self.ml_pipeline.config.clf_bagging_lr:
            self.fetch_model_save_predictions("lr_bagging")

    def apply_rf(self):
        if self.ml_pipeline.config.clf_rf_flg:
            self.fetch_model_save_predictions("rf")

        if self.ml_pipeline.config.clf_bagging_rf:
            self.fetch_model_save_predictions("rf_bagging")

    def apply_et(self):
        if self.ml_pipeline.config.clf_et_flg:
            self.fetch_model_save_predictions("et")

        if self.ml_pipeline.config.clf_bagging_et:
            self.fetch_model_save_predictions("et_bagging")

    def apply_mlp(self):
        if self.ml_pipeline.config.clf_mlp_flg:
            self.fetch_model_save_predictions("mlp")

        if self.ml_pipeline.config.clf_bagging_mlp:
            self.fetch_model_save_predictions("mlp_bagging")
