import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from joblib import dump, load

import MLPipeline
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
ALL_TEST_COMPOUNDS = None


class TestSetPrediction:

    def __init__(self, ml_pipeline: MLPipeline):
        print("Inside TestSetGeneration initialization")

        self.ml_pipeline = ml_pipeline

        if self.ml_pipeline.status == "test_set_preprocessing":  # resuming at step 1
            self.apply_classification_models()

    def apply_classification_models(self):
        # self.apply_gbm()
        # self.apply_svm()
        # self.apply_rf()
        # self.apply_lr()
        self.apply_gnb()
        # self.apply_et()
        # self.apply_mlp()

    def load_all_test_files(self):
        if len(ALL_TEST_DF) == 0:

            padel_pp_fld_path = os.path.join(
                *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, PADEL_FLD_NAME, PADEL_FLD_PP_NAME])

            test_cmpnd_fld_path = os.path.join(
                *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, PADEL_FLD_NAME, TEST_CMPNDS_FLD_NAME,
                  TEST_CMPNDS_FILE_NAME])

            all_test_compounds_df = pd.read_csv(test_cmpnd_fld_path)
            ALL_TEST_COMPOUNDS = all_test_compounds_df[all_test_compounds_df.columns[0]]

            for file in os.listdir(padel_pp_fld_path):
                print(file)

                if file.endswith(".csv"):  # checking only csv files
                    padel_test_fp = os.path.join(padel_pp_fld_path, file)
                    df = pd.read_csv(padel_test_fp)
                    ALL_TEST_DF[file] = df

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

    def apply_gnb(self):

        if self.ml_pipeline.config.clf_gnb_flg:

            model_name = "gnb"
            model_pkl_path = os.path.join(
                *[self.ml_pipeline.job_data['job_data_path'], MODEL_FLD, model_name, "clf_" + model_name + ".pkl"])

            with open(model_pkl_path, 'rb') as f:
                model = pickle.load(f)

            all_test_df, all_test_compounds = self.load_all_test_files()

            for padel_fname, test_df in all_test_df.items():
                novel_compounds_predictions = self.apply_model_for_predictions(model, test_df, all_test_compounds)

                novel_pred_fld_p = os.path.join(
                    *[self.ml_pipeline.job_data['job_results_path'], RESULTS_FLD_NAME, model_name])
                os.makedirs(novel_pred_fld_p, exist_ok=True)

                # TODO Add model name in prediction file
                novel_pred_fp = os.path.join(novel_pred_fld_p, padel_fname)

                novel_compounds_predictions.to_csv(novel_pred_fp, index=False)

    #TODO add other classifiers also here
