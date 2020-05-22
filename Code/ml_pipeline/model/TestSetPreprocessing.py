import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from joblib import dump, load

import MLPipeline
from CompoundSimilarity import CompoundSimilarity
from ml_pipeline.settings import APP_STATIC

DATA_FLD_NAME = "step6"
TEST_FLD_NAME = "test"
PADEL_FLD_NAME = "fg_padel"
PADEL_FLD_RAW_NAME = "raw"
PADEL_FLD_PP_NAME = "preprocessed"

TEST_CMPNDS_FLD_NAME = "test_compounds"
TEST_CMPNDS_FILE_NAME = "test_compounds.csv"

PP_FLD = "step2"
PP_FIN_NAME = "PP_train.csv"
PP_NORM_NAME = "PP_data_normalization.joblib"

BORUTA_FLD = "step3"
BORUTA_FS_NAME = "FS_train.csv"

PCA_FLD = "step4"
PCA_MODEL = "FE_PCA.joblib"


class TestSetPreprocessing:

    def __init__(self, ml_pipeline: MLPipeline):
        print("Inside TestSetGeneration initialization")

        self.ml_pipeline = ml_pipeline

        if self.ml_pipeline.status == "test_set_generation":  # resuming at step 1
            self.preprocess_test_padel()

    def preprocess_test_padel(self):
        padel_raw_fld_path = os.path.join(
            *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, PADEL_FLD_NAME, PADEL_FLD_RAW_NAME])

        padel_pp_fld_path = os.path.join(
            *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, PADEL_FLD_NAME, PADEL_FLD_PP_NAME])

        padel_test_cmpnd_fld_path = os.path.join(
            *[self.ml_pipeline.job_data['job_data_path'], DATA_FLD_NAME, PADEL_FLD_NAME, TEST_CMPNDS_FLD_NAME])

        os.makedirs(padel_pp_fld_path, exist_ok=True)
        os.makedirs(padel_test_cmpnd_fld_path, exist_ok=True)

        for file in os.listdir(padel_raw_fld_path):
            print(file)

            if file.endswith(".csv"):  # checking only csv files
                padel_fp = os.path.join(padel_raw_fld_path, file)
                ligands, padel_pp_df = self.preprocess_now(padel_fp)

                ligands_df = pd.DataFrame(ligands, columns=["Ligand"])
                test_cmpnd_fp = os.path.join(padel_test_cmpnd_fld_path, file)
                ligands_df.to_csv(test_cmpnd_fp, index=False)

                padel_pp_fp = os.path.join(padel_pp_fld_path, file)
                padel_pp_df.to_csv(padel_pp_fp, index=False)

    def preprocess_now(self, padel_fp):
        df_test = pd.read_csv(padel_fp)
        print(df_test.columns)
        ligands = df_test['Ligand']

        print("Before shape test ", df_test.shape)

        df_init_train, init_features = self.extract_initial_train_features()
        df_init_test_fltrd = df_test[init_features]
        df_test_pp = self.apply_other_preprocess(df_init_train, df_init_test_fltrd)

        print("After preprocessing shape test ", df_test_pp.shape)

        df_fin_train, fin_features = self.extract_final_train_features()
        df_test_pp_final = df_test_pp[fin_features]

        print("After feature selection shape test ", df_test_pp_final.shape)

        test_final_np = self.apply_pca(df_test_pp_final)

        print("After feature extraction shape test ", test_final_np.shape)

        df_test_final = pd.DataFrame(test_final_np)

        return ligands, df_test_final

    def extract_initial_train_features(self):
        # TODO make sure final features files is there (a copy of final features with given naming convention)
        pp_train_path = os.path.join(
            *[self.ml_pipeline.job_data['job_data_path'], PP_FLD, "PP_init_train.csv"])
        df = pd.read_csv(pp_train_path)
        features = df.columns.to_list()

        return df, features

    def extract_final_train_features(self):
        fin_features = []
        df = None
        if self.ml_pipeline.config.fs_boruta_flg:
            boruta_train_path = os.path.join(
                *[self.ml_pipeline.job_data['job_data_path'], BORUTA_FLD, BORUTA_FS_NAME])
            df = pd.read_csv(boruta_train_path)
            fin_features = df.columns.to_list()
        else:
            # TODO make sure final features files is there (a copy of final features with given naming convention)
            pp_train_path = os.path.join(
                *[self.ml_pipeline.job_data['job_data_path'], PP_FLD, PP_FIN_NAME])
            df = pd.read_csv(pp_train_path)
            fin_features = df.columns.to_list()

        return df, fin_features

    def handle_missing_values(self, df_train, df_test_fltrd):
        if self.ml_pipeline.config.pp_mv_imputation_flg:
            impute_mthd = self.ml_pipeline.config.pp_mv_imputation_mthd

            X_train = df_train
            X_test = df_test_fltrd

            X_test = X_test.replace(r'\s+', np.nan, regex=True)
            X_test = X_test.replace(r'^\s*$', np.nan, regex=True)
            X_test[X_test == np.inf] = np.nan

            X_test = X_test.replace([np.inf, -np.inf, "", " "], np.nan)
            X_test = X_test.replace(["", " "], np.nan)

            if impute_mthd == 'mean':
                X_test.fillna(X_train.mean(), inplace=True)
                df_test_fltrd = X_test.copy()
            else:
                # TODO if not mean, handle other cases
                pass

        return df_test_fltrd

    def perform_data_normalization(self, df_test_fltrd):

        if self.ml_pipeline.config.pp_normalization_flg:

            testdata = df_test_fltrd

            if self.ml_pipeline.config.pp_normalization_mthd == 'min_max':
                print("Inside performing minmax normalization")

                pp_norm_model_path = os.path.join(
                    *[self.ml_pipeline.job_data['job_data_path'], PP_FLD, PP_NORM_NAME])
                print(pp_norm_model_path)

                min_max_scaler = load(pp_norm_model_path)

                # min_max_scaler = preprocessing.MinMaxScaler()

                test = testdata.values
                test = min_max_scaler.transform(test)
                test_normal = pd.DataFrame(test)
                test_normal.columns = list(testdata.columns.values)

                df_test_fltrd = test_normal
            else:
                # TODO if not minmax, handle other cases
                pass

        return df_test_fltrd

    def apply_pca(self, df_test_fltrd):
        if self.ml_pipeline.config.fe_pca_flg:
            xtest = df_test_fltrd

            pca_model_path = os.path.join(
                *[self.ml_pipeline.job_data['job_data_path'], PCA_FLD, PCA_MODEL])
            print(pca_model_path)

            pca = load(pca_model_path)

            print("Inside PCA, Before Shape Test: ", xtest.shape)
            xtest_new = pca.transform(xtest)
            print("Inside PCA, After Shape Test: ", xtest_new.shape)

            pca_op = xtest_new

        else:
            pca_op = df_test_fltrd.values

        return pca_op

    def apply_other_preprocess(self, df_train, df_test_fltrd):
        df_test_fltrd = self.handle_missing_values(df_train, df_test_fltrd)
        df_test_fltrd = self.perform_data_normalization(df_test_fltrd)
        return df_test_fltrd
